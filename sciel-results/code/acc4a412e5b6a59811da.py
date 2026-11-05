"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute vertex priorities for a greedy Maximum Independent Set decoder.
    
    The greedy decoder picks vertices in descending score order, skipping
    neighbours of already-picked vertices. We want to maximize the final
    independent set size.
    
    Strategy:
    1. Hub-avoidance: deprioritize vertices adjacent to high-degree hubs.
    2. Ensemble-based dynamic degree: simulate multiple randomized greedy
       passes to capture how degrees change as vertices are removed, giving
       a more informed static priority.
    3. Combine static structural features with ensemble statistics.
    
    All within a strict time budget (~145s safety margin for 150s kill).
    """
    deadline = time.time() + 145.0  # safety margin
    
    A = np.asarray(adjacency, dtype=np.int32)
    n = A.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Basic features
    degree = A.sum(axis=1).astype(float)
    
    # --- Static hub-avoidance score ---
    # For each vertex i, compute sum of 1/degree[j] over neighbors j.
    # Vertices adjacent to hubs (low 1/deg) get penalized.
    # Actually, we want to AVOID being adjacent to hubs, so vertices with
    # neighbors that are hubs should be deprioritized.
    # hub_penalty[i] = sum over neighbors j of degree[j]
    # High hub_penalty means i is near many hubs -> deprioritize i.
    
    # Also: vertices with low degree are generally better to keep.
    # But a low-degree vertex next to a hub is still bad if the hub gets picked.
    
    # Let's compute:
    # score_static[i] = -alpha * degree[i] - beta * hub_penalty[i] + gamma * (1/degree[i]+eps)
    
    eps = 1e-6
    inv_degree = 1.0 / (degree + eps)
    
    # hub_penalty: for each vertex, sum of neighbor degrees
    hub_penalty = (A @ degree)  # A is symmetric adjacency, A @ degree gives sum of neighbor degrees
    
    # Normalize features
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    norm_deg = normalize(degree)
    norm_hub = normalize(hub_penalty)
    norm_inv_deg = normalize(inv_degree)
    
    # Static score: prefer low degree, low hub-adjacency, high inverse degree
    alpha = 1.0
    beta = 0.5
    gamma = 0.5
    score_static = -alpha * norm_deg - beta * norm_hub + gamma * norm_inv_deg
    
    # --- Ensemble-based dynamic scoring ---
    # Simulate multiple randomized greedy passes. In each pass, at each step,
    # among available vertices, pick one with probability proportional to 
    # some weight (e.g., inverse current degree). Track how often each vertex
    # gets picked and at what "rank" (step number).
    
    # Number of ensemble simulations: scale with available time
    # Each simulation is O(n^2) worst case but typically faster.
    # Let's estimate: n=200, each sim maybe ~200 steps * O(n) = 40k ops.
    # We can afford ~5000 sims in 10 seconds. Let's target ~3-5 seconds for ensemble.
    
    ensemble_deadline = time.time() + 10.0  # cap ensemble at 10s
    
    # Precompute neighbor lists for faster simulation
    neighbors = [np.where(A[i] > 0)[0] for i in range(n)]
    
    num_sims = 0
    pick_counts = np.zeros(n, dtype=np.float64)
    step_sum = np.zeros(n, dtype=np.float64)  # sum of step indices when picked
    pick_steps = np.zeros((n,), dtype=np.float64)  # count of times picked at each step
    
    # We'll track: for each vertex, how many times it was picked, and the
    # average step at which it was picked. Earlier pick = higher priority.
    
    # Also track: how many times a vertex was "blocked" (skipped because a
    # neighbor was picked first). This indicates it's often a victim.
    
    blocked_counts = np.zeros(n, dtype=np.float64)
    
    # Randomized greedy simulation
    # At each step, among available (not picked, not blocked) vertices,
    # pick with probability proportional to w(v) = 1/(current_degree(v)+1)
    # current_degree = number of still-available neighbors
    
    # For efficiency with n=200, we can do this naively.
    
    max_sims = 5000  # cap to avoid overrun
    
    rng = np.random.RandomState(42)
    
    for sim in range(max_sims):
        if time.time() > ensemble_deadline:
            break
        if time.time() > deadline - 30:  # hard safety
            break
        
        # State: 0=available, 1=picked, 2=blocked
        state = np.zeros(n, dtype=np.int8)
        # Current degree: number of available neighbors
        cur_deg = degree.copy()
        
        # Initialize available set
        available = set(range(n))
        
        # We need to track which vertices are available for fast lookup
        # For n=200, a set is fine.
        
        step = 0
        while available:
            if time.time() > ensemble_deadline:
                break
            
            # Compute weights for available vertices
            avail_list = list(available)
            k = len(avail_list)
            
            if k == 0:
                break
            
            # Weight: inverse of current degree + 1
            weights = np.array([1.0 / (cur_deg[v] + 1.0) for v in avail_list])
            total_w = weights.sum()
            if total_w < 1e-12:
                # Uniform
                weights = np.ones(k)
                total_w = k
            
            probs = weights / total_w
            idx = rng.choice(k, p=probs)
            v = avail_list[idx]
            
            # Pick v
            state[v] = 1
            available.discard(v)
            pick_counts[v] += 1
            step_sum[v] += step
            
            # Block neighbors of v
            for nb in neighbors[v]:
                if state[nb] == 0:
                    state[nb] = 2
                    available.discard(nb)
                    blocked_counts[nb] += 1
                    cur_deg[nb] -= 1  # v is no longer available
            
            # Update current degrees of remaining available vertices
            # For each available vertex, v was a neighbor if A[v, u] > 0
            # We already handled neighbors of v above. But we also need to
            # decrement cur_deg for available vertices that are neighbors of v.
            # Actually, we did: for nb in neighbors[v], if state[nb]==0, we
            # set state[nb]=2 and cur_deg[nb]-=1. But for available vertices
            # that are neighbors of v, they get blocked. For non-neighbor
            # available vertices, their degree doesn't change.
            # Wait: cur_deg[u] should be the number of AVAILABLE neighbors of u.
            # When v is picked, for each neighbor u of v:
            #   if u is available, u becomes blocked, and cur_deg[u] decreases by 1 (v removed)
            #   but also, u is no longer available, so for other available vertices w
            #   that are neighbors of u, their cur_deg should decrease by 1.
            # This is getting complex. Let me simplify: recompute cur_deg for
            # available vertices after each pick. With n=200, this is O(n^2)
            # per step, which is too slow for 5000 sims.
            
            # Alternative: don't track cur_deg dynamically. Use static degree
            # for weighting. This is simpler and still captures the ensemble
            # effect of tie-breaking.
            
            step += 1
        
        num_sims += 1
    
    # If no simulations ran, fall back to static
    if num_sims == 0:
        return score_static
    
    # Ensemble features:
    # 1. pick_frequency: how often each vertex was picked (higher = more central in greedy order)
    # 2. avg_step: average step at which picked (lower = earlier = higher priority)
    # 3. block_frequency: how often blocked (higher = more often a victim)
    
    pick_freq = pick_counts / num_sims
    avg_step = np.where(pick_counts > 0, step_sum / np.maximum(pick_counts, 1), n)
    block_freq = blocked_counts / num_sims
    
    # Normalize
    norm_pick_freq = normalize(pick_freq)
    norm_avg_step = normalize(avg_step)  # lower is better
    norm_block_freq = normalize(block_freq)  # higher is worse
    
    # Ensemble score: prefer high pick frequency, low avg step, low block frequency
    score_ensemble = (
        1.0 * norm_pick_freq
        - 1.0 * norm_avg_step
        - 0.5 * norm_block_freq
    )
    
    # Combine static and ensemble
    # Weight ensemble more heavily since it captures dynamics
    w_static = 0.3
    w_ensemble = 0.7
    final_score = w_static * score_static + w_ensemble * score_ensemble
    
    # Add small random tie-breaker to avoid pathological ties
    # Actually, the evaluator handles ties; we should be deterministic.
    # But tiny noise can help break symmetric ties in favor of better outcomes.
    # Let's add a very small deterministic perturbation based on vertex index.
    # No, keep it deterministic.
    
    return final_score
# EVOLVE-BLOCK-END
