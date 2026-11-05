import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Simulated Annealing on Vertex Permutation with Bitset Acceleration
    1. Represent the greedy decoder state using Python integers as bitsets for O(1) neighbor removal.
    2. Initialize the permutation using a heuristic (low degree, high neighbor degree).
    3. Use Simulated Annealing with swap moves to optimize the permutation directly.
    4. Each move is evaluated by a fast bitset-based greedy simulation.
    5. Return the scores corresponding to the best permutation found (rank-based scores).
    """
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor bitsets
    # adj is n x n boolean. Convert rows to Python integers.
    # Bit i is 1 if vertex i is a neighbor.
    # We need to be careful with bit order. Let's say bit i corresponds to vertex i.
    # Python ints are arbitrary precision.
    # adj[i] is a boolean array of length n.
    # We can construct the int by iterating or using a vectorized trick if possible, 
    # but for n=200, a loop is fine and clear.
    neighbor_bits = [0] * n
    for i in range(n):
        bits = 0
        row = adj[i]
        # Iterate over True indices
        # For speed, we can use np.where
        idxs = np.where(row)[0]
        for j in idxs:
            bits |= (1 << j)
        neighbor_bits[i] = bits
    
    # Also precompute the bit for each vertex
    vertex_bits = [1 << i for i in range(n)]
    
    def simulate_greedy(perm):
        """
        Run greedy decoder on the given permutation.
        Returns the size of the independent set.
        Uses bitsets for fast neighbor removal.
        """
        active_mask = (1 << n) - 1  # All vertices active initially
        count = 0
        for v in perm:
            if (active_mask & vertex_bits[v]):
                # v is active
                count += 1
                # Remove v and its neighbors from active_mask
                # v is already removed by AND with ~vertex_bits[v]
                # Neighbors are in neighbor_bits[v]
                active_mask &= ~(vertex_bits[v] | neighbor_bits[v])
        return count

    # 1. Initial Heuristic Permutation
    # We want to pick low-degree vertices first, especially those adjacent to hubs.
    deg = adj.sum(axis=1).astype(float)
    # Score for initialization: lower is better (picks earlier)
    # We want to prioritize vertices that are "safe" to pick.
    # A common heuristic is degree. Low degree is generally better for MIS.
    # Let's use a mix of degree and local density.
    # Simple: sort by degree ascending.
    init_scores = deg.copy()
    # Maybe add a small penalty for being connected to high degree nodes?
    # Let's stick to a strong simple heuristic: degree.
    # Actually, the current program uses hub-blocking. Let's use a similar init score 
    # but convert to a permutation.
    
    # Re-calculate hub-blocking score for initialization
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    second_deg = adj @ (adj @ deg)
    safe_second = second_deg + 1.0
    second_density = second_deg / safe_second
    
    init_priority = -deg + 0.5 * avg_neighbor_deg - 0.1 * second_density
    # We want to pick high init_priority first.
    # So sort descending by init_priority.
    init_perm = np.argsort(-init_priority, kind='stable')
    
    best_perm = init_perm.copy()
    best_val = simulate_greedy(best_perm)
    
    # 2. Simulated Annealing
    rng = np.random.default_rng(42)
    
    # Parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.9995 # Tune based on time
    
    current_perm = best_perm.copy()
    current_val = best_val
    
    # Precompute indices for fast swapping
    # We will swap positions in the permutation array.
    
    steps = 0
    while time.time() < deadline - 2.0:
        # Generate a move: swap two random positions in the permutation
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        
        if i == j:
            continue
            
        # Swap
        current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        new_val = simulate_greedy(current_perm)
        
        delta = new_val - current_val
        
        if delta > 0:
            # Accept improvement
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best_perm = current_perm.copy()
        else:
            # Accept with probability exp(delta / T)
            # Since delta is integer (negative), and T is small, this is rare later on.
            # delta is in {-1, -2, ...} usually -1 or 0 for swaps? 
            # Actually, swapping two vertices in the greedy order can change the IS size by more than 1?
            # Yes, potentially. But usually small.
            if T_start > 0:
                # Avoid division by zero or underflow
                if T_start > 1e-9:
                    prob = np.exp(delta / T_start)
                else:
                    prob = 0.0
                
                if rng.random() < prob:
                    current_val = new_val
                else:
                    # Reject: swap back
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        steps += 1
        
        # Cooling
        T_start *= cooling_rate
        
        # If we hit a very low temperature and no improvements for a while, we could restart or stop.
        # But the deadline guard is the primary stop condition.
        
        # Optimization: If T is very low, we can increase cooling rate or break.
        if T_start < T_end and current_val == best_val and steps > 1000:
             # We are stuck at low temp. Let's do a few random restarts from best?
             # Or just break to save time for potential final polish.
             # For now, let's just let it run until deadline, as SA with low T is just local search.
             pass

    # 3. Final Polish: Local Search on the best permutation
    # Do a few iterations of best-improvement local search on the best_perm found.
    polish_perm = best_perm.copy()
    polish_val = best_val
    polish_rng = np.random.default_rng(123)
    
    # Limit polish time
    polish_deadline = time.time() + 2.0
    
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        # Try swapping each vertex with a few neighbors in the permutation
        # To keep it fast, we don't do O(n^2) swaps, but random samples.
        for _ in range(500):
            if time.time() > polish_deadline:
                break
            i = polish_rng.integers(0, n)
            j = polish_rng.integers(0, n)
            if i == j:
                continue
            
            polish_perm[i], polish_perm[j] = polish_perm[j], polish_perm[i]
            new_val = simulate_greedy(polish_perm)
            
            if new_val > polish_val:
                polish_val = new_val
                best_val = polish_val
                best_perm = polish_perm.copy()
                improved = True
            else:
                polish_perm[i], polish_perm[j] = polish_perm[j], polish_perm[i]
    
    # 4. Convert best permutation to scores
    # The greedy decoder picks vertices in descending score order.
    # So, the vertex at index 0 in best_perm should have the highest score.
    # The vertex at index n-1 should have the lowest.
    # We assign scores based on their rank in the permutation.
    scores = np.zeros(n)
    # Invert permutation: rank[vertex] = position in perm
    # If perm[k] = v, then v is picked at step k.
    # Higher priority = picked earlier = lower k.
    # So score should be decreasing with k.
    # Let score = n - k.
    for k, v in enumerate(best_perm):
        scores[v] = n - k
        
    return scores
