"""Assembly Line Balancing - Adaptive Weight Optimization via Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque
import heapq
import random

def priority(instance):
    """
    Score each assembly task for the priority rule.
    Uses an adaptive approach:
    1. Compute structural features (LPE, LPS, Successors, etc.) for each task.
    2. Define priority as a linear combination of these features with learnable weights.
    3. Optimize the weights using local search to minimize the number of stations
       for this specific instance.
    """
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    # Build graph structures
    adj = [[] for _ in range(n)]  # adj[i] = successors of i
    preds_list = []
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        preds_list.append(preds)
        for p in preds:
            adj[p].append(i)

    # Topological sort
    in_deg = np.array([len(p) for p in preds_list], dtype=int)
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    in_deg_tmp = in_deg.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in adj[node]:
            in_deg_tmp[succ] -= 1
            if in_deg_tmp[succ] == 0:
                queue.append(succ)

    if len(topo_order) < n:
        # Cycle detected; fall back to task times
        return times.copy()

    # Compute structural features
    # 1. Longest Path to End (LPE)
    lpe = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in adj[node]:
            val = times[succ] + lpe[succ]
            if val > max_succ:
                max_succ = val
        lpe[node] = times[node] + max_succ

    # 2. Longest Path from Start (LPS)
    lps = np.zeros(n, dtype=float)
    for node in topo_order:
        for succ in adj[node]:
            val = times[node] + lps[node]
            if val > lps[succ]:
                lps[succ] = val

    # 3. Number of transitive successors
    num_succ = np.zeros(n, dtype=int)
    for node in reversed(topo_order):
        for succ in adj[node]:
            num_succ[node] += num_succ[succ] + 1

    # 4. Number of direct successors
    direct_succ = np.array([len(adj[i]) for i in range(n)], dtype=float)

    # 5. Normalized topological position (0 = earliest, 1 = latest)
    pos_in_topo = np.zeros(n, dtype=float)
    for idx, node in enumerate(topo_order):
        pos_in_topo[node] = idx
    if n > 1:
        norm_pos = pos_in_topo / (n - 1)
    else:
        norm_pos = np.zeros(n, dtype=float)

    # 6. Task time
    task_time = times.copy()

    # Normalize features to [0, 1] or similar scale to help weight optimization
    # We want weights to be comparable
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        min_val = arr.min()
        max_val = arr.max()
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)

    feat_lpe = normalize(lpe)
    feat_lps = normalize(lps)
    feat_succ = normalize(num_succ.astype(float))
    feat_dsucc = normalize(direct_succ)
    feat_pos = 1.0 - norm_pos # Earlier is better
    feat_time = normalize(task_time)

    # Stack features into a matrix (n tasks x d features)
    # Features: [LPE, LPS, Succ, DSucc, Pos, Time]
    features = np.column_stack([
        feat_lpe, 
        feat_lps, 
        feat_succ, 
        feat_dsucc, 
        feat_pos, 
        feat_time
    ])

    # Fast Simulation Function
    # Optimized for speed: minimal object creation, tight loops
    def simulate(priority_scores):
        completed = [False] * n
        stations = 0
        current_load = 0.0
        remaining_count = n
        
        # pending[t] = number of uncompleted predecessors of t
        pending = [len(p) for p in preds_list]
        
        # Heap entries: (-priority, task_id)
        heap = []
        
        # Initially eligible: tasks with no predecessors that fit
        for t in range(n):
            if pending[t] == 0 and times[t] <= cycle_time + 1e-9:
                heapq.heappush(heap, (-priority_scores[t], t))
        
        while remaining_count > 0:
            # Pop highest-priority eligible task that fits remaining capacity
            placed = False
            while heap:
                neg_pri, t = heapq.heappop(heap)
                if completed[t]:
                    continue
                if times[t] <= cycle_time - current_load + 1e-9:
                    # Place it
                    completed[t] = True
                    current_load += times[t]
                    remaining_count -= 1
                    placed = True
                    # Check successors
                    for s in adj[t]:
                        pending[s] -= 1
                        if pending[s] == 0 and times[s] <= cycle_time + 1e-9:
                            heapq.heappush(heap, (-priority_scores[s], s))
                    break
            
            if not placed:
                # No eligible task fits in current station; start new station
                stations += 1
                current_load = 0.0
                # Re-push all eligible (pending==0, not completed) tasks
                for t in range(n):
                    if not completed[t] and pending[t] == 0 and times[t] <= cycle_time + 1e-9:
                        heapq.heappush(heap, (-priority_scores[t], t))
            
            if not heap:
                # Shouldn't happen for feasible instances; break to avoid infinite loop
                stations += 1
                break
        
        return stations

    # Local Search for Weights
    # Weights vector w of length 6
    # Priority = features @ w
    
    # Initial guess: favor LPE and Early Position
    best_w = np.array([1.0, 0.0, 0.5, 0.2, 0.5, 0.0])
    best_priority = features @ best_w
    best_stations = simulate(best_priority)
    
    # Local Search: Random Perturbation with Acceptance
    # Since the objective is discrete, we use a "hill climbing" approach
    # We try small perturbations to the weights and accept if it improves station count
    
    rng = random.Random(42)
    step_size = 0.1
    max_iterations = 5000 # Cap on iterations, controlled by deadline
    
    # Pre-compute feature matrix transposed for faster dot products if needed, 
    # but numpy @ is already fast for small n.
    
    current_w = best_w.copy()
    current_stations = best_stations
    no_improvement_count = 0
    
    for i in range(max_iterations):
        if time.time() > DEADLINE - 10:
            break
        
        # Perturb weights
        # Choose a random subset of weights to perturb to avoid large jumps
        num_perturb = rng.randint(1, 3)
        indices = rng.sample(range(6), num_perturb)
        
        new_w = current_w.copy()
        for idx in indices:
            # Random direction
            direction = 1 if rng.random() > 0.5 else -1
            # Step size might need to adapt, but fixed is okay for small search space
            new_w[idx] += direction * step_size * rng.random()
            # Keep weights non-negative to maintain monotonicity of features? 
            # Actually, negative weights might be useful (e.g., penalize late tasks if pos is 1 for late).
            # But let's constrain to reasonable range [-1, 2]
            new_w[idx] = np.clip(new_w[idx], -1.0, 2.0)
        
        # Calculate new priority
        new_priority = features @ new_w
        
        # Simulate
        new_stations = simulate(new_priority)
        
        if new_stations < current_stations:
            current_w = new_w
            current_stations = new_stations
            no_improvement_count = 0
            if current_stations < best_stations:
                best_stations = current_stations
                best_w = current_w.copy()
                best_priority = new_priority.copy()
            
            # Increase step size slightly on success? Or keep same.
            # Let's keep step size constant for stability in discrete space.
            
        else:
            no_improvement_count += 1
            # If stuck for a while, increase step size to escape local minima
            if no_improvement_count > 50:
                step_size *= 1.5
                no_improvement_count = 0
                if step_size > 2.0:
                    step_size = 2.0

    # Final check: try a few specific strong heuristics if search didn't beat them
    # Sometimes the search gets stuck in a local minimum that is worse than a simple heuristic
    # Simple heuristics to compare against:
    
    # 1. Pure LPE
    p_lpe = feat_lpe * 1.0 # Just LPE
    s_lpe = simulate(p_lpe)
    if s_lpe < best_stations:
        best_stations = s_lpe
        best_priority = p_lpe
    
    # 2. LPE + Successors
    p_combo = feat_lpe + 0.5 * feat_succ
    s_combo = simulate(p_combo)
    if s_combo < best_stations:
        best_stations = s_combo
        best_priority = p_combo

    # 3. LPE + Early Pos
    p_combo2 = feat_lpe + 0.5 * feat_pos
    s_combo2 = simulate(p_combo2)
    if s_combo2 < best_stations:
        best_stations = s_combo2
        best_priority = p_combo2
        
    # 4. The optimized one
    p_opt = features @ best_w
    s_opt = simulate(p_opt)
    if s_opt < best_stations:
        best_stations = s_opt
        best_priority = p_opt

    return np.asarray(best_priority, dtype=float)

# EVOLVE-BLOCK-END
