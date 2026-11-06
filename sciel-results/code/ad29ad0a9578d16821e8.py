import numpy as np
import time
import random
import sys

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    
    Strategy:
    1. Extract instance data and normalize predecessors.
    2. Define a highly optimized Serial SGS simulator (pure Python loops, minimal overhead).
    3. Generate a diverse set of initial priority permutations based on standard heuristics 
       (CPL, LPT, RI, etc.).
    4. Perform a deterministic multi-start local search (2-opt swaps) to minimize makespan.
       - We treat the priority vector as a permutation of ranks [0..N-1].
       - This avoids the issue of continuous value tuning and focuses on the relative order,
         which is the only thing that matters for SGS.
    5. Return the best priority vector found.
    """
    start_time = time.time()
    # Safety margin: 145s out of 150s limit
    deadline = start_time + 145.0

    # 1. Extract and process instance data
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)  # Shape (N, R)
    capacities = np.array(instance["capacities"], dtype=float) # Shape (R,)
    R = len(capacities)
    
    # Handle predecessors
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             # Assuming 1-indexed or 0-indexed? Usually 0-indexed in these tasks unless specified.
             # The current code assumes 0-indexed lists. 
             # If it's a matrix, often 1 is True.
             # Let's stick to the logic that converts to lists of indices.
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 # Convert to int list
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
            
    # 2. Compute Static Heuristics for Initial Candidates
    # Topological Sort
    in_degree = [len(p) for p in preds]
    queue = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
    q = queue[:]
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            indeg_temp[v] -= 1
            if indeg_temp[v] == 0:
                q.append(v)
                
    # Compute Critical Path Length (CPL) backwards
    cpl = np.zeros(N)
    for i in range(len(topo_order) - 1, -1, -1):
        u = topo_order[i]
        max_succ_cpl = 0
        for v in succs[u]:
            if cpl[v] > max_succ_cpl:
                max_succ_cpl = cpl[v]
        cpl[u] = durations[u] + max_succ_cpl
        
    # Compute Resource Intensity (RI)
    cap_safe = np.where(capacities > 0, capacities, 1.0)
    ratios = demands / cap_safe  # Shape (N, R)
    ri = np.sum(ratios, axis=1)  # Shape (N,)

    # Compute Successor Count (SC)
    succ_count = np.zeros(N, dtype=int)
    for i in range(N):
        visited = set()
        stack = [i]
        while stack:
            node = stack.pop()
            for s in succs[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        succ_count[i] = len(visited)

    # 3. Optimized SGS Simulator
    # Key optimization: Avoid NumPy operations in the inner loop.
    # Use lists of floats for remaining resources.
    # Pre-compute eligible sets? No, dynamic.
    
    def simulate_sgs_fast(priorities):
        """
        priorities: list of floats or ints, higher is better.
        Returns makespan.
        """
        scheduled = [False] * N
        # active: list of (finish_time, idx)
        active = []
        # remaining_res: list of floats
        remaining_res = [float(c) for c in capacities]
        time_current = 0.0
        makespan = 0.0
        
        # Pre-compute demand rows as lists for faster access?
        # demands is a numpy array. Accessing demands[i] returns an array.
        # Let's convert demands to list of lists for faster element access in Python.
        # Actually, demands[i] in numpy is fast enough, but let's try to be safe.
        # For N=30, R=4, it's tiny.
        
        # To speed up sorting, we can pre-sort the indices by priority once?
        # No, eligibility changes.
        
        while True:
            # Check if all done
            # Optimization: count scheduled?
            # if all(scheduled): break -> O(N)
            # Let's just check in the loop below.
            
            # Find eligible
            eligible = []
            for i in range(N):
                if not scheduled[i]:
                    # Check preds
                    is_eligible = True
                    for p in preds[i]:
                        if not scheduled[p]:
                            is_eligible = False
                            break
                    if is_eligible:
                        eligible.append(i)
                        
            if not eligible:
                # Wait for next event
                if not active:
                    break # Should not happen if valid
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                # Process completions
                new_active = []
                for f, idx in active:
                    if f > time_current:
                        new_active.append((f, idx))
                    else:
                        # Release resources
                        d_idx = demands[idx]
                        for r in range(R):
                            remaining_res[r] += d_idx[r]
                        makespan = max(makespan, f)
                active = new_active
                continue
                
            # Sort eligible by priority (descending)
            # Using sorted with key is O(K log K)
            eligible.sort(key=lambda x: priorities[x], reverse=True)
            
            # Attempt to schedule
            any_scheduled = False
            for i in eligible:
                # Check resources
                d_i = demands[i]
                can_schedule = True
                for r in range(R):
                    if remaining_res[r] < d_i[r]:
                        can_schedule = False
                        break
                if can_schedule:
                    scheduled[i] = True
                    # Consume resources
                    for r in range(R):
                        remaining_res[r] -= d_i[r]
                    finish_t = time_current + durations[i]
                    active.append((finish_t, i))
                    makespan = max(makespan, finish_t)
                    any_scheduled = True
                    
            if not any_scheduled:
                # Wait
                if not active:
                    break
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                new_active = []
                for f, idx in active:
                    if f > time_current:
                        new_active.append((f, idx))
                    else:
                        d_idx = demands[idx]
                        for r in range(R):
                            remaining_res[r] += d_idx[r]
                        makespan = max(makespan, f)
                active = new_active
                
        return makespan

    # 4. Generate Initial Candidates
    # We want to search the space of permutations.
    # We will represent the priority vector as a list of floats.
    # However, for local search, it's easier to work with a permutation of indices 0..N-1
    # and then map that to a priority vector?
    # Actually, the SGS only cares about the relative order.
    # So we can just return a vector of ranks.
    # Let's define candidates as lists of floats.
    
    candidates = []
    
    # Heuristic 1: CPL
    candidates.append(cpl.tolist())
    
    # Heuristic 2: CPL * (1 + RI)
    candidates.append((cpl * (1.0 + ri)).tolist())
    
    # Heuristic 3: LPT
    candidates.append(durations.tolist())
    
    # Heuristic 4: LPT * (1 + RI)
    candidates.append((durations * (1.0 + ri)).tolist())
    
    # Heuristic 5: Successor Count
    candidates.append(succ_count.astype(float).tolist())
    
    # Heuristic 6: Random (fixed seed)
    random.seed(42)
    random_cand = [random.random() for _ in range(N)]
    candidates.append(random_cand)
    
    # 5. Local Search
    # We will perform hill climbing on the priority vector.
    # Move: Swap the priority values of two activities i and j.
    # This changes the relative order.
    
    best_score = float('inf')
    best_priorities = candidates[0]
    
    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        score = simulate_sgs_fast(cand)
        if score < best_score:
            best_score = score
            best_priorities = cand.copy()
            
    # Deterministic Multi-Start Local Search
    # We will iterate through all pairs (i, j) and try swapping them.
    # If improvement, accept. Repeat until no improvement (local optimum).
    # Then, to escape local optima, we can try a few random restarts or perturbations.
    # Given the budget, a single strong local search from the best initial candidate might be enough,
    # but multi-start is safer.
    
    # Let's do a "First Improvement" or "Best Improvement" local search?
    # Best Improvement is slower but more effective.
    # First Improvement is faster.
    # Let's use a hybrid: iterate through swaps, if improvement, accept and restart the loop?
    # Or just scan all pairs, accept best improvement, repeat.
    
    # Let's try a simple "Scan and Accept" (First Improvement) to maximize iterations.
    
    current_priorities = best_priorities
    current_score = best_score
    
    # We will run local search until time runs out or no improvement.
    # To avoid getting stuck in a very shallow local optimum, we can also try 
    # perturbing the best solution found so far if we have time.
    
    improved = True
    no_improve_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs (i, j)
        for i in range(N):
            if time.time() > deadline:
                break
            for j in range(i + 1, N):
                # Swap i and j
                temp = current_priorities[i]
                current_priorities[i] = current_priorities[j]
                current_priorities[j] = temp
                
                new_score = simulate_sgs_fast(current_priorities)
                
                if new_score < current_score - 1e-9: # Strict improvement
                    current_score = new_score
                    best_score = new_score
                    best_priorities = current_priorities.copy()
                    improved = True
                    # Restart the scan? Or continue?
                    # Restarting ensures we don't miss better moves that become available.
                    # But it's expensive.
                    # Let's just break the inner loop and restart outer?
                    # Actually, if we accept, the landscape changes.
                    # Let's break both loops and restart from scratch with the new best.
                    # This is "First Improvement" with restart.
                    break
                else:
                    # Swap back
                    current_priorities[j] = current_priorities[i]
                    current_priorities[i] = temp
                    
            if improved:
                break
                
        if not improved:
            # Local optimum reached for this start.
            # Try a random perturbation to escape?
            # Let's try a random swap from the best solution found so far.
            if time.time() < deadline - 10: # Keep 10s margin
                # Perturb
                p_i, p_j = random.sample(range(N), 2)
                temp = best_priorities[p_i]
                best_priorities[p_i] = best_priorities[p_j]
                best_priorities[p_j] = temp
                current_priorities = best_priorities.copy()
                current_score = simulate_sgs_fast(current_priorities)
                improved = True # Force another round of local search
            else:
                break

    return best_priorities

