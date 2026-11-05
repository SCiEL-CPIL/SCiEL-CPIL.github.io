import random
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Precompute all edges: (u, v, dim) where u < v and dim is the bit that differs
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))
    
    total_edges = len(all_edges)  # 448
    
    # Build a mapping from edge tuple to index
    edge_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx
    
    # For each edge, precompute the "forbidden triples" - the other 3 edges that would complete a C4
    edge_to_forbidden = {}
    for idx, (u, v, i) in enumerate(all_edges):
        forbidden = []
        for j in range(d):
            if j != i:
                c = u ^ (1 << j)
                e = v ^ (1 << j)
                e1 = (min(u, c), max(u, c))
                e2 = (min(v, e), max(v, e))
                e3 = (min(c, e), max(c, e))
                forbidden.append((e1, e2, e3))
        edge_to_forbidden[idx] = forbidden
    
    # Convert forbidden to index lists for faster lookup
    edge_to_forbidden_idx = {}
    for idx in range(total_edges):
        forbidden = []
        for (e1, e2, e3) in edge_to_forbidden[idx]:
            i1 = edge_to_idx.get(e1)
            i2 = edge_to_idx.get(e2)
            i3 = edge_to_idx.get(e3)
            if i1 is not None and i2 is not None and i3 is not None:
                forbidden.append((i1, i2, i3))
        edge_to_forbidden_idx[idx] = forbidden
    
    def is_addable(idx, chosen_arr):
        """Check if edge idx can be added to chosen set (numpy bool array) without creating a C4"""
        for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
            if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                return False
        return True
    
    def is_addable_set(idx, chosen):
        """Check if edge idx can be added to chosen set (Python set) without creating a C4"""
        for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
            if i1 in chosen and i2 in chosen and i3 in chosen:
                return False
        return True
    
    def greedy_randomized(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = np.zeros(total_edges, dtype=bool)
        # Shuffle edges for random order
        order = list(range(total_edges))
        rng.shuffle(order)
        
        for idx in order:
            if is_addable(idx, chosen_arr):
                chosen_arr[idx] = True
        
        return chosen_arr
    
    def local_search(chosen_arr, max_iters=10000, seed=None):
        """Local search: try to add unchosen edges, removing blocking edges if needed"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Build list of unchosen edges
            unchosen = np.where(~chosen_arr)[0]
            if len(unchosen) == 0:
                break
            
            # Try a random unchosen edge
            idx = int(unchosen[rng.randint(0, len(unchosen) - 1)])
            
            if is_addable(idx, chosen_arr):
                # Can add directly
                chosen_arr[idx] = True
                current_size += 1
            else:
                # Find blocking triples
                blocking = []
                for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        blocking.append((i1, i2, i3))
                
                if not blocking:
                    chosen_arr[idx] = True
                    current_size += 1
                else:
                    # Try removing one edge from a random blocking triple
                    triple = blocking[rng.randint(0, len(blocking) - 1)]
                    for rem_idx in triple:
                        chosen_arr[rem_idx] = False
                        if is_addable(idx, chosen_arr):
                            chosen_arr[idx] = True
                            # Net change: 0 (removed 1, added 1)
                            break
                        else:
                            # Restore
                            chosen_arr[rem_idx] = True
        
        return chosen_arr
    
    def swap_search(chosen_arr, max_iters=5000, seed=None):
        """Try swap moves: remove 1 edge, add 1 different edge (net 0), 
        but the goal is to enable future additions. Also try remove 1 add 2 (net +1)"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            unchosen = np.where(~chosen_arr)[0]
            if len(unchosen) == 0:
                break
            
            # Pick a random unchosen edge
            idx = int(unchosen[rng.randint(0, len(unchosen) - 1)])
            
            if is_addable(idx, chosen_arr):
                chosen_arr[idx] = True
                current_size += 1
            else:
                # Find blocking triples
                blocking = []
                for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        blocking.append((i1, i2, i3))
                
                if not blocking:
                    chosen_arr[idx] = True
                    current_size += 1
                else:
                    # Try removing one edge from each blocking triple
                    # and check if we can add idx
                    for triple in blocking:
                        for rem_idx in triple:
                            chosen_arr[rem_idx] = False
                            if is_addable(idx, chosen_arr):
                                chosen_arr[idx] = True
                                # Net: 0
                                break
                            else:
                                chosen_arr[rem_idx] = True
        
        return chosen_arr
    
    def two_opt_search(chosen_arr, max_iters=3000, seed=None):
        """Try 2-Opt: remove 2 edges, add 2 edges (net 0), or remove 1 add 2 (net +1)"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            unchosen = np.where(~chosen_arr)[0]
            if len(unchosen) < 2:
                break
            
            # Pick 2 random unchosen edges
            s = rng.sample(range(len(unchosen)), 2)
            idx1 = int(unchosen[s[0]])
            idx2 = int(unchosen[s[1]])
            
            # Check if both can be added directly
            if is_addable(idx1, chosen_arr) and is_addable(idx2, chosen_arr):
                chosen_arr[idx1] = True
                chosen_arr[idx2] = True
                current_size += 2
            else:
                # Try removing 1 edge to enable adding both
                # Get all blocking edges for idx1 and idx2
                block1 = set()
                for (i1, i2, i3) in edge_to_forbidden_idx[idx1]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        block1.update([i1, i2, i3])
                
                block2 = set()
                for (i1, i2, i3) in edge_to_forbidden_idx[idx2]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        block2.update([i1, i2, i3])
                
                # Try removing one edge from the union of blocking sets
                candidates = list(block1 | block2)
                rng.shuffle(candidates)
                
                for rem_idx in candidates[:3]:  # Try at most 3 removals
                    chosen_arr[rem_idx] = False
                    if is_addable(idx1, chosen_arr) and is_addable(idx2, chosen_arr):
                        chosen_arr[idx1] = True
                        chosen_arr[idx2] = True
                        current_size += 1  # Net +1
                        break
                    else:
                        chosen_arr[rem_idx] = True
        
        return chosen_arr
    
    def simulated_annealing(chosen_arr, max_iters=20000, seed=None):
        """Simulated Annealing: accept worse states with temperature-dependent probability"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        best_arr = chosen_arr.copy()
        best_size = current_size
        
        T_start = 5.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))
        T = T_start
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            T *= cooling_rate
            
            unchosen = np.where(~chosen_arr)[0]
            chosen_indices = np.where(chosen_arr)[0]
            
            if len(unchosen) == 0:
                break
            
            # Random move: either add an unchosen edge or remove a chosen edge
            if rng.random() < 0.7 and len(unchosen) > 0:
                # Try to add a random unchosen edge
                idx = int(unchosen[rng.randint(0, len(unchosen) - 1)])
                
                if is_addable(idx, chosen_arr):
                    chosen_arr[idx] = True
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_arr = chosen_arr.copy()
                else:
                    # Find blocking triples and try a swap
                    blocking = []
                    for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                        if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                            blocking.append((i1, i2, i3))
                    
                    if blocking:
                        triple = blocking[rng.randint(0, len(blocking) - 1)]
                        rem_idx = triple[rng.randint(0, 2)]
                        old_size = current_size
                        chosen_arr[rem_idx] = False
                        if is_addable(idx, chosen_arr):
                            chosen_arr[idx] = True
                            # Net 0, always accept
                            if current_size > best_size:
                                best_size = current_size
                                best_arr = chosen_arr.copy()
                        else:
                            # Restore
                            chosen_arr[rem_idx] = True
            else:
                # Try to remove a random chosen edge (to escape local optima)
                if len(chosen_indices) > 0:
                    rem_idx = int(chosen_indices[rng.randint(0, len(chosen_indices) - 1)])
                    old_size = current_size
                    chosen_arr[rem_idx] = False
                    current_size -= 1
                    
                    # Accept with probability exp(-delta / T) where delta = -1 (size decreased)
                    delta = -1
                    accept_prob = np.exp(-delta / T) if T > 1e-6 else 0.0
                    if rng.random() < accept_prob:
                        # Accept the removal
                        if current_size > best_size:
                            best_size = current_size
                            best_arr = chosen_arr.copy()
                    else:
                        # Reject: restore
                        chosen_arr[rem_idx] = True
                        current_size = old_size
        
        return best_arr
    
    best_solution = None
    best_size = 0
    
    # Phase 1: Multiple randomized greedy restarts
    num_restarts = 0
    while time.time() - start_time < time_limit - 40:
        seed = num_restarts
        solution = greedy_randomized(seed)
        size = int(solution.sum())
        if size > best_size:
            best_size = size
            best_solution = solution
        num_restarts += 1
    
    # Phase 2: Local search on best solution
    if best_solution is not None and time.time() - start_time < time_limit - 25:
        improved = local_search(best_solution, max_iters=15000, seed=42)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    # Phase 3: Simulated Annealing
    if best_solution is not None and time.time() - start_time < time_limit - 15:
        improved = simulated_annealing(best_solution, max_iters=25000, seed=789)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    # Phase 4: Swap search
    if best_solution is not None and time.time() - start_time < time_limit - 10:
        improved = swap_search(best_solution, max_iters=8000, seed=123)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    # Phase 5: 2-Opt search
    if best_solution is not None and time.time() - start_time < time_limit - 8:
        improved = two_opt_search(best_solution, max_iters=5000, seed=456)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    # Phase 6: More local search + SA iterations if time permits
    if best_solution is not None and time.time() - start_time < time_limit - 5:
        for _ in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            improved = local_search(best_solution, max_iters=8000, seed=random.randint(0, 100000))
            if int(improved.sum()) > best_size:
                best_size = int(improved.sum())
                best_solution = improved
            if time.time() - start_time > time_limit - 5:
                break
            improved = simulated_annealing(best_solution, max_iters=10000, seed=random.randint(0, 100000))
            if int(improved.sum()) > best_size:
                best_size = int(improved.sum())
                best_solution = improved
    
    if best_solution is None:
        # Fallback: deterministic greedy
        best_solution = greedy_randomized(0)
    
    # Convert to output format
    result = []
    for idx in np.where(best_solution)[0]:
        u, v, i = all_edges[idx]
        result.append([u, v])
    
    return result
