# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import milp, Bounds, LinearConstraint

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large square-difference-free set modulo 205.
    
    Strategy:
    1. Compute the conflict graph using bitmasks for fast adjacency checks.
    2. Phase 1: Run multiple randomized greedy constructions followed by 
       a powerful local search that includes 1-swap and 2-swap (remove 2, add 2) moves.
       This helps escape local optima better than simple hill climbing.
    3. Phase 2: Use MILP to attempt to find the exact maximum independent set 
       or improve upon the heuristic solution.
    4. Return the best valid set found.
    """
    m = 205
    k = 2
    
    # Step 1: Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # Build adjacency as bitmasks for fast local search
    # adj_mask[i] is an integer where bit j is set if i and j are adjacent (conflict)
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j = (i - d) % m
            if i != j:
                mask |= (1 << j)
        adj_mask[i] = mask
        
    # Precompute edges for MILP
    edges = []
    for i in range(m):
        for j in range(i + 1, m):
            if (adj_mask[i] >> j) & 1:
                edges.append((i, j))
    
    def is_valid(subset):
        if len(subset) == 0:
            return False
        if len(set(subset)) != len(subset):
            return False
        if any(x < 0 or x >= m for x in subset):
            return False
        s = set(subset)
        for a in subset:
            for b in subset:
                if a == b:
                    continue
                d = (a - b) % m
                if d in forbidden:
                    return False
        return True

    best_set = []
    best_size = 0
    
    def update_best(subset):
        nonlocal best_set, best_size
        if is_valid(subset):
            if len(subset) > best_size:
                best_set = list(subset)
                best_size = len(subset)

    def local_search(initial_set, max_iter=2000):
        """
        Local search with 1-add, 1-swap, and 2-swap moves.
        """
        current = set(initial_set)
        current_mask = 0
        for v in current:
            current_mask |= (1 << v)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # 1. Try to Add a node
            for v in range(m):
                if v not in current:
                    if (adj_mask[v] & current_mask) == 0:
                        current.add(v)
                        current_mask |= (1 << v)
                        improved = True
                        break
            
            # 2. Try 1-Swap (Remove u, Add v)
            if not improved:
                found_swap = False
                # Iterate u in current
                for u in list(current):
                    mask_without_u = current_mask ^ (1 << u)
                    # Find v not in current such that v is compatible with mask_without_u
                    for v in range(m):
                        if v in current:
                            continue
                        if (adj_mask[v] & mask_without_u) == 0:
                            # Swap u with v
                            current.remove(u)
                            current.add(v)
                            current_mask ^= (1 << u)
                            current_mask |= (1 << v)
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
            
            # 3. Try 2-Swap (Remove u1, u2, Add v1, v2)
            if not improved:
                # This is more expensive, so we do it less frequently or with limits
                # We iterate over pairs to remove
                found_2swap = False
                current_list = list(current)
                n_curr = len(current_list)
                
                # Optimization: Only try 2-swaps if we've been stuck for a few iterations?
                # For now, just try it.
                for i in range(n_curr):
                    if found_2swap: break
                    u1 = current_list[i]
                    for j in range(i + 1, n_curr):
                        if found_2swap: break
                        u2 = current_list[j]
                        
                        mask_wo_u1u2 = current_mask ^ (1 << u1) ^ (1 << u2)
                        
                        # Find candidates v not in current that are compatible with mask_wo_u1u2
                        candidates = []
                        for v in range(m):
                            if v in current:
                                continue
                            if (adj_mask[v] & mask_wo_u1u2) == 0:
                                candidates.append(v)
                        
                        # If we have at least 2 candidates, check if any pair is compatible
                        if len(candidates) >= 2:
                            # Check pairs in candidates
                            for a in range(len(candidates)):
                                if found_2swap: break
                                v1 = candidates[a]
                                for b in range(a + 1, len(candidates)):
                                    v2 = candidates[b]
                                    # Check if v1 and v2 are compatible
                                    if (adj_mask[v1] & (1 << v2)) == 0:
                                        # Perform swap
                                        current.remove(u1)
                                        current.remove(u2)
                                        current.add(v1)
                                        current.add(v2)
                                        current_mask ^= (1 << u1)
                                        current_mask ^= (1 << u2)
                                        current_mask |= (1 << v1)
                                        current_mask |= (1 << v2)
                                        improved = True
                                        found_2swap = True
                                        break
                                        
        return list(current)

    deadline = time.time() + 580.0
    
    # --- Phase 1: Randomized Greedy + Local Search ---
    # Run multiple restarts to find a good initial solution
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 300: # Leave 300s for MILP
            break
            
        order = list(range(m))
        random.shuffle(order)
        current = []
        current_mask = 0
        for candidate in order:
            if (adj_mask[candidate] & current_mask) == 0:
                current.append(candidate)
                current_mask |= (1 << candidate)
        
        # Refine with local search
        refined = local_search(current, max_iter=1000)
        update_best(refined)
        
        # If we found a very large set, we can reduce restarts?
        # But let's keep exploring for now.

    # --- Phase 2: MILP ---
    # If we still have time, try MILP to potentially find the exact maximum
    if time.time() < deadline - 10:
        try:
            n_vars = m
            c = -np.ones(n_vars)
            
            n_constraints = len(edges)
            if n_constraints > 0:
                A = np.zeros((n_constraints, n_vars))
                for idx, (i, j) in enumerate(edges):
                    A[idx, i] = 1
                    A[idx, j] = 1
                
                bounds = Bounds(lb=0, ub=1)
                integrality = np.ones(n_vars)
                cons = LinearConstraint(A, ub=np.ones(n_constraints))
                
                # Calculate remaining time
                remaining_time = deadline - time.time() - 5.0
                if remaining_time > 10:
                    res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, options={"time_limit": remaining_time})
                    
                    if res.status == 0 or res.status == 1: # Optimal or Feasible
                        solution = res.x.astype(int)
                        subset = [i for i in range(n_vars) if solution[i] == 1]
                        update_best(subset)
        except Exception:
            pass

    if best_set:
        return np.array(best_set, dtype=np.int64)
    
    # Fallback
    return np.array([0], dtype=np.int64)

# EVOLVE-BLOCK-END
