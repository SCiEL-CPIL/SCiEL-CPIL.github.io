# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import milp, Bounds, LinearConstraint

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large square-difference-free set modulo 205.
    
    Strategy:
    1. Compute the conflict graph (edges connect pairs with forbidden difference).
    2. Use a hybrid approach:
       a. Run MILP to get an exact or high-quality initial solution.
       b. Use a specialized local search (1-swap, 2-swap, 3-swap) to try to improve 
          upon the MILP solution, in case the solver hit a time limit or returned a 
          sub-optimal solution due to numerical issues or solver settings.
       c. If MILP fails or times out, use randomized greedy + local search.
    3. The local search is optimized for speed using bitsets (integers) for adjacency.
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
    # adj_mask[i] is an integer where bit j is set if i and j are adjacent
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j = (i - d) % m
            if i != j:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # Precompute the list of forbidden pairs for MILP constraints
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

    # --- Local Search Helper ---
    # Uses bitmasks for speed.
    # state: set of nodes in current solution
    # We try to add a node, or swap one node for another, etc.
    
    def local_search(initial_set, max_iter=5000):
        current = set(initial_set)
        current_mask = 0
        for v in current:
            current_mask |= (1 << v)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Try 1-move: Add a node
            # A node v can be added if it has no neighbors in current set.
            # i.e., adj_mask[v] & current_mask == 0
            for v in range(m):
                if v not in current:
                    if (adj_mask[v] & current_mask) == 0:
                        current.add(v)
                        current_mask |= (1 << v)
                        improved = True
                        break
            
            # Try 2-move: Swap one node out, one in
            # Remove u, add v.
            # Valid if:
            # 1. u in current, v not in current
            # 2. v is not adjacent to any node in (current - {u})
            #    i.e., adj_mask[v] & (current_mask ^ (1 << u)) == 0
            # 3. u is not adjacent to v (implied by 2? No, u is removed. But if u and v were adjacent, 
            #    they couldn't be in the set together. But we are swapping. 
            #    Actually, if u and v are adjacent, they can't be in the set together anyway.
            #    The condition is just that v fits into the set without u.)
            
            # To make this faster, we can iterate over u in current and v not in current.
            # But m=205, so 205*205 ~ 40k checks per pass. With 5000 iter, 200M ops. Might be slow in Python.
            # Let's restrict the number of swaps or optimize.
            
            # Optimization: Only try swaps if we didn't improve via add.
            if not improved:
                # Try to find a swap
                # We can precompute "potential candidates" for v?
                # Let's just do a limited search for swaps to avoid slowdown.
                found_swap = False
                # Shuffle order for diversity? Or deterministic?
                # Let's iterate u
                for u in list(current):
                    # Mask of current set without u
                    mask_without_u = current_mask ^ (1 << u)
                    # Find v not in current such that adj_mask[v] & mask_without_u == 0
                    # This means v is compatible with all other nodes.
                    # We can iterate v
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
            
            # Try 3-move: Remove 2, add 1? Or Remove 1, add 2?
            # Usually 1-swap and 2-swap (remove 2 add 1) or (remove 1 add 2) are enough for small graphs.
            # Let's stick to 1-add and 1-swap for now.
            
        return list(current)

    deadline = time.time() + 580.0
    
    # --- MILP Approach ---
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
            
            start_time = time.time()
            # Give MILP 60 seconds
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, options={"time_limit": 60})
            
            if res.status == 0 or res.status == 1: # Optimal or Feasible
                solution = res.x.astype(int)
                subset = [i for i in range(n_vars) if solution[i] == 1]
                update_best(subset)
            else:
                # Try to use the incumbent if available?
                # milp doesn't easily expose incumbent in all versions/statuses simply.
                # We'll rely on fallback.
                pass
    except Exception:
        pass

    # --- Local Search Refinement ---
    # If we have a solution from MILP, refine it.
    if best_size > 0:
        refined = local_search(best_set, max_iter=1000)
        update_best(refined)
    
    # --- Randomized Greedy + Local Search Fallback ---
    # If MILP failed or we want to explore more, run random greedy starts.
    # This is a different design family: Stochastic Local Search (Simulated Annealing-like or simple Hill Climbing)
    # from multiple random starts.
    
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            # Random greedy construction
            order = list(range(m))
            random.shuffle(order)
            current = []
            current_mask = 0
            for candidate in order:
                if (adj_mask[candidate] & current_mask) == 0:
                    current.append(candidate)
                    current_mask |= (1 << candidate)
            
            # Local search from this random start
            refined = local_search(current, max_iter=2000)
            update_best(refined)

    if best_set:
        return np.array(best_set, dtype=np.int64)
    
    return np.array([0], dtype=np.int64)

# EVOLVE-BLOCK-END
