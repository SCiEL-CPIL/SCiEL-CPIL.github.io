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
       b. Use a specialized local search (1-add, 1-swap, 2-for-1, 1-for-2) to try to improve 
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
        """Hill-climbing local search with greedy improvements."""
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
            for v in range(m):
                if v not in current:
                    if (adj_mask[v] & current_mask) == 0:
                        current.add(v)
                        current_mask |= (1 << v)
                        improved = True
                        break
            
            # Try 2-move: Swap one node out, one in
            if not improved:
                found_swap = False
                for u in list(current):
                    mask_without_u = current_mask ^ (1 << u)
                    for v in range(m):
                        if v in current:
                            continue
                        if (adj_mask[v] & mask_without_u) == 0:
                            current.remove(u)
                            current.add(v)
                            current_mask ^= (1 << u)
                            current_mask |= (1 << v)
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
            
            # Try 3-move: Remove 2 nodes, add 1 node
            if not improved and len(current) >= 2:
                found_2for1 = False
                u_list = list(current)
                for i_idx in range(len(u_list)):
                    u1 = u_list[i_idx]
                    for j_idx in range(i_idx + 1, len(u_list)):
                        u2 = u_list[j_idx]
                        mask_without_u1u2 = current_mask ^ (1 << u1) ^ (1 << u2)
                        for v in range(m):
                            if v in current:
                                continue
                            if (adj_mask[v] & mask_without_u1u2) == 0:
                                current.remove(u1)
                                current.remove(u2)
                                current.add(v)
                                current_mask ^= (1 << u1)
                                current_mask ^= (1 << u2)
                                current_mask |= (1 << v)
                                improved = True
                                found_2for1 = True
                                break
                        if found_2for1:
                            break
                    if found_2for1:
                        break
            
            # Try 4-move: Remove 1 node, add 2 nodes
            if not improved and len(current) >= 1:
                found_1for2 = False
                for u in list(current):
                    mask_without_u = current_mask ^ (1 << u)
                    candidates = []
                    for v in range(m):
                        if v in current:
                            continue
                        if (adj_mask[v] & mask_without_u) == 0:
                            candidates.append(v)
                    
                    for idx1 in range(len(candidates)):
                        v1 = candidates[idx1]
                        for idx2 in range(idx1 + 1, len(candidates)):
                            v2 = candidates[idx2]
                            if (adj_mask[v1] >> v2) & 1:
                                continue
                            current.remove(u)
                            current.add(v1)
                            current.add(v2)
                            current_mask ^= (1 << u)
                            current_mask |= (1 << v1)
                            current_mask |= (1 << v2)
                            improved = True
                            found_1for2 = True
                            break
                        if found_1for2:
                            break
                    if found_1for2:
                        break

        return list(current)
    
    def simulated_annealing(initial_set, time_budget=30.0):
        """Simulated annealing to escape local optima."""
        current = set(initial_set)
        current_mask = 0
        for v in current:
            current_mask |= (1 << v)
        
        start_time = time.time()
        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling = 0.995
        
        best_set = set(current)
        best_size = len(current)
        
        while time.time() - start_time < time_budget:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a random node
                v = random.randint(0, m - 1)
                if v in current:
                    continue
                if (adj_mask[v] & current_mask) != 0:
                    # Invalid move - accept with probability exp(-delta/T)
                    # Delta is positive (bad move)
                    delta = 1.0  # Penalty for adding conflicting node
                    if random.random() < np.exp(-delta / T):
                        current.add(v)
                        current_mask |= (1 << v)
                else:
                    current.add(v)
                    current_mask |= (1 << v)
                    
            elif move_type < 0.6:
                # Remove a random node
                if not current:
                    continue
                u = random.choice(list(current))
                current.remove(u)
                current_mask ^= (1 << u)
                
            elif move_type < 0.85:
                # Swap: remove u, add v
                if not current:
                    continue
                u = random.choice(list(current))
                v = random.randint(0, m - 1)
                if v in current:
                    continue
                mask_without_u = current_mask ^ (1 << u)
                if (adj_mask[v] & mask_without_u) == 0:
                    # Valid swap
                    current.remove(u)
                    current.add(v)
                    current_mask ^= (1 << u)
                    current_mask |= (1 << v)
                else:
                    # Invalid swap - accept with probability
                    delta = 1.0
                    if random.random() < np.exp(-delta / T):
                        current.remove(u)
                        current.add(v)
                        current_mask ^= (1 << u)
                        current_mask |= (1 << v)
                        
            else:
                # 1-for-2: remove u, add v1, v2
                if not current:
                    continue
                u = random.choice(list(current))
                mask_without_u = current_mask ^ (1 << u)
                candidates = [v for v in range(m) if v not in current and (adj_mask[v] & mask_without_u) == 0]
                if len(candidates) >= 2:
                    v1, v2 = random.sample(candidates, 2)
                    if not ((adj_mask[v1] >> v2) & 1):
                        current.remove(u)
                        current.add(v1)
                        current.add(v2)
                        current_mask ^= (1 << u)
                        current_mask |= (1 << v1)
                        current_mask |= (1 << v2)
                    else:
                        delta = 1.0
                        if random.random() < np.exp(-delta / T):
                            current.remove(u)
                            current.add(v1)
                            current.add(v2)
                            current_mask ^= (1 << u)
                            current_mask |= (1 << v1)
                            current_mask |= (1 << v2)
                
            # Track best
            if len(current) > best_size:
                best_size = len(current)
                best_set = set(current)
                
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Final polish with hill climbing
        polished = local_search(list(best_set), max_iter=1000)
        return polished

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
        
        # Run simulated annealing to escape local optima
        if time.time() < deadline - 60:
            sa_result = simulated_annealing(best_set, time_budget=30.0)
            update_best(sa_result)
    
    # --- Randomized Greedy + Local Search + Simulated Annealing ---
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline - 30:
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
            
            # If this is a good start, try simulated annealing
            if len(refined) >= best_size and time.time() < deadline - 30:
                sa_result = simulated_annealing(refined, time_budget=15.0)
                update_best(sa_result)

    if best_set:
        return np.array(best_set, dtype=np.int64)
    
    return np.array([0], dtype=np.int64)

# EVOLVE-BLOCK-END
