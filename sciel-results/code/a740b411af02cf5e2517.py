import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit
    
    N = 256
    max_sum = 2 * N
    
    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
    
    # 2. Filter vertices: 2*v must not be a square (a=b case)
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1]
        
    # Map vertex value to index
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 3. Build adjacency list and edge list
    # Edge exists if vi + vj is a square
    edges = []
    neighbors = {v: set() for v in valid_vertices}
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                edges.append((i, j))
                neighbors[vi].add(vj)
                neighbors[vj].add(vi)
                
    n_edges = len(edges)
    
    # If no edges, all valid vertices can be chosen
    if n_edges == 0:
        return valid_vertices[:]
        
    # --- Strategy 1: Fast Heuristic (Greedy + Local Search) ---
    # This provides a strong baseline quickly.
    
    # Greedy: Sort by degree ascending
    vs_sorted = sorted(valid_vertices, key=lambda v: len(neighbors[v]))
    
    S_heuristic = set()
    for v in vs_sorted:
        if neighbors[v].isdisjoint(S_heuristic):
            S_heuristic.add(v)
            
    best_solution = list(S_heuristic)
    best_score = len(best_solution)
    
    # --- Strategy 2: Custom Branch and Bound ---
    
    # Build A matrix (n_edges x n_vars)
    A = lil_matrix((n_edges, n_vars))
    for k, (i, j) in enumerate(edges):
        A[k, i] = 1
        A[k, j] = 1
        
    A = A.tocsr()
    
    # Objective: maximize sum(x) -> minimize -sum(x)
    c = -np.ones(n_vars)
    
    # Bounds for LP
    bounds = [(0, 1) for _ in range(n_vars)]
    
    # Helper to solve LP relaxation
    def solve_lp(fixed_vars):
        # fixed_vars: dict {index: value}
        # We modify bounds for fixed variables
        bnds = list(bounds)
        for idx, val in fixed_vars.items():
            bnds[idx] = (val, val)
            
        res = linprog(c, A_ub=A, b_ub=np.ones(n_edges), bounds=bnds, method='highs')
        if res.status != 0:
            return None, None
        return res.fun, res.x
    
    # B&B State
    # Node: (lp_bound, fixed_vars_dict, x_solution_if_integral)
    # We use a list as a stack/queue.
    # To prioritize, we can sort by bound, but simple DFS/LIFO is often faster due to cache locality and quick incumbent updates.
    # Let's use LIFO (DFS) but with a check: if bound <= best_score, prune.
    
    import heapq
    # Min-heap for max-bound priority? No, standard heap is min-heap.
    # We want to explore nodes with *highest* upper bound first (Best-First).
    # So we push (-bound, counter, fixed_vars, x_sol)
    
    counter = 0
    nodes = []
    
    # Root node
    fun, x_sol = solve_lp({})
    if fun is not None:
        # fun is negative sum. Upper bound on sum is -fun.
        ub = -fun
        if ub > best_score:
            heapq.heappush(nodes, (-ub, counter, {}, x_sol))
            counter += 1
        else:
            # Even LP is not better than heuristic
            pass

    while nodes and time.time() < deadline - 2.0:
        neg_ub, _, fixed_vars, x_sol = heapq.heappop(nodes)
        ub = -neg_ub
        
        if ub <= best_score:
            continue
            
        # Check if x_sol is integral (within tolerance)
        if np.allclose(x_sol, np.round(x_sol), atol=1e-6):
            sol_int = np.round(x_sol).astype(int)
            sol_size = sol_int.sum()
            if sol_size > best_score:
                best_score = sol_size
                # Construct actual solution
                selected_vertices = [valid_vertices[i] for i in range(n_vars) if sol_int[i] == 1]
                if _is_valid(selected_vertices, squares):
                    best_solution = selected_vertices
            continue
            
        # Fractional solution: Branch
        # Find variable with value closest to 0.5 or just first fractional
        # Heuristic: Branch on variable with highest degree among fractional ones?
        # For now, pick the one with value closest to 0.5
        frac_indices = np.where((x_sol > 1e-6) & (x_sol < 1 - 1e-6))[0]
        if len(frac_indices) == 0:
            continue
            
        # Pick index
        # Simple: first one
        # Better: max degree
        # Let's stick to first one for speed, or random?
        # Let's pick the one with value closest to 0.5
        dists = np.abs(x_sol[frac_indices] - 0.5)
        min_idx_local = np.argmin(dists)
        branch_idx = frac_indices[min_idx_local]
        
        # Branch 1: x_branch_idx = 0
        fixed_0 = dict(fixed_vars)
        fixed_0[branch_idx] = 0
        fun0, x_sol0 = solve_lp(fixed_0)
        if fun0 is not None:
            ub0 = -fun0
            if ub0 > best_score:
                heapq.heappush(nodes, (-ub0, counter, fixed_0, x_sol0))
                counter += 1
                
        # Branch 2: x_branch_idx = 1
        fixed_1 = dict(fixed_vars)
        fixed_1[branch_idx] = 1
        fun1, x_sol1 = solve_lp(fixed_1)
        if fun1 is not None:
            ub1 = -fun1
            if ub1 > best_score:
                heapq.heappush(nodes, (-ub1, counter, fixed_1, x_sol1))
                counter += 1

    # Final verification
    if not _is_valid(best_solution, squares):
        for v in valid_vertices:
            if _is_valid([v], squares):
                return [v]
        return [valid_vertices[0]]
        
    return best_solution

def _is_valid(subset, squares):
    s = set(subset)
    for a in s:
        # Check a=a
        if 2 * a in squares:
            return False
        for b in s:
            if a < b:
                if a + b in squares:
                    return False
    return True
