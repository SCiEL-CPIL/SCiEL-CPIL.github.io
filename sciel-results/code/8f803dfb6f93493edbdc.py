import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 52.0  # Safety margin for 90s limit
    
    N = 256
    max_sum = 2 * N
    
    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
        
    # 2. Filter vertices: 2*v must not be a square
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    if not valid_vertices:
        return [1]
        
    M = len(valid_vertices)
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 3. Build conflict adjacency list (index-based for speed)
    adj = [[] for _ in range(M)]
    for i in range(M):
        vi = valid_vertices[i]
        for j in range(i + 1, M):
            vj = valid_vertices[j]
            if vi + vj in squares:
                adj[i].append(j)
                adj[j].append(i)
    
    # 4. Solve using SciPy MILP
    milp_solution = None
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Build constraint pairs
        constraint_pairs = []
        for i in range(M):
            for j in adj[i]:
                if i < j:
                    constraint_pairs.append((i, j))
        
        if not constraint_pairs:
            return list(valid_vertices)
        
        # Build dense constraint matrix
        A = np.zeros((len(constraint_pairs), M))
        b = np.ones(len(constraint_pairs))
        for k, (i, j) in enumerate(constraint_pairs):
            A[k, i] = 1
            A[k, j] = 1
            
        c = -np.ones(M)
        lincons = LinearConstraint(A, ub=b)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(M)
        
        try:
            res = milp(c=c, constraints=lincons, integrality=integrality, bounds=bounds,
                       options={'time_limit': 25, 'mip_gap': 1e-10})
        except TypeError:
            res = milp(c=c, constraints=lincons, integrality=integrality, bounds=bounds)
        
        if res.success and res.x is not None:
            selected_indices = [i for i, val in enumerate(res.x) if val > 0.5]
            if selected_indices:
                # Verify solution
                sol_set = set(selected_indices)
                valid = True
                for i in sol_set:
                    for j in adj[i]:
                        if j in sol_set and i < j:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    milp_solution = selected_indices
    except Exception:
        pass

    # 5. Local search using index-based conflict tracking
    def is_compatible_idx(idx, S):
        for neighbor in adj[idx]:
            if neighbor in S:
                return False
        return True

    def local_search(S):
        S = set(S)
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            # Try adding
            for idx in range(M):
                if idx not in S:
                    if is_compatible_idx(idx, S):
                        S.add(idx)
                        improved = True
            # Try swap: remove one, add many
            if not improved:
                best_move = None
                best_gain = 0
                for u in list(S):
                    S_temp = S - {u}
                    gain = 0
                    changed = True
                    while changed:
                        changed = False
                        for idx in range(M):
                            if idx not in S_temp:
                                if is_compatible_idx(idx, S_temp):
                                    S_temp.add(idx)
                                    changed = True
                                    gain += 1
                    net_gain = gain - 1
                    if net_gain > best_gain:
                        best_gain = net_gain
                        best_move = S_temp
                if best_move is not None and best_gain > 0:
                    S = best_move
                    improved = True
        return S

    # 6. Get initial solution
    if milp_solution is not None:
        best_S = set(milp_solution)
        best_S = local_search(best_S)
    else:
        # Fallback: Greedy init
        sorted_v = sorted(range(M), key=lambda i: len(adj[i]))
        S = set()
        for idx in sorted_v:
            if is_compatible_idx(idx, S):
                S.add(idx)
        best_S = local_search(S)
    
    # 7. Aggressive randomized local search
    best_size = len(best_S)
    no_improve_count = 0
    
    while time.time() < deadline - 2.0:
        if len(best_S) > 3:
            # Perturb: remove random elements
            remove_count = random.randint(2, 6)
            removals = random.sample(list(best_S), min(remove_count, len(best_S)))
            perturbed_S = set(best_S)
            for r in removals:
                perturbed_S.remove(r)
            
            improved_S = local_search(perturbed_S)
            
            if len(improved_S) > best_size:
                best_S = improved_S
                best_size = len(best_S)
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Occasionally try a different perturbation strategy
                if no_improve_count % 5 == 0 and len(best_S) > 4:
                    # Swap-based perturbation
                    u = random.choice(list(best_S))
                    perturbed_S = set(best_S)
                    perturbed_S.remove(u)
                    # Try to add a few random compatible elements
                    candidates = [i for i in range(M) if i not in perturbed_S]
                    random.shuffle(candidates)
                    for idx in candidates[:5]:
                        if is_compatible_idx(idx, perturbed_S):
                            perturbed_S.add(idx)
                    improved_S = local_search(perturbed_S)
                    if len(improved_S) > best_size:
                        best_S = improved_S
                        best_size = len(best_S)
                        no_improve_count = 0
    
    # 8. Final polish: try to add any remaining compatible elements
    final_S = set(best_S)
    for idx in range(M):
        if idx not in final_S:
            if is_compatible_idx(idx, final_S):
                final_S.add(idx)
    best_S = final_S
    
    # 9. Convert back to values
    result = [valid_vertices[i] for i in sorted(best_S)]
    if not result:
        result = [valid_vertices[0]]
    
    return result
