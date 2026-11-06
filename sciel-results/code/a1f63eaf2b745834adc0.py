import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 256
    
    # Precompute conflict pairs for each x
    # conflict_pairs[x] = list of (a, b) such that {a, b, x} forms a 3-AP
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x is middle: (x-d, x, x+d)
        for d in range(1, min(x, N - x + 1)):
            a, b = x - d, x + d
            if a >= 1 and b <= N:
                conflict_pairs[x].append((a, b))
        # x is largest: (x-2d, x-d, x)
        for d in range(1, (x + 1) // 2 + 1):
            a, b = x - 2 * d, x - d
            if a >= 1 and b <= N:
                conflict_pairs[x].append((a, b))
        # x is smallest: (x, x+d, x+2d)
        for d in range(1, (N - x) // 2 + 1):
            a, b = x + d, x + 2 * d
            if a <= N and b <= N:
                conflict_pairs[x].append((a, b))
    
    def can_add(x, in_set):
        """Check if x can be added given current set represented by bool array (1-indexed)."""
        for a, b in conflict_pairs[x]:
            if in_set[a] and in_set[b]:
                return False
        return True
    
    def compute_score(in_set):
        return sum(1.0 / x for x in range(1, N + 1) if in_set[x])
    
    # --- MILP Initialization ---
    weights = np.array([1.0 / i for i in range(1, N + 1)])
    
    # Generate all 3-term arithmetic progressions
    constraints = []
    for a in range(1, N + 1):
        max_d = (N - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            constraints.append((a - 1, b - 1, c - 1))
    
    num_constraints = len(constraints)
    
    rows = []
    cols = []
    data = []
    for i, (a, b, c) in enumerate(constraints):
        rows.extend([i, i, i])
        cols.extend([a, b, c])
        data.extend([1, 1, 1])
    
    A = csr_matrix((data, (rows, cols)), shape=(num_constraints, N))
    b_vec = np.full(num_constraints, 2.0)
    
    c_obj = -weights
    integrality = np.ones(N)
    lin_cons = LinearConstraint(A, lb=None, ub=b_vec)
    var_bounds = Bounds(lb=0, ub=1)
    
    best_in_set = [False] * (N + 1)
    best_score = 0.0
    best_solution = None
    
    # Try MILP with time limit
    milp_time_limit = 25
    if time.time() - start_time < deadline - 30:
        try:
            res = milp(c=c_obj, constraints=lin_cons, integrality=integrality, bounds=var_bounds,
                       options={'time_limit': milp_time_limit, 'mip_gap': 0.005})
            if res.x is not None:
                sol_mask = (res.x > 0.5).astype(int)
                selected = [i + 1 for i in range(N) if sol_mask[i] == 1]
                if selected:
                    in_set_milp = [False] * (N + 1)
                    for x in selected:
                        in_set_milp[x] = True
                    score_milp = compute_score(in_set_milp)
                    if score_milp > best_score:
                        best_score = score_milp
                        best_in_set = in_set_milp
                        best_solution = selected
        except Exception:
            pass
    
    # If MILP didn't give good result, use greedy
    if best_solution is None or best_score < 1.0:
        in_set_greedy = [False] * (N + 1)
        S_greedy = set()
        for x in range(1, N + 1):
            if can_add(x, in_set_greedy):
                in_set_greedy[x] = True
                S_greedy.add(x)
        score_greedy = compute_score(in_set_greedy)
        if score_greedy > best_score:
            best_score = score_greedy
            best_in_set = in_set_greedy
            best_solution = list(S_greedy)
    
    # --- Local Search Refinement ---
    in_set = best_in_set[:]
    score = best_score
    improved = True
    no_improve_rounds = 0
    
    while improved and time.time() < deadline - 2:
        improved = False
        no_improve_rounds += 1
        
        # Strategy 1: Remove-and-refill
        # Remove each element, then greedily add back what fits
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if not in_set[x]:
                continue
            
            in_set_temp = in_set[:]
            in_set_temp[x] = False
            
            # Greedily add back elements in order of decreasing weight
            for y in range(1, N + 1):
                if in_set_temp[y]:
                    continue
                if can_add(y, in_set_temp):
                    in_set_temp[y] = True
            
            new_score = compute_score(in_set_temp)
            if new_score > score + 1e-12:
                in_set = in_set_temp
                score = new_score
                improved = True
                no_improve_rounds = 0
        
        # Strategy 2: Single swap
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if not in_set[x]:
                continue
            
            in_set_temp = in_set[:]
            in_set_temp[x] = False
            
            best_y = -1
            best_new_score = score
            for y in range(1, N + 1):
                if in_set_temp[y]:
                    continue
                if can_add(y, in_set_temp):
                    new_score = score - 1.0 / x + 1.0 / y
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_y = y
            
            if best_y > 0 and best_new_score > score + 1e-12:
                in_set[x] = False
                in_set[best_y] = True
                score = best_new_score
                improved = True
                no_improve_rounds = 0
        
        # Strategy 3: Pair removal and refill
        if time.time() < deadline - 5:
            elements = [x for x in range(1, N + 1) if in_set[x]]
            # Try removing pairs of larger (less valuable) elements
            elements_sorted = sorted(elements, reverse=True)
            max_pairs = min(30, len(elements_sorted))
            
            for i in range(max_pairs):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, min(i + 15, len(elements_sorted))):
                    if time.time() > deadline - 2:
                        break
                    x1, x2 = elements_sorted[i], elements_sorted[j]
                    
                    in_set_temp = in_set[:]
                    in_set_temp[x1] = False
                    in_set_temp[x2] = False
                    
                    for y in range(1, N + 1):
                        if in_set_temp[y]:
                            continue
                        if can_add(y, in_set_temp):
                            in_set_temp[y] = True
                    
                    new_score = compute_score(in_set_temp)
                    if new_score > score + 1e-12:
                        in_set = in_set_temp
                        score = new_score
                        improved = True
                        no_improve_rounds = 0
        
        # Strategy 4: Multi-add after single removal
        if time.time() < deadline - 3:
            for x in range(1, N + 1):
                if time.time() > deadline - 2:
                    break
                if not in_set[x]:
                    continue
                
                in_set_temp = in_set[:]
                in_set_temp[x] = False
                
                # Try adding up to 2 elements
                candidates = []
                for y in range(1, N + 1):
                    if in_set_temp[y]:
                        continue
                    if can_add(y, in_set_temp):
                        candidates.append(y)
                
                # Try single additions first
                best_improve = 0
                best_adds = []
                for y in candidates:
                    ns = score - 1.0 / x + 1.0 / y
                    if ns - score > best_improve:
                        best_improve = ns - score
                        best_adds = [y]
                
                # Try double additions
                for i in range(len(candidates)):
                    if time.time() > deadline - 2:
                        break
                    for j in range(i + 1, len(candidates)):
                        y1, y2 = candidates[i], candidates[j]
                        in_set_check = in_set_temp[:]
                        in_set_check[y1] = True
                        if can_add(y2, in_set_check):
                            ns = score - 1.0 / x + 1.0 / y1 + 1.0 / y2
                            if ns - score > best_improve:
                                best_improve = ns - score
                                best_adds = [y1, y2]
                
                if best_adds and best_improve > 1e-12:
                    in_set[x] = False
                    for y in best_adds:
                        in_set[y] = True
                    score += best_improve
                    improved = True
                    no_improve_rounds = 0
        
        # Early exit if no improvement for several rounds
        if no_improve_rounds >= 5:
            break
    
    # Convert to list
    result = [x for x in range(1, N + 1) if in_set[x]]
    return result
