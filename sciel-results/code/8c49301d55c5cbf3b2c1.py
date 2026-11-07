import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin for 90s limit
    
    def can_add(S, x):
        """Check if S ∪ {x} is valid. Any new violation must involve x."""
        V = [v for v in S if (x + v) <= 128 and (x + v) in S]
        n = len(V)
        if n < 2:
            return True
        for i in range(n):
            b = V[i]
            for j in range(i + 1, n):
                c = V[j]
                s = b + c
                if s == x or s in S:
                    return False
        return True

    def is_valid(S):
        """Full validity check."""
        S_list = sorted(S)
        n = len(S_list)
        S_set = set(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (a + b) not in S_set:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if (a + c) in S_set and (b + c) in S_set:
                        return False
        return True

    # Try MILP first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # Generate cores
        cores = []
        for a in range(1, 129):
            for b in range(a + 1, 129):
                max_c = 128 - b
                if max_c <= b:
                    continue
                for c in range(b + 1, max_c + 1):
                    s1 = a + b
                    s2 = a + c
                    s3 = b + c
                    cores.append((a, b, c, s1, s2, s3))
        
        num_cores = len(cores)
        if num_cores == 0:
            return list(range(1, 129)) # Trivial case, unlikely
            
        c_obj = np.ones(128) * -1
        
        rows = []
        cols = []
        data = []
        for idx, core in enumerate(cores):
            for val in core:
                rows.append(idx)
                cols.append(val - 1)
                data.append(1)
                
        A_sparse = sp.csr_matrix((data, (rows, cols)), shape=(num_cores, 128))
        b_ub = np.ones(num_cores) * 5
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(128)
        constraint = LinearConstraint(A_sparse, ub=b_ub)
        
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            raise Exception("Not enough time")
            
        res = milp(c_obj, constraints=constraint, integrality=integrality, bounds=bounds, options={'time_limit': remaining_time - 2})
        
        if res.x is not None:
            sol = np.round(res.x).astype(int)
            S = set()
            for i, val in enumerate(sol):
                if val == 1:
                    S.add(i + 1)
            
            if is_valid(S):
                return [int(x) for x in S]
            else:
                # Cleanup
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    for core in cores:
                        if all(v in S for v in core):
                            S.remove(core[0])
                            changed = True
                            break
                return [int(x) for x in S]
                
    except Exception:
        pass

    # Fallback Heuristic
    best_set = set()
    best_size = 0
    
    random.seed(42)
    
    def greedy_random_order():
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = set()
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        return S

    def greedy_ascending():
        S = set()
        for x in range(1, 129):
            if can_add(S, x):
                S.add(x)
        return S

    def greedy_descending():
        S = set()
        for x in range(128, 0, -1):
            if can_add(S, x):
                S.add(x)
        return S

    def odd_numbers():
        return set(range(1, 129, 2))

    def high_numbers():
        return set(range(65, 129))

    strategies = [
        greedy_ascending,
        greedy_descending,
        odd_numbers,
        high_numbers,
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        S = strategy()
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        S = greedy_random_order()
        
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            added = False
            for x in range(1, 129):
                if x in S:
                    continue
                if can_add(S, x):
                    S.add(x)
                    added = True
                    break
            
            if not added:
                improved = False
                S_list = list(S)
                not_in_S = [x for x in range(1, 129) if x not in S]
                random.shuffle(S_list)
                random.shuffle(not_in_S)
                
                for x in not_in_S[:20]:
                    for y in S_list[:30]:
                        S_temp = S - {y}
                        if can_add(S_temp, x):
                            S = S_temp
                            S.add(x)
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    improved = False
                    random.shuffle(S_list)
                    random.shuffle(not_in_S)
                    for x in not_in_S[:40]:
                        for y in S_list[:50]:
                            S_temp = S - {y}
                            if can_add(S_temp, x):
                                S = S_temp
                                S.add(x)
                                improved = True
                                break
                        if improved:
                            break
                    if not improved:
                        break
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 129):
            if x in best_set:
                continue
            if can_add(best_set, x):
                best_set.add(x)
                changed = True

    if not is_valid(best_set):
        best_set = set(range(1, 129, 2))

    result = [int(x) for x in best_set]
    return result
