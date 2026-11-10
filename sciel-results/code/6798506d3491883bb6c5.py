import time
import random
import numpy as np
import scipy.optimize
import scipy.sparse

def construct():
    N = 512
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit

    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) for d >= 1, a >= 1, a+3d <= N
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(all_aps)
    
    # Map each number to the indices of APs it belongs to (for greedy/local search)
    ap_indices_by_num = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_by_num[x].append(idx)

    def is_valid_set(S):
        """Check if set S contains any 4-AP"""
        S_set = set(S) if not isinstance(S, set) else S
        for ap in all_aps:
            if all(x in S_set for x in ap):
                return False
        return True

    def can_add_element(x, S):
        """Check if x can be added to set S without creating a 4-AP"""
        for ap_idx in ap_indices_by_num[x]:
            ap = all_aps[ap_idx]
            count_in_set = 0
            for elem in ap:
                if elem == x:
                    continue
                if elem in S:
                    count_in_set += 1
            if count_in_set == 3:
                return False
        return True

    def greedy_with_order(order, S_init):
        """Greedy add elements in the given order to S_init"""
        S = set(S_init)
        for x in order:
            if x not in S and can_add_element(x, S):
                S.add(x)
        return S

    # Base-4 {0,1} construction (Salem-Spencer style)
    base4_set = set()
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_set.add(i)
    
    if not is_valid_set(base4_set):
        base4_set = set()
    
    # Initial best from various heuristics
    candidates_for_best = []
    
    # 1. Base-4 set
    candidates_for_best.append(base4_set)
    
    # 2. Greedy from low to high
    S_greedy_low = greedy_with_order(range(1, N + 1), set())
    candidates_for_best.append(S_greedy_low)
    
    # 3. Greedy from high to low
    S_greedy_high = greedy_with_order(range(N, 0, -1), set())
    candidates_for_best.append(S_greedy_high)
    
    # 4. Greedy starting from base-4 set, try to add more
    S_greedy_from_base4 = greedy_with_order(range(1, N + 1), set(base4_set))
    candidates_for_best.append(S_greedy_from_base4)
    
    best_set = max(candidates_for_best, key=len)
    best_size = len(best_set)
    
    # Local Search: try to improve by removing one element and adding others
    # This provides a strong lower bound for the ILP
    for restart in range(500):
        if time.time() - start_time > deadline - 10:
            break
            
        current_set = set(best_set)
        elements = list(current_set)
        random.shuffle(elements)
        
        for i in range(len(elements)):
            if time.time() - start_time > deadline - 10:
                break
            
            remove_elem = elements[i]
            current_set.remove(remove_elem)
            
            # Try to add candidates
            candidates = [j for j in range(1, N + 1) if j not in current_set]
            random.shuffle(candidates)
            
            added = []
            for c in candidates:
                if can_add_element(c, current_set):
                    current_set.add(c)
                    added.append(c)
            
            if len(added) > 1:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set
                break
            else:
                current_set.add(remove_elem)
                for a in added:
                    if a in current_set:
                        current_set.remove(a)

    # ATTEMPT 1: Integer Linear Programming (MIS formulation)
    # This is the structural change: exact optimization via MILP
    try:
        if time.time() - start_time < deadline - 5:
            # Build constraint matrix A
            # Rows: APs, Cols: Elements 1..N
            # A[i, j] = 1 if element j+1 is in AP i
            # We want A @ x <= 3
            # Maximize sum(x)
            
            rows = []
            cols = []
            data = []
            
            for idx, ap in enumerate(all_aps):
                for x in ap:
                    rows.append(idx)
                    cols.append(x - 1) # 0-indexed for numpy
                    data.append(1)
            
            A = scipy.sparse.csr_array((data, (rows, cols)), shape=(num_aps, N))
            
            # Objective: maximize sum(x) -> minimize -sum(x)
            c = -np.ones(N)
            
            # Bounds: 0 <= x_i <= 1
            bounds = [(0, 1) for _ in range(N)]
            
            # Constraints: A x <= 3
            b = np.full(num_aps, 3)
            
            # Integers: all variables
            integrality = np.ones(N)
            
            # Run MILP
            # time_limit is supported in HiGHS via options
            # Note: scipy.optimize.milp might not accept time_limit directly in older versions?
            # It accepts options dict.
            options = {
                "time_limit": 40, # Leave 10s for setup and fallback
                "mip_rel_gap": 0.0
            }
            
            res = scipy.optimize.milp(c, constraints=scipy.optimize.LinearConstraint(A, -np.inf, b), bounds=bounds, integrality=integrality, options=options)
            
            if res.success and res.x is not None:
                # res.x is the solution
                # Verify it
                sol_set = set()
                for i in range(N):
                    if res.x[i] > 0.5:
                        sol_set.add(i + 1)
                
                if is_valid_set(sol_set):
                    if len(sol_set) > best_size:
                        best_size = len(sol_set)
                        best_set = sol_set
                    else:
                        # If ILP found a smaller set (shouldn't happen if it's optimal),
                        # keep the larger one. But if ILP proved optimality, they should be same.
                        pass
                else:
                    # Invalid solution from solver? Fall back.
                    pass
            else:
                # If not success, check if we have a feasible solution
                if res.x is not None and res.status in [1, 2]: # 1: optimal, 2: infeasible? No, 2 is infeasible. 3: unbounded. 4: iteration limit?
                    # HiGHS status:
                    # 0: optimal
                    # 1: infeasible
                    # 2: unbounded
                    # 3: iteration limit
                    # 4: time limit
                    # 5: numerical error
                    # We want status 0 or 4 (time limit with incumbent)
                    if res.status == 0 or res.status == 4:
                        sol_set = set()
                        for i in range(N):
                            if res.x[i] > 0.5:
                                sol_set.add(i + 1)
                        if is_valid_set(sol_set) and len(sol_set) > best_size:
                            best_size = len(sol_set)
                            best_set = sol_set
                            
    except Exception as e:
        # Fallback if MILP fails or is not available
        pass

    # If we have time, do a final quick local search polish on the best set
    if time.time() - start_time < deadline:
        for _ in range(100):
            if time.time() - start_time > deadline:
                break
            current_set = set(best_set)
            elements = list(current_set)
            random.shuffle(elements)
            if not elements:
                break
            remove_elem = elements[0]
            current_set.remove(remove_elem)
            candidates = [j for j in range(1, N + 1) if j not in current_set]
            random.shuffle(candidates)
            added = []
            for c in candidates:
                if can_add_element(c, current_set):
                    current_set.add(c)
                    added.append(c)
            if len(added) > 1:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set
                break
            else:
                current_set.add(remove_elem)
                for a in added:
                    if a in current_set:
                        current_set.remove(a)

    return sorted(best_set)
