import time
import random
import numpy as np
import scipy.optimize
import scipy.sparse

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) for d >= 1, a >= 1, a+3d <= N
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(all_aps)
    
    # Precompute AP data as numpy arrays for fast vectorized checks
    # ap_members[i] = [x0, x1, x2, x3] (0-indexed) for AP i
    ap_members = np.array([[x - 1 for x in ap] for ap in all_aps], dtype=np.int32)
    
    # For each element, store which APs it belongs to (as indices into all_aps)
    ap_indices_by_num = [[] for _ in range(N)]
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_by_num[x - 1].append(idx)
    
    # Precompute for each element, the list of (ap_idx, other_positions) 
    # where other_positions are the 0-indexed positions of the other 3 elements in that AP
    # This lets us check: for element x, for each AP containing x, are the other 3 in S?
    element_ap_info = []
    for x in range(1, N + 1):
        info = []
        for ap_idx in ap_indices_by_num[x - 1]:
            members = all_aps[ap_idx]
            # Find position of x in this AP
            pos = members.index(x)
            others = [members[j] - 1 for j in range(4) if j != pos]
            info.append((ap_idx, others))
        element_ap_info.append(info)
    
    def is_valid_set(S):
        """Check if set S contains any 4-AP using numpy"""
        if not S:
            return True
        S_arr = np.zeros(N, dtype=bool)
        for x in S:
            S_arr[x - 1] = True
        # Check all APs: for each AP, are all 4 members in S?
        # ap_members is (num_aps, 4), S_arr[ap_members] gives (num_aps, 4) boolean
        all_in = np.all(S_arr[ap_members], axis=1)
        return not np.any(all_in)
    
    def can_add_element(x, S_arr):
        """Check if x can be added to set represented by boolean array S_arr"""
        x_idx = x - 1
        for ap_idx, others in element_ap_info[x_idx]:
            # Check if all 3 others are in S
            if np.all(S_arr[others]):
                return False
        return True
    
    def greedy_with_order(order, S_init, S_arr_init=None):
        """Greedy add elements in the given order to S_init"""
        if S_arr_init is not None:
            S_arr = S_arr_init.copy()
        else:
            S_arr = np.zeros(N, dtype=bool)
            for x in S_init:
                S_arr[x - 1] = True
        S = set(S_init)
        for x in order:
            if x not in S and can_add_element(x, S_arr):
                S.add(x)
                S_arr[x - 1] = True
        return S, S_arr

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
    S_greedy_low, _ = greedy_with_order(range(1, N + 1), set())
    candidates_for_best.append(S_greedy_low)
    
    # 3. Greedy from high to low
    S_greedy_high, _ = greedy_with_order(range(N, 0, -1), set())
    candidates_for_best.append(S_greedy_high)
    
    # 4. Greedy starting from base-4 set, try to add more
    S_greedy_from_base4, _ = greedy_with_order(range(1, N + 1), set(base4_set))
    candidates_for_best.append(S_greedy_from_base4)
    
    # 5. Randomized greedy with different seeds (from inspiration 1)
    for seed in range(30):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S_rand, _ = greedy_with_order(order, set())
        candidates_for_best.append(S_rand)
    
    best_set = max(candidates_for_best, key=len)
    best_size = len(best_set)
    best_S_arr = np.zeros(N, dtype=bool)
    for x in best_set:
        best_S_arr[x - 1] = True
    
    # Local Search Phase 1: remove one, add many
    for restart in range(5000):
        if time.time() > deadline - 5:
            break
            
        current_set = set(best_set)
        S_arr = best_S_arr.copy()
        elements = list(current_set)
        random.shuffle(elements)
        
        for i in range(len(elements)):
            if time.time() > deadline - 5:
                break
            
            remove_elem = elements[i]
            current_set.remove(remove_elem)
            S_arr[remove_elem - 1] = False
            
            # Try to add candidates
            candidates = [j for j in range(1, N + 1) if j not in current_set]
            random.shuffle(candidates)
            
            added = []
            for c in candidates:
                if can_add_element(c, S_arr):
                    current_set.add(c)
                    S_arr[c - 1] = True
                    added.append(c)
            
            if len(added) > 1:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set
                    best_S_arr = S_arr
                break
            else:
                current_set.add(remove_elem)
                S_arr[remove_elem - 1] = True
                for a in added:
                    if a in current_set:
                        current_set.remove(a)
                        S_arr[a - 1] = False

    # Local Search Phase 2: More aggressive search (from inspiration 1)
    for restart in range(3000):
        if time.time() > deadline - 2:
            break
        
        # Start from best_set with random perturbations
        current_set = set(best_set)
        S_arr = best_S_arr.copy()
        elements = list(current_set)
        random.shuffle(elements)
        
        # Remove 1-3 random elements
        num_remove = random.randint(1, 3)
        for _ in range(num_remove):
            if elements:
                rem = elements.pop()
                current_set.remove(rem)
                S_arr[rem - 1] = False
        
        # Try to add back and new elements
        candidates = [j for j in range(1, N + 1) if j not in current_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if can_add_element(c, S_arr):
                current_set.add(c)
                S_arr[c - 1] = True
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            best_S_arr = S_arr

    # ATTEMPT: Integer Linear Programming (MIS formulation)
    # This is the structural change: exact optimization via MILP
    try:
        if time.time() - start_time < deadline - 5:
            # Build constraint matrix A
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
            options = {
                "time_limit": 20, # Leave time for setup and fallback
                "mip_rel_gap": 0.0
            }
            
            res = scipy.optimize.milp(c, constraints=scipy.optimize.LinearConstraint(A, -np.inf, b), bounds=bounds, integrality=integrality, options=options)
            
            if res.success and res.x is not None:
                sol_set = set()
                for i in range(N):
                    if res.x[i] > 0.5:
                        sol_set.add(i + 1)
                
                if is_valid_set(sol_set):
                    if len(sol_set) > best_size:
                        best_size = len(sol_set)
                        best_set = sol_set
                        best_S_arr = np.zeros(N, dtype=bool)
                        for x in sol_set:
                            best_S_arr[x - 1] = True
            else:
                # Check if we have a feasible solution even if not "success"
                if res.x is not None:
                    sol_set = set()
                    for i in range(N):
                        if res.x[i] > 0.5:
                            sol_set.add(i + 1)
                    if is_valid_set(sol_set) and len(sol_set) > best_size:
                        best_size = len(sol_set)
                        best_set = sol_set
                        best_S_arr = np.zeros(N, dtype=bool)
                        for x in sol_set:
                            best_S_arr[x - 1] = True
                            
    except Exception as e:
        # Fallback if MILP fails or is not available
        pass

    # Final polish local search
    if time.time() - start_time < deadline:
        for _ in range(100):
            if time.time() > deadline:
                break
            current_set = set(best_set)
            S_arr = best_S_arr.copy()
            elements = list(current_set)
            random.shuffle(elements)
            if not elements:
                break
            remove_elem = elements[0]
            current_set.remove(remove_elem)
            S_arr[remove_elem - 1] = False
            candidates = [j for j in range(1, N + 1) if j not in current_set]
            random.shuffle(candidates)
            added = []
            for c in candidates:
                if can_add_element(c, S_arr):
                    current_set.add(c)
                    S_arr[c - 1] = True
                    added.append(c)
            if len(added) > 1:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set
                    best_S_arr = S_arr
                break
            else:
                current_set.add(remove_elem)
                S_arr[remove_elem - 1] = True
                for a in added:
                    if a in current_set:
                        current_set.remove(a)
                        S_arr[a - 1] = False

    return sorted(best_set)
