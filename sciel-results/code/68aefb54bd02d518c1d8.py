import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55.0
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1
    ALL_ELEMENTS = list(range(0, 129))
    
    # State
    A = set(range(64, 129))
    M = np.zeros(NUM_SUMS, dtype=np.int32)
    
    # Init M
    A_list = list(A)
    for i in range(len(A_list)):
        a = A_list[i]
        for j in range(i, len(A_list)):
            b = A_list[j]
            s = a + b
            if MIN_SUM <= s <= MAX_SUM:
                M[s - MIN_SUM] += 1
                
    J = float(np.sum(M**2))
    current_max = int(np.max(M))
    
    best_max = current_max
    best_J = J
    best_A = set(A)
    
    def get_remove_delta_indices(x, A_set):
        indices = []
        for a in A_set:
            if a == x: continue
            s = x + a
            if MIN_SUM <= s <= MAX_SUM:
                indices.append(s - MIN_SUM)
        s = 2 * x
        if MIN_SUM <= s <= MAX_SUM:
            indices.append(s - MIN_SUM)
        return indices

    def get_add_delta_indices(x, A_set):
        indices = []
        for a in A_set:
            s = x + a
            if MIN_SUM <= s <= MAX_SUM:
                indices.append(s - MIN_SUM)
        s = 2 * x
        if MIN_SUM <= s <= MAX_SUM:
            indices.append(s - MIN_SUM)
        return indices

    def calc_J_delta(M_arr, delta_indices, sign):
        if not delta_indices:
            return 0.0
        sum_M = np.sum(M_arr[delta_indices])
        return 2.0 * sign * sum_M + len(delta_indices)

    def is_valid_after_change(M_arr, delta_indices, sign):
        if sign > 0:
            return True
        for idx in delta_indices:
            if M_arr[idx] == 1:
                return False
        return True

    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # 1. Greedy Removal
        best_rem_x = None
        best_rem_delta_J = 0.0
        best_rem_indices = []
        
        for x in A:
            indices = get_remove_delta_indices(x, A)
            if not is_valid_after_change(M, indices, -1):
                continue
            delta_J = calc_J_delta(M, indices, -1)
            if delta_J < best_rem_delta_J:
                best_rem_delta_J = delta_J
                best_rem_x = x
                best_rem_indices = indices
        
        if best_rem_x is not None:
            x = best_rem_x
            M[best_rem_indices] -= 1
            J += best_rem_delta_J
            A.remove(x)
            current_max = int(np.max(M))
            
            if current_max < best_max or (current_max == best_max and J < best_J):
                best_max = current_max
                best_J = J
                best_A = set(A)
                no_improve_count = 0
            else:
                no_improve_count += 1
            improved = True

        # 2. Greedy Swap
        if not improved:
            A_list = list(A)
            candidates = [y for y in ALL_ELEMENTS if y not in A]
            
            best_swap = None
            best_swap_delta_J = 0.0
            best_swap_x = None
            best_swap_y = None
            best_swap_rem_indices = []
            best_swap_add_indices = []
            
            # Limit swap search if A is large? No, A is small.
            for x in A_list:
                rem_indices = get_remove_delta_indices(x, A)
                if not is_valid_after_change(M, rem_indices, -1):
                    continue
                
                delta_J_rem = calc_J_delta(M, rem_indices, -1)
                
                # Precompute rem_indices as a set for fast intersection?
                rem_set = set(rem_indices)
                
                for y in candidates:
                    add_indices = get_add_delta_indices(y, A - {x})
                    
                    # Calculate sum_M_rem on add_indices
                    # sum(M_rem[i] for i in add_indices)
                    # M_rem[i] = M[i] - 1 if i in rem_indices else M[i]
                    
                    # Optimization:
                    # sum_M = sum(M[add_indices])
                    # intersection_count = len(set(add_indices) & rem_set)
                    # sum_M_rem = sum_M - intersection_count
                    
                    sum_M = np.sum(M[add_indices])
                    intersection_count = len(set(add_indices) & rem_set)
                    sum_M_rem = sum_M - intersection_count
                    
                    delta_J_add = 2.0 * 1.0 * sum_M_rem + len(add_indices)
                    
                    total_delta = delta_J_rem + delta_J_add
                    
                    if total_delta < best_swap_delta_J:
                        best_swap_delta_J = total_delta
                        best_swap = True
                        best_swap_x = x
                        best_swap_y = y
                        best_swap_rem_indices = rem_indices
                        best_swap_add_indices = add_indices
            
            if best_swap:
                x = best_swap_x
                y = best_swap_y
                # Apply
                M[best_swap_rem_indices] -= 1
                M[best_swap_add_indices] += 1
                J += best_swap_delta_J
                A.remove(x)
                A.add(y)
                current_max = int(np.max(M))
                
                if current_max < best_max or (current_max == best_max and J < best_J):
                    best_max = current_max
                    best_J = J
                    best_A = set(A)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                improved = True

        # 3. Simulated Annealing Step (if stuck)
        if not improved:
            # Random move
            if random.random() < 0.5:
                # Remove
                if len(A) > 20:
                    x = random.choice(list(A))
                    indices = get_remove_delta_indices(x, A)
                    if is_valid_after_change(M, indices, -1):
                        delta_J = calc_J_delta(M, indices, -1)
                        if delta_J < 0 or random.random() < np.exp(-delta_J / T):
                            M[indices] -= 1
                            J += delta_J
                            A.remove(x)
                            current_max = int(np.max(M))
                            if current_max < best_max or (current_max == best_max and J < best_J):
                                best_max = current_max
                                best_J = J
                                best_A = set(A)
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
            else:
                # Swap
                if len(A) > 0:
                    x = random.choice(list(A))
                    candidates = [y for y in ALL_ELEMENTS if y not in A]
                    if candidates:
                        y = random.choice(candidates)
                        rem_indices = get_remove_delta_indices(x, A)
                        if is_valid_after_change(M, rem_indices, -1):
                            add_indices = get_add_delta_indices(y, A - {x})
                            delta_J_rem = calc_J_delta(M, rem_indices, -1)
                            rem_set = set(rem_indices)
                            sum_M = np.sum(M[add_indices])
                            intersection_count = len(set(add_indices) & rem_set)
                            sum_M_rem = sum_M - intersection_count
                            delta_J_add = 2.0 * sum_M_rem + len(add_indices)
                            total_delta = delta_J_rem + delta_J_add
                            
                            if total_delta < 0 or random.random() < np.exp(-total_delta / T):
                                M[rem_indices] -= 1
                                M[add_indices] += 1
                                J += total_delta
                                A.remove(x)
                                A.add(y)
                                current_max = int(np.max(M))
                                if current_max < best_max or (current_max == best_max and J < best_J):
                                    best_max = current_max
                                    best_J = J
                                    best_A = set(A)
                                    no_improve_count = 0
                                else:
                                    no_improve_count += 1

        T = max(T * cooling, min_T)
        
        # Restart if stuck too long
        if no_improve_count > 100:
            # Perturb best_A
            A = set(best_A)
            # Randomly swap a few elements
            for _ in range(5):
                if len(A) == 0: break
                x = random.choice(list(A))
                candidates = [y for y in ALL_ELEMENTS if y not in A]
                if not candidates: continue
                y = random.choice(candidates)
                # Just apply blindly, will be fixed by search
                A.remove(x)
                A.add(y)
            # Recompute M and J
            M = np.zeros(NUM_SUMS, dtype=np.int32)
            A_list = list(A)
            for i in range(len(A_list)):
                a = A_list[i]
                for j in range(i, len(A_list)):
                    b = A_list[j]
                    s = a + b
                    if MIN_SUM <= s <= MAX_SUM:
                        M[s - MIN_SUM] += 1
            J = float(np.sum(M**2))
            current_max = int(np.max(M))
            no_improve_count = 0
            T = 5.0

    return list(best_A)
