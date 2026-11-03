import time
import sys
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    total_space = q ** n
    
    # 1. Precompute neighbors
    neighbors = [[] for _ in range(total_space)]
    
    def get_digits(num):
        digits = [0] * n
        for i in range(n):
            digits[i] = num % q
            num //= q
        return digits

    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q

    for i in range(total_space):
        digits = get_digits(i)
        neighbors[i].append(i) # self
        for pos in range(n):
            orig_val = digits[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                delta = (val - orig_val) * q_powers[pos]
                new_index = i + delta
                neighbors[i].append(new_index)
        
        neighbors[i] = np.array(neighbors[i], dtype=np.int32)

    # 2. Greedy Phase with Gain Array
    uncovered_mask = np.ones(total_space, dtype=bool)
    gain = np.zeros(total_space, dtype=np.int32)
    for w in range(total_space):
        gain[w] = len(neighbors[w])
        
    code_list = []
    code_set = set()
    
    while uncovered_mask.any():
        if time.time() > deadline - 10:
            break
            
        # Find w with max gain
        w_star = np.argmax(gain)
        
        if gain[w_star] == 0:
            break
            
        code_list.append(w_star)
        code_set.add(w_star)
        
        # Update uncovered_mask and gains
        # Words covered by w_star are neighbors[w_star]
        # For each u in neighbors[w_star]:
        #   if uncovered_mask[u]:
        #      uncovered_mask[u] = False
        #      for v in neighbors[u]:
        #         gain[v] -= 1
        
        for u in neighbors[w_star]:
            if uncovered_mask[u]:
                uncovered_mask[u] = False
                # Update gains for all v that cover u
                # v is in neighbors[u]
                gain[neighbors[u]] -= 1
                
    # 3. Annealing Phase
    def build_coverage(code_set):
        cov = np.zeros(total_space, dtype=np.int32)
        for c in code_set:
            cov[neighbors[c]] += 1
        return cov

    coverage = build_coverage(code_set)
    
    def can_remove(c):
        return np.min(coverage[neighbors[c]]) > 1

    def is_valid_swap(c_out, c_in):
        # Check if removing c_out and adding c_in keeps coverage >= 1
        # Words at risk are in neighbors[c_out] \ neighbors[c_in]
        # If min coverage in neighbors[c_out] > 1, it's safe.
        min_cov_out = np.min(coverage[neighbors[c_out]])
        if min_cov_out > 1:
            return True
            
        # If min_cov_out == 1, check if the critical words are covered by c_in
        mask = coverage[neighbors[c_out]] == 1
        if not np.any(mask):
            return True
        critical_words = neighbors[c_out][mask]
        # Check if critical_words are subset of neighbors[c_in]
        # Using in1d is efficient for small arrays
        return np.all(np.isin(critical_words, neighbors[c_in]))

    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    
    while T > min_T and time.time() < deadline:
        # 1. Try to remove
        if code_list:
            c = random.choice(code_list)
            if can_remove(c):
                code_list.remove(c)
                code_set.remove(c)
                coverage[neighbors[c]] -= 1
                continue
        
        # 2. Try a swap
        if not code_list:
            break
            
        c_out = random.choice(code_list)
        c_in = random.randint(0, total_space - 1)
        if c_in in code_set:
            T *= cooling
            continue
            
        if is_valid_swap(c_out, c_in):
            # Accept swap
            code_list.remove(c_out)
            code_set.remove(c_out)
            code_list.append(c_in)
            code_set.add(c_in)
            
            coverage[neighbors[c_out]] -= 1
            coverage[neighbors[c_in]] += 1
            
            # After swap, try to remove c_in
            if can_remove(c_in):
                code_list.remove(c_in)
                code_set.remove(c_in)
                coverage[neighbors[c_in]] -= 1
        else:
            pass
            
        T *= cooling

    # Final conversion
    result = []
    for idx in code_list:
        digits = get_digits(idx)
        result.append(digits)
        
    return result
