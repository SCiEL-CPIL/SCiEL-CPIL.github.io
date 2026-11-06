import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM of denominators 2..48
    def lcm(a, b):
        return a // math.gcd(a, b) * b
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Integer weights: w_i = L / d_i
    weights = [L // d for d in denominators]
    # Constraint: sum(w_i) < L (strictly)
    # Equivalently: sum(w_i) <= L - 1
    
    best_indices = [0] # Fallback: {2} -> sum 1/2 < 1
    best_weight_sum = weights[0]
    
    def local_search(initial_indices):
        indices = initial_indices.copy()
        curr_wsum = sum(weights[i] for i in indices)
        in_set = set(indices)
        
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any single element that fits
            best_add_i = -1
            best_add_wsum = curr_wsum
            for i in range(n):
                if i in in_set:
                    continue
                new_wsum = curr_wsum + weights[i]
                if new_wsum < L and new_wsum > best_add_wsum:
                    best_add_wsum = new_wsum
                    best_add_i = i
            
            if best_add_i != -1:
                indices.append(best_add_i)
                in_set.add(best_add_i)
                curr_wsum = best_add_wsum
                improved = True
                continue
                
            # 2. Try to swap one out, one in
            # Maximize gain = w_in - w_out such that curr_wsum + gain < L
            best_swap_gain = 0
            best_swap_out = -1
            best_swap_in = -1
            
            for out_i in indices:
                w_out = weights[out_i]
                for in_i in range(n):
                    if in_i in in_set:
                        continue
                    w_in = weights[in_i]
                    gain = w_in - w_out
                    if gain <= 0:
                        continue
                    new_wsum = curr_wsum + gain
                    if new_wsum < L and gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_out = out_i
                        best_swap_in = in_i
                        
            if best_swap_gain > 0:
                indices.remove(best_swap_out)
                in_set.remove(best_swap_out)
                indices.append(best_swap_in)
                in_set.add(best_swap_in)
                curr_wsum += best_swap_gain
                improved = True
                continue
                
            # 3. Try 2-out, 1-in moves
            # Remove two, add one. Gain = w_in - w_out1 - w_out2
            best_21_gain = 0
            best_21_out1 = -1
            best_21_out2 = -1
            best_21_in = -1
            
            m = len(indices)
            for a in range(m):
                for b in range(a + 1, m):
                    out1 = indices[a]
                    out2 = indices[b]
                    w_out = weights[out1] + weights[out2]
                    for in_i in range(n):
                        if in_i in in_set:
                            continue
                        w_in = weights[in_i]
                        gain = w_in - w_out
                        if gain <= 0:
                            continue
                        new_wsum = curr_wsum + gain
                        if new_wsum < L and gain > best_21_gain:
                            best_21_gain = gain
                            best_21_out1 = out1
                            best_21_out2 = out2
                            best_21_in = in_i
            
            if best_21_gain > 0:
                indices.remove(best_21_out1)
                indices.remove(best_21_out2)
                in_set.remove(best_21_out1)
                in_set.remove(best_21_out2)
                indices.append(best_21_in)
                in_set.add(best_21_in)
                curr_wsum += best_21_gain
                improved = True
                continue
                
            # 4. Try 1-out, 2-in moves
            # Remove one, add two. Gain = w_in1 + w_in2 - w_out
            best_12_gain = 0
            best_12_out = -1
            best_12_in1 = -1
            best_12_in2 = -1
            
            for out_i in indices:
                w_out = weights[out_i]
                # Find best pair of non-set elements to add
                # To keep it efficient, only try pairs where both are not in set
                # We can precompute available elements
                avail = [i for i in range(n) if i not in in_set]
                for a in range(len(avail)):
                    for b in range(a + 1, len(avail)):
                        in1 = avail[a]
                        in2 = avail[b]
                        w_in = weights[in1] + weights[in2]
                        gain = w_in - w_out
                        if gain <= 0:
                            continue
                        new_wsum = curr_wsum + gain
                        if new_wsum < L and gain > best_12_gain:
                            best_12_gain = gain
                            best_12_out = out_i
                            best_12_in1 = in1
                            best_12_in2 = in2
            
            if best_12_gain > 0:
                indices.remove(best_12_out)
                in_set.remove(best_12_out)
                indices.append(best_12_in1)
                indices.append(best_12_in2)
                in_set.add(best_12_in1)
                in_set.add(best_12_in2)
                curr_wsum += best_12_gain
                improved = True
                continue
                
        return indices, curr_wsum

    # Initial deterministic greedy: sort by weight descending, add if fits
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    init_indices = []
    for i in order:
        ns = sum(weights[j] for j in init_indices) + weights[i]
        if ns < L:
            init_indices.append(i)
            
    idx, wsum = local_search(init_indices)
    if wsum > best_weight_sum:
        best_weight_sum = wsum
        best_indices = idx
        
    # Random restarts with diverse initializations
    while time.time() < deadline:
        # Random permutation greedy
        order = list(range(n))
        random.shuffle(order)
        curr_indices = []
        curr_wsum = 0
        for i in order:
            ns = curr_wsum + weights[i]
            if ns < L:
                curr_indices.append(i)
                curr_wsum = ns
                
        idx, wsum = local_search(curr_indices)
        if wsum > best_weight_sum:
            best_weight_sum = wsum
            best_indices = idx
            
        # Complement-based restart: start from complement of best, then search
        if len(best_indices) > 1:
            comp_indices = [i for i in range(n) if i not in set(best_indices)]
            # Greedy fill from complement
            comp_order = sorted(comp_indices, key=lambda i: weights[i], reverse=True)
            comp_curr_indices = []
            comp_curr_wsum = 0
            for i in comp_order:
                ns = comp_curr_wsum + weights[i]
                if ns < L:
                    comp_curr_indices.append(i)
                    comp_curr_wsum = ns
            if comp_curr_indices:
                idx, wsum = local_search(comp_curr_indices)
                if wsum > best_weight_sum:
                    best_weight_sum = wsum
                    best_indices = idx
                    
    return best_indices

if __name__ == "__main__":
    res = construct()
    denominators = list(range(2, 49))
    s = Fraction(0)
    for idx in res:
        s += Fraction(1, denominators[idx])
    print(f"Indices: {res}")
    print(f"Sum: {s}")
    print(f"Deficit: {1 - s}")
