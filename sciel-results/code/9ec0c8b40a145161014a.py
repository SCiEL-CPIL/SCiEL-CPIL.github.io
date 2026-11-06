import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 145.0
    
    N = 512
    max_a = N - 1
    
    # Pre-compute weights
    weights = [0.0] * (max_a + 1)
    for a in range(1, max_a + 1):
        weights[a] = 1.0 / (N - a)
        
    # Pre-compute conflict masks
    # conflict_mask[a] has bit (b-1) set if gcd(a, b) > 1
    conflict_mask = [0] * (max_a + 1)
    for a in range(1, max_a + 1):
        mask = 0
        for b in range(1, max_a + 1):
            if a != b and gcd(a, b) > 1:
                mask |= (1 << (b - 1))
        conflict_mask[a] = mask
        
    best_weight = 0.0
    best_mask = 0
    best_list = []
    
    def update_best(weight, mask):
        nonlocal best_weight, best_mask, best_list
        if weight > best_weight:
            best_weight = weight
            best_mask = mask
            lst = []
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                lst.append(idx + 1)
                m ^= lsb
            best_list = lst
            
    def branch_and_bound(avail, current_weight, current_mask):
        nonlocal best_weight, best_mask, best_list
        nonlocal start_time, deadline
        
        if time.time() > deadline:
            return
            
        if current_weight > best_weight:
            update_best(current_weight, current_mask)
            
        if avail == 0:
            return
            
        # Calculate bound
        sum_avail = 0.0
        tmp = avail
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            sum_avail += weights[idx + 1]
            tmp ^= lsb
            
        if current_weight + sum_avail <= best_weight:
            return
            
        # Pick highest weight available vertex
        # Weights are monotonic with value (higher value -> higher weight)
        # So highest set bit is highest weight
        v_idx = avail.bit_length() - 1
        v = v_idx + 1
        
        # Branch 1: Include v
        forbidden = conflict_mask[v] | (1 << v_idx)
        new_avail = avail & ~forbidden
        branch_and_bound(new_avail, current_weight + weights[v], current_mask | (1 << v_idx))
        
        # Branch 2: Exclude v
        new_avail_excl = avail & ~(1 << v_idx)
        branch_and_bound(new_avail_excl, current_weight, current_mask)
        
    initial_avail = (1 << max_a) - 1
    branch_and_bound(initial_avail, 0.0, 0)
    
    return best_list
