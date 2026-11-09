import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP tuples and their bitmasks
    four_ap_masks = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            mask = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1)) | (1 << (a+3*d-1))
            four_ap_masks.append(mask)
    
    # Precompute 3-AP tuples and their bitmasks
    three_ap_masks = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            mask = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1))
            three_ap_masks.append(mask)
    
    # Helper: count 3-APs in a set using bitmasks
    def count_3aps(bits):
        count = 0
        for mask in three_ap_masks:
            if (bits & mask) == mask:
                count += 1
        return count
        
    # Helper: count 4-AP violations in a set using bitmasks
    def count_4aps(bits):
        count = 0
        for mask in four_ap_masks:
            if (bits & mask) == mask:
                count += 1
        return count
        
    # Helper: fast check if adding x creates a 4-AP
    # Precompute for each x, the list of 4-AP masks containing x
    four_ap_by_element = {i: [] for i in range(1, N + 1)}
    for idx, mask in enumerate(four_ap_masks):
        for i in range(1, N + 1):
            if (mask >> (i-1)) & 1:
                four_ap_by_element[i].append(mask)
                
    def is_valid_addition(bits, x):
        x_bit = 1 << (x-1)
        for mask in four_ap_by_element[x]:
            # Check if the other 3 elements are in bits
            others_mask = mask ^ x_bit
            if (bits & others_mask) == others_mask:
                return False
        return True
        
    # Density heuristic greedy construction
    def greedy_density_construction():
        bits = 0
        # Calculate potential 3-AP contribution for each element
        # An element x contributes to 3-APs if it can be the middle, left, or right term
        # We prioritize elements that form many 3-APs without creating 4-APs
        
        # Start with a random subset of size ~N/2
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        # First pass: add elements that don't create 4-APs
        for x in candidates:
            if is_valid_addition(bits, x):
                bits |= (1 << (x-1))
                
        return bits
    
    # Local search with smooth surrogate
    def local_search(initial_bits, steps=100000):
        current_bits = initial_bits
        current_score = count_3aps(current_bits)
        current_violations = count_4aps(current_bits)
        
        best_bits = current_bits
        best_score = current_score
        
        # Smooth surrogate: use soft-max to guide moves
        # We want to maximize 3-APs and minimize 4-APs
        # Use a temperature-based acceptance with smooth penalty
        
        T_start = 10.0
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / steps)
        T = T_start
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Pick a random move: flip a random bit
            x = random.randint(1, N)
            x_bit = 1 << (x-1)
            new_bits = current_bits ^ x_bit
            
            # Calculate new score and violations
            new_score = count_3aps(new_bits)
            new_violations = count_4aps(new_bits)
            
            # Smooth surrogate objective
            # Use log-sum-exp style soft-max for smoothness
            penalty_weight = 50.0
            current_obj = current_score - penalty_weight * math.log(1 + current_violations)
            new_obj = new_score - penalty_weight * math.log(1 + new_violations)
            
            delta_obj = new_obj - current_obj
            
            if delta_obj >= 0:
                # Accept if better or equal
                current_bits = new_bits
                current_score = new_score
                current_violations = new_violations
                if current_score > best_score and current_violations == 0:
                    best_score = current_score
                    best_bits = current_bits
            else:
                # Accept with probability exp(delta_obj / T)
                if random.random() < math.exp(delta_obj / T):
                    current_bits = new_bits
                    current_score = new_score
                    current_violations = new_violations
                    if current_score > best_score and current_violations == 0:
                        best_score = current_score
                        best_bits = current_bits
            
            T *= cooling_rate
            
        return best_bits, best_score
    
    # Multi-Start Strategy
    best_bits = 0
    best_score = -1
    
    # Seed 1: Greedy density construction
    S_bits = greedy_density_construction()
    bits, score = local_search(S_bits)
    if score > best_score:
        best_score = score
        best_bits = bits
        
    # Seed 2-10: Random starts with local search
    for i in range(10):
        if time.time() > deadline - 5:
            break
            
        # Random subset
        S_bits = 0
        for x in range(1, N + 1):
            if random.random() < 0.5:
                S_bits |= (1 << (x-1))
                
        bits, score = local_search(S_bits)
        if score > best_score:
            best_score = score
            best_bits = bits
            
    # Final high-precision polish pass
    if time.time() < deadline - 2:
        # Tighter tolerance, more iterations
        bits, score = local_search(best_bits, steps=50000)
        if score > best_score:
            best_score = score
            best_bits = bits
            
    # Convert best_bits to list
    result = []
    for i in range(1, N + 1):
        if (best_bits >> (i-1)) & 1:
            result.append(i)
            
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
