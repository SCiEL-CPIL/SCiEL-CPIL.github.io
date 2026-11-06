import time
import random
import math

def construct():
    start_time = time.time()
    # 120s limit, use 115s for search to be safe
    deadline = start_time + 115
    
    N = 32
    MAX_VAL = 512
    MAX_D = MAX_VAL // 2  # 255
    
    def score_bitset(S):
        count = 0
        # Check d from 1 to 255
        # S & (S >> d) & (S >> 2*d)
        # If non-zero, d is a valid difference
        for d in range(1, MAX_D + 1):
            if (S & (S >> d) & (S >> 2*d)) != 0:
                count += 1
        return count

    def random_bitset():
        elements = random.sample(range(1, MAX_VAL + 1), N)
        S = 0
        for e in elements:
            S |= (1 << (e - 1))
        return S

    # Initial random start
    current_S = random_bitset()
    current_score = score_bitset(current_S)
    
    best_S = current_S
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # Generate move: flip two bits
        i = random.randint(0, MAX_VAL - 1)
        bit_i = (current_S >> i) & 1
        
        # Find a j with opposite bit
        # Try a few times
        j = -1
        for _ in range(20):
            j_candidate = random.randint(0, MAX_VAL - 1)
            if j_candidate == i:
                continue
            bit_j_candidate = (current_S >> j_candidate) & 1
            if bit_i != bit_j_candidate:
                j = j_candidate
                break
        
        if j == -1:
            continue
            
        if bit_i:
            # Remove i, Add j
            new_S = (current_S & ~(1 << i)) | (1 << j)
        else:
            # Add i, Remove j
            new_S = (current_S | (1 << i)) & ~(1 << j)
            
        new_score = score_bitset(new_S)
        delta = new_score - current_score
        
        if delta > 0:
            current_S = new_S
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_S = current_S
        else:
            if delta == 0:
                prob = 1.0
            else:
                # Accept with prob exp(delta/T)
                # If T is very small, prob is ~0
                if T > 0.001:
                    prob = math.exp(delta / T)
                else:
                    prob = 0.0
                    
            if random.random() < prob:
                current_S = new_S
                current_score = new_score
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # Restart logic
        if no_improve_count > 20000:
            current_S = best_S
            current_score = best_score
            T = 5.0
            no_improve_count = 0

    # Convert best_S to list
    result = []
    for i in range(MAX_VAL):
        if (best_S >> i) & 1:
            result.append(i + 1)
            
    return result
