"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
from fractions import Fraction
from math import lcm, exp

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85
    N = 30
    
    # Compute LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
        
    # Weights w_i = L / i
    weights = [L // i for i in range(1, N + 1)]
    
    best_abs = float('inf')
    best_coeffs = [0] * N
    
    def update_best(coeffs, s):
        nonlocal best_abs, best_coeffs
        if s != 0 and abs(s) < best_abs:
            best_abs = abs(s)
            best_coeffs = coeffs[:]

    # Initialize with a few random seeds
    for _ in range(1000):
        c = [random.randint(-1, 1) for _ in range(N)]
        s = sum(c[i] * weights[i] for i in range(N))
        if s != 0:
            update_best(c, s)
            
    # Ensure we have a valid start
    if best_abs == float('inf'):
        c = [0]*N
        c[0] = 1
        s = weights[0]
        update_best(c, s)

    current_coeffs = best_coeffs[:]
    current_sum = sum(current_coeffs[i] * weights[i] for i in range(N))
    
    T = 1000.0
    T_min = 1e-9
    cooling = 0.999
    
    while time.time() < deadline:
        # Move generation
        if random.random() < 0.5:
            # 1-bit move
            i = random.randint(0, N - 1)
            old_val = current_coeffs[i]
            new_val = random.randint(-1, 1)
            if new_val == old_val:
                continue
            delta = (new_val - old_val) * weights[i]
            new_sum = current_sum + delta
            move_type = 1
            move_data = (i, new_val)
        else:
            # 2-bit move
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            while j == i:
                j = random.randint(0, N - 1)
            old_i, old_j = current_coeffs[i], current_coeffs[j]
            new_i, new_j = random.randint(-1, 1), random.randint(-1, 1)
            if (new_i == old_i and new_j == old_j):
                continue
            delta = (new_i - old_i) * weights[i] + (new_j - old_j) * weights[j]
            new_sum = current_sum + delta
            move_type = 2
            move_data = (i, new_i, j, new_j)
            
        if new_sum == 0:
            # Explore neighbors of 0
            temp_coeffs = current_coeffs[:]
            if move_type == 1:
                temp_coeffs[move_data[0]] = move_data[1]
            else:
                temp_coeffs[move_data[0]] = move_data[1]
                temp_coeffs[move_data[2]] = move_data[3]
                
            improved = False
            for k in range(N):
                old_k = temp_coeffs[k]
                for new_k in [-1, 0, 1]:
                    if new_k == old_k:
                        continue
                    flip_delta = (new_k - old_k) * weights[k]
                    flip_sum = flip_delta
                    if flip_sum != 0 and abs(flip_sum) < best_abs:
                        temp_coeffs[k] = new_k
                        update_best(temp_coeffs, flip_sum)
                        current_coeffs = temp_coeffs[:]
                        current_sum = flip_sum
                        improved = True
                        break
                if improved:
                    break
            continue
            
        # Acceptance
        if abs(new_sum) < abs(current_sum):
            accept = True
        else:
            delta_E = abs(new_sum) - abs(current_sum)
            if delta_E < 0:
                accept = True
            else:
                if T > 0:
                    accept = random.random() < exp(-delta_E / T)
                else:
                    accept = False
        
        if accept:
            if move_type == 1:
                current_coeffs[move_data[0]] = move_data[1]
            else:
                current_coeffs[move_data[0]] = move_data[1]
                current_coeffs[move_data[2]] = move_data[3]
            current_sum = new_sum
            update_best(current_coeffs, current_sum)
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_coeffs
# EVOLVE-BLOCK-END
