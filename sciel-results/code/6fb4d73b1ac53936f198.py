"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import bisect

def construct():
    # Compute L = lcm(1, 2, ..., 30)
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
    
    # w[i] = L / (i+1) for i in 0..29
    w = [L // (i + 1) for i in range(30)]
    
    # Split into two halves
    n1 = 15
    n2 = 15
    
    # Generate all (sum, state) for the second half (indices 15..29)
    # state uses 2 bits per index.
    # Global index j corresponds to bits 2*j and 2*j+1.
    # 00: 0, 01: +1, 10: -1
    
    list2 = []
    
    # Recursive generation for second half
    def gen2(idx, current_sum, current_state):
        if idx == 15:
            list2.append((current_sum, current_state))
            return
        w_idx = w[15 + idx]
        # 0
        gen2(idx + 1, current_sum, current_state)
        # +1
        gen2(idx + 1, current_sum + w_idx, current_state | (1 << (2 * (15 + idx))))
        # -1
        gen2(idx + 1, current_sum - w_idx, current_state | (2 << (2 * (15 + idx))))
        
    gen2(0, 0, 0)
    
    # Sort list2 by sum
    list2.sort(key=lambda x: x[0])
    
    # Extract sums for bisect
    sums2 = [s for s, _ in list2]
    
    best_abs_T = float('inf')
    best_state = 0
    
    # Search first half
    def search1(idx, current_sum, current_state):
        nonlocal best_abs_T, best_state
        if idx == 15:
            # Find s2 in sums2 closest to -current_sum
            target = -current_sum
            pos = bisect.bisect_left(sums2, target)
            
            # Check pos and pos-1
            candidates = []
            if pos < len(sums2):
                candidates.append(pos)
            if pos > 0:
                candidates.append(pos - 1)
            
            for p in candidates:
                s2, state2 = list2[p]
                total_sum = current_sum + s2
                if total_sum == 0:
                    continue
                abs_t = abs(total_sum)
                if abs_t < best_abs_T:
                    best_abs_T = abs_t
                    best_state = current_state | state2
                    if best_abs_T == 1:
                        # Can't get better than 1
                        return True
            return False
        else:
            w_idx = w[idx]
            # 0
            if search1(idx + 1, current_sum, current_state):
                return True
            # +1
            if search1(idx + 1, current_sum + w_idx, current_state | (1 << (2 * idx))):
                return True
            # -1
            if search1(idx + 1, current_sum - w_idx, current_state | (2 << (2 * idx))):
                return True
            return False

    search1(0, 0, 0)
    
    # Construct coefficients
    coeffs = [0] * 30
    for i in range(30):
        val = (best_state >> (2 * i)) & 3
        if val == 1:
            coeffs[i] = 1
        elif val == 2:
            coeffs[i] = -1
        else:
            coeffs[i] = 0
            
    return coeffs
# EVOLVE-BLOCK-END
