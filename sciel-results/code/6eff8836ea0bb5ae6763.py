import math
import bisect

def construct():
        # Compute L = lcm(1, 2, ..., 30)
        L = 1
        for i in range(1, 31):
            L = L * i // math.gcd(L, i)
        
        # w[i] = L / (i+1) for i in 0..29
        w = [L // (i + 1) for i in range(30)]
        
        n = 30
        half = 15
        
        # Generate sums for second half (indices 15..29)
        # We need to store (sum, state)
        # State bits: 2 bits per index. 00=0, 01=1, 10=-1
        # We can generate iteratively.
        
        sums2 = []
        states2 = []
        
        # Start with 0
        current_sums = [0]
        current_states = [0]
        
        for idx in range(half, n):
            new_sums = []
            new_states = []
            w_idx = w[idx]
            bit_pos = 2 * idx
            for s, st in zip(current_sums, current_states):
                # 0
                new_sums.append(s)
                new_states.append(st)
                # +1
                new_sums.append(s + w_idx)
                new_states.append(st | (1 << bit_pos))
                # -1
                new_sums.append(s - w_idx)
                new_states.append(st | (2 << bit_pos))
            current_sums = new_sums
            current_states = new_states
            
        sums2 = current_sums
        states2 = current_states
        
        # Sort by sum
        # We need to sort sums and keep states aligned
        # Create list of indices or tuples
        # Tuples are easier
        pairs = list(zip(sums2, states2))
        pairs.sort(key=lambda x: x[0])
        
        sorted_sums = [p[0] for p in pairs]
        sorted_states = [p[1] for p in pairs]
        
        best_abs_T = float('inf')
        best_state = 0
        
        # Search first half (indices 0..14)
        # Generate all sums/states for first half and check against sorted_sums
        # Instead of recursion, we can iterate through all 3^15 states of first half
        
        current_sums1 = [0]
        current_states1 = [0]
        
        for idx in range(half):
            new_sums1 = []
            new_states1 = []
            w_idx = w[idx]
            bit_pos = 2 * idx
            for s, st in zip(current_sums1, current_states1):
                # 0
                new_sums1.append(s)
                new_states1.append(st)
                # +1
                new_sums1.append(s + w_idx)
                new_states1.append(st | (1 << bit_pos))
                # -1
                new_sums1.append(s - w_idx)
                new_states1.append(st | (2 << bit_pos))
            current_sums1 = new_sums1
            current_states1 = new_states1
            
        # Now check all combinations
        # For each s1 in current_sums1, find best s2 in sorted_sums
        # Target is -s1
        # We want s1 + s2 to be close to 0 but not 0.
        
        for s1, st1 in zip(current_sums1, current_states1):
            target = -s1
            pos = bisect.bisect_left(sorted_sums, target)
            
            # Check pos and pos-1
            if pos < len(sorted_sums):
                s2 = sorted_sums[pos]
                total = s1 + s2
                if total != 0:
                    abs_t = abs(total)
                    if abs_t < best_abs_T:
                        best_abs_T = abs_t
                        best_state = st1 | sorted_states[pos]
                        if best_abs_T == 1:
                            break # Can't get better than 1
            
            if pos > 0:
                s2 = sorted_sums[pos - 1]
                total = s1 + s2
                if total != 0:
                    abs_t = abs(total)
                    if abs_t < best_abs_T:
                        best_abs_T = abs_t
                        best_state = st1 | sorted_states[pos - 1]
                        if best_abs_T == 1:
                            break
                            
        # If we broke out of loop early, we might have missed other s1s?
        # No, if we found 1, it's the global minimum possible.
        # But the `break` only exits the inner `for` loop? No, it exits the `for s1...` loop.
        # This is correct.
        
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
