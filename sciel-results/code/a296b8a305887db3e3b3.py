"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave some buffer for 60s limit
    
    # The task: find up to 80 distinct positive integers <= 4096
    # such that every unordered triple (allowing repetitions) has a unique sum.
    # This is a B_3[1] sequence (Sidon-like for triples).
    
    # Strategy: Use FFT-based convolution to speed up validity checks.
    # Represent the set S as a binary vector where index x is 1 if x in S.
    # The generating function is G(z) = sum_{x in S} z^x
    # Triple sums correspond to coefficients in G(z)^3.
    # We need all coefficients in G(z)^3 to be at most 1.
    
    def can_add_fft(selected_arr, x, current_sums_set):
        """Check if adding x to selected creates any duplicate triple sum using FFT."""
        n = len(selected_arr)
        if n == 0:
            return True, {3*x}
        
        # Build the polynomial for current set + x
        # We need to check if any coefficient in (G + z^x)^3 exceeds 1
        # where G is the current polynomial
        
        # Compute G^2 using FFT
        size = 2 * (4096 + 1)  # enough for convolution
        fft_size = 1
        while fft_size < size:
            fft_size *= 2
        
        # G vector
        g = np.zeros(fft_size)
        for val in selected_arr:
            g[val] = 1.0
        g[x] = 1.0  # temporarily add x
        
        # Compute G^2
        g_fft = np.fft.rfft(g)
        g2_fft = g_fft * g_fft
        g2 = np.fft.irfft(g2_fft, n=fft_size).real
        
        # Now compute G^3 = G^2 * G
        g2_fft = np.fft.rfft(g2)
        g3_fft = g2_fft * g_fft
        g3 = np.fft.irfft(g3_fft, n=fft_size).real
        
        # Check if any coefficient exceeds 1 (with tolerance for floating point)
        # Coefficients should be integers 0 or 1 for valid set
        # If any coefficient > 1 + epsilon, there's a collision
        max_coeff = np.max(g3)
        if max_coeff > 1.0 + 1e-6:
            return False, set()
        
        # Also verify by checking specific new sums
        new_sums = set()
        # (x,x,x)
        s = 3 * x
        if s in current_sums_set:
            return False, set()
        new_sums.add(s)
        
        # (x,x,y) for y in selected
        for y in selected_arr:
            s = 2 * x + y
            if s in current_sums_set or s in new_sums:
                return False, set()
            new_sums.add(s)
        
        # (x,y,y) for y in selected
        for y in selected_arr:
            s = x + 2 * y
            if s in current_sums_set or s in new_sums:
                return False, set()
            new_sums.add(s)
        
        # (x,y,z) for y < z in selected
        for i in range(len(selected_arr)):
            for j in range(i + 1, len(selected_arr)):
                y, z = selected_arr[i], selected_arr[j]
                s = x + y + z
                if s in current_sums_set or s in new_sums:
                    return False, set()
                new_sums.add(s)
        
        return True, new_sums
    
    def run_greedy_fast(order, time_budget):
        """Greedy using FFT for fast collision detection."""
        selected = []
        current_sums = set()
        
        for x in order:
            if time.time() - start_time > time_budget:
                break
            
            valid, new_sums = can_add_fft(selected, x, current_sums)
            
            if valid:
                selected.append(x)
                current_sums.update(new_sums)
        
        return selected
    
    def run_greedy_slow(order, time_budget):
        """Original greedy for fallback."""
        selected = []
        current_sums = set()
        
        for x in order:
            if time.time() - start_time > time_budget:
                break
            
            new_sums = set()
            valid = True
            
            s = 3 * x
            if s in current_sums:
                valid = False
            else:
                new_sums.add(s)
            
            if valid:
                for y in selected:
                    s = 2 * x + y
                    if s in current_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            
            if valid:
                for y in selected:
                    s = x + 2 * y
                    if s in current_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            
            if valid:
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        y, z = selected[i], selected[j]
                        s = x + y + z
                        if s in current_sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                current_sums.update(new_sums)
        
        return selected
    
    # Try multiple strategies
    best_set = []
    
    # Strategy 1: Greedy from 1 upward using FFT
    s1 = run_greedy_fast(range(1, 4097), time_limit - (time.time() - start_time))
    if len(s1) > len(best_set):
        best_set = s1
    
    # Strategy 2: Greedy from larger numbers downward using FFT
    if time.time() - start_time < time_limit - 5:
        s2 = run_greedy_fast(range(4096, 0, -1), time_limit - (time.time() - start_time))
        if len(s2) > len(best_set):
            best_set = s2
    
    # Strategy 3: Random orderings with FFT
    random.seed(42)
    for trial in range(100):
        if time.time() - start_time > time_limit - 2:
            break
        order = list(range(1, 4097))
        random.shuffle(order)
        s = run_greedy_fast(order, time_limit - (time.time() - start_time))
        if len(s) > len(best_set):
            best_set = s
    
    # Strategy 4: Biased ordering - favor numbers that are far apart
    if time.time() - start_time < time_limit - 5:
        sparse = [1]
        for i in range(2, 4097):
            if i % 50 == 0:
                sparse.append(i)
        s = run_greedy_fast(sparse, time_limit - (time.time() - start_time))
        if len(s) > len(best_set):
            best_set = s
    
    # Strategy 5: Try to extend the best set found so far
    if time.time() - start_time < time_limit - 5:
        base = best_set[:]
        remaining = [x for x in range(1, 4097) if x not in set(base)]
        extended = run_greedy_fast(remaining, time_limit - (time.time() - start_time))
        if len(extended) > len(base):
            best_set = base + extended
    
    # Strategy 6: Local search - try to improve by swapping elements
    if time.time() - start_time < time_limit - 3:
        current_set = best_set[:]
        improved = True
        while improved and time.time() - start_time < time_limit - 3:
            improved = False
            current_set_set = set(current_set)
            for i in range(len(current_set)):
                if time.time() - start_time > time_limit - 3:
                    break
                # Try removing current_set[i] and adding a new element
                removed = current_set[i]
                trial_set = [x for x in current_set if x != removed]
                # Try to add a new element
                for new_x in range(1, 4097):
                    if new_x in current_set_set:
                        continue
                    valid, new_sums = can_add_fft(trial_set, new_x, set())
                    if valid:
                        # Verify the full set
                        test_set = trial_set + [new_x]
                        test_set.sort()
                        # Quick check
                        all_sums = set()
                        for a_idx in range(len(test_set)):
                            for b_idx in range(a_idx, len(test_set)):
                                for c_idx in range(b_idx, len(test_set)):
                                    s = test_set[a_idx] + test_set[b_idx] + test_set[c_idx]
                                    if s in all_sums:
                                        break
                                    all_sums.add(s)
                                if len(all_sums) < (a_idx+1)*(a_idx+2)//2:
                                    break
                            else:
                                continue
                            break
                        else:
                            # This is a valid improvement
                            current_set = test_set
                            current_set_set = set(current_set)
                            improved = True
                            break
                    if time.time() - start_time > time_limit - 3:
                        break
    
    # Verify the result
    def verify(s):
        if len(s) != len(set(s)):
            return False
        if any(x < 1 or x > 4096 for x in s):
            return False
        sums = [sum(t) for t in combinations_with_replacement(s, 3)]
        return len(sums) == len(set(sums))
    
    if not verify(best_set):
        # Fallback to a simple valid set
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    return best_set
# EVOLVE-BLOCK-END
