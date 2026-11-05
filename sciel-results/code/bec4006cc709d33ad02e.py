import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Pre-compute popcount table for 17-bit integers
    popcount = np.zeros(1 << 17, dtype=np.uint8)
    for i in range(1, 1 << 17):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def popcount_fast(x):
        return popcount[x & 0x1FFFF]

    def vpopcount(x):
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        return ((x + (x >> 4)) & 0x0F0F0F0F) % 255

    def check_compatibility_vec(code_arr, candidate):
        xored = code_arr ^ candidate
        dists = vpopcount(xored)
        return np.min(dists) >= 6

    # Step 1: Build the [17,8,6] linear code
    # Use the known generator polynomial for the [17,9,5] QR code
    # g(x) = x^8 + x^5 + x^3 + x^2 + x + 1
    # This gives a [17, 9, 5] cyclic code.
    # Extend to [18,9,6] by parity, then shorten to [17,8,6].
    
    g_poly = 0b100101101  # x^8 + x^5 + x^3 + x^2 + x + 1
    # g coefficients (bit 0 = constant term): 
    # g(x) = x^8 + x^5 + x^3 + x^2 + x + 1
    # bits: pos 8,5,3,2,1,0 set
    g_coeffs = [1, 1, 1, 1, 0, 1, 0, 0, 1]  # g[0]..g[8]
    
    # Generate all 512 words of the [17,9,5] cyclic code
    words_17_9_5 = set()
    for m in range(512):
        c = [0] * 17
        for j in range(9):
            if (m >> j) & 1:
                for k in range(9):
                    if g_coeffs[k]:
                        pos = (j + k) % 17
                        c[pos] ^= 1
        val = 0
        for bit in c:
            val = (val << 1) | bit
        words_17_9_5.add(val)
    
    # Extend to [18,9,6] by appending overall parity
    words_18_9_6 = []
    for w in words_17_9_5:
        parity = popcount_fast(w) % 2
        ext_val = w | (parity << 17)
        words_18_9_6.append(ext_val)
    
    # Shorten: keep words with parity bit = 0
    words_17_8_6 = []
    for w in words_18_9_6:
        if (w >> 17) & 1 == 0:
            w_short = w & 0x1FFFF
            words_17_8_6.append(w_short)
    
    # Deduplicate
    words_17_8_6 = list(set(words_17_8_6))
    base_code = words_17_8_6
    
    # Verify base code is valid
    def is_valid_code(words):
        n = len(words)
        for i in range(n):
            wi = words[i]
            for j in range(i + 1, n):
                if popcount_fast(wi ^ words[j]) < 6:
                    return False
        return True
    
    if not is_valid_code(base_code):
        # Fallback: greedily build a valid code
        valid_code = []
        for w in base_code:
            ok = True
            for v in valid_code:
                if popcount_fast(w ^ v) < 6:
                    ok = False
                    break
            if ok:
                valid_code.append(w)
        base_code = valid_code
    
    if len(base_code) < 32:
        return [0]
    
    current_code = list(base_code)
    best_code = list(base_code)
    best_size = len(current_code)
    code_set = set(current_code)
    
    # Step 2: Build targeted candidate pool
    # Focus on words at distance 6 from zero (weight 6) and nearby weights
    # C(17,6) = 12376, C(17,7) = 19448, C(17,8) = 24310
    # Total ~56k candidates, manageable
    
    candidates = []
    for w in range(1 << 17):
        wt = popcount[w]
        if wt == 6 or wt == 7:
            candidates.append(w)
    
    random.shuffle(candidates)
    
    # Step 3: Greedy add pass - try to add all compatible candidates
    code_arr = np.array(current_code, dtype=np.uint32)
    
    greedy_end = min(deadline - 10.0, start_time + 15.0)
    i = 0
    while i < len(candidates) and time.time() < greedy_end:
        w_new = candidates[i]
        if w_new not in code_set:
            if check_compatibility_vec(code_arr, w_new):
                current_code.append(w_new)
                code_set.add(w_new)
                code_arr = np.append(code_arr, w_new)
                if len(current_code) > best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
        i += 1
    
    # Step 4: Local search with swaps
    ls_end = deadline - 3.0
    
    while time.time() < ls_end:
        # Try random swaps: remove one codeword, try to add a better one
        if len(current_code) <= 30:
            break
        
        idx_rm = random.randint(0, len(current_code) - 1)
        w_rm = current_code[idx_rm]
        
        # Try to find a replacement
        found = False
        for _ in range(20):
            idx_new = random.randint(0, len(candidates) - 1)
            w_new = candidates[idx_new]
            if w_new in code_set:
                continue
            
            # Test compatibility without w_rm
            # Create a view excluding w_rm
            if idx_rm == len(current_code) - 1:
                code_arr_tmp = code_arr[:-1]
            else:
                code_arr_tmp = np.concatenate([code_arr[:idx_rm], code_arr[idx_rm+1:]])
            
            if check_compatibility_vec(code_arr_tmp, w_new):
                # Also need w_new compatible with w_rm's "slot" - actually we just replace
                current_code[idx_rm] = w_new
                code_set.remove(w_rm)
                code_set.add(w_new)
                code_arr = np.array(current_code, dtype=np.uint32)
                found = True
                break
        
        if not found and random.random() < 0.3:
            # Occasionally try to add without removing
            idx_new = random.randint(0, len(candidates) - 1)
            w_new = candidates[idx_new]
            if w_new not in code_set:
                if check_compatibility_vec(code_arr, w_new):
                    current_code.append(w_new)
                    code_set.add(w_new)
                    code_arr = np.append(code_arr, w_new)
                    if len(current_code) > best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
    
    # Final verification
    if len(best_code) >= 32 and is_valid_code(best_code):
        return best_code
    else:
        # Filter best_code to ensure validity
        final = []
        for w in best_code:
            ok = True
            for v in final:
                if popcount_fast(w ^ v) < 6:
                    ok = False
                    break
            if ok:
                final.append(w)
        if len(final) >= 32:
            return final
        return base_code
