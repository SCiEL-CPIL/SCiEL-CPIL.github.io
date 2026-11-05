import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Pre-compute popcount table for 17-bit integers (2^17 = 131072 entries)
    popcount = [0] * (1 << 17)
    for i in range(1, 1 << 17):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def popcount_fast(x):
        return popcount[x & 0x1FFFF]

    def is_valid_code(words):
        for i in range(len(words)):
            wi = words[i]
            for j in range(i + 1, len(words)):
                if popcount_fast(wi ^ words[j]) < 6:
                    return False
        return True

    # Step 1: Build the [17,8,6] linear code (256 words)
    gf_poly = 0x1d

    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = ((a << 1) ^ gf_poly) & 0xFF
            else:
                a = (a << 1) & 0xFF
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    alpha = None
    for candidate in range(1, 256):
        if gf_pow(candidate, 17) == 1 and candidate != 1:
            alpha = candidate
            break

    if alpha is None:
        alpha = 3

    Q = [1, 2, 4, 8, 9, 13, 15, 16]

    g = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            new_g[j] ^= gf_mul(g[j], root)
            new_g[j + 1] ^= g[j]
        g = new_g

    words_17_9_5 = []
    for m in range(512):
        c = [0] * 17
        for j in range(9):
            if (m >> j) & 1:
                for k in range(9):
                    if (g[k] != 0):
                        pos = (j + k) % 17
                        c[pos] ^= 1
        val = 0
        for bit in c:
            val = (val << 1) | bit
        words_17_9_5.append(val)

    words_18_9_6 = []
    for w in words_17_9_5:
        parity = popcount_fast(w) % 2
        ext_val = w | (parity << 17)
        words_18_9_6.append(ext_val)

    words_17_8_6 = []
    for w in words_18_9_6:
        if (w >> 17) & 1 == 0:
            w_short = w & 0x1FFFF
            words_17_8_6.append(w_short)

    words_17_8_6 = list(set(words_17_8_6))
    base_code = words_17_8_6

    if not is_valid_code(base_code):
        valid_code = [base_code[0]]
        for w in base_code[1:]:
            ok = True
            for v in valid_code:
                if popcount_fast(w ^ v) < 6:
                    ok = False
                    break
            if ok:
                valid_code.append(w)
        base_code = valid_code

    current_code = list(base_code)
    best_code = list(base_code)
    best_size = len(current_code)
    if best_size < 32:
        return [0]

    # Vectorized extension search
    popcount_np = np.array(popcount, dtype=np.int8)

    # Generate candidates: all weight-6 words (C(17,6) = 12376)
    candidates = []
    for combo in combinations(range(17), 6):
        w = 0
        for bit in combo:
            w |= (1 << bit)
        candidates.append(w)
    candidates = np.array(candidates, dtype=np.int64)
    
    # Also add weight-7 words as secondary candidates if time permits later
    # For now, focus on weight 6 which is the tightest constraint boundary
    
    code_np = np.array(current_code, dtype=np.int64)
    
    # Active candidates: start with all weight-6 words
    active_indices = np.arange(len(candidates))
    np.random.shuffle(active_indices)
    
    # Greedy vectorized addition loop
    # We process candidates one by one, but filter the remaining pool vectorized
    while time.time() < deadline - 5.0 and len(active_indices) > 0:
        idx = active_indices[0]
        w_new = candidates[idx]
        
        # Check validity against current code
        dists = popcount_np[w_new ^ code_np]
        
        if np.min(dists) >= 6:
            # Valid
            current_code.append(int(w_new))
            code_np = np.append(code_np, w_new)
            
            if len(current_code) > best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            
            active_indices = active_indices[1:]
            
            # Filter remaining active candidates
            if len(active_indices) > 0:
                active_cands = candidates[active_indices]
                dists_to_new = popcount_np[w_new ^ active_cands]
                mask = dists_to_new >= 6
                active_indices = active_indices[mask]
        else:
            active_indices = active_indices[1:]

    # If we still have time, try to add weight-7 or higher words via random sampling
    # This can sometimes fill gaps that weight-6 words couldn't
    remaining_time = deadline - 5.0 - time.time()
    if remaining_time > 2.0:
        # Try random words of weight 7 to 10
        # Generate a pool of random candidates
        random_candidates = []
        for _ in range(50000):
            if random.random() < 0.5:
                # Weight 7
                bits = random.sample(range(17), 7)
                w = 0
                for b in bits:
                    w |= (1 << b)
                random_candidates.append(w)
            else:
                # Weight 8
                bits = random.sample(range(17), 8)
                w = 0
                for b in bits:
                    w |= (1 << b)
                random_candidates.append(w)
        
        random_candidates = np.array(random_candidates, dtype=np.int64)
        # Remove duplicates and those already in code
        code_set = set(current_code)
        unique_random = []
        seen = set()
        for w in random_candidates:
            wi = int(w)
            if wi not in code_set and wi not in seen:
                seen.add(wi)
                unique_random.append(wi)
        
        if unique_random:
            unique_random = np.array(unique_random, dtype=np.int64)
            np.random.shuffle(unique_random)
            
            # Greedy add from this pool
            for w_new in unique_random:
                if time.time() > deadline - 2.0:
                    break
                dists = popcount_np[w_new ^ code_np]
                if np.min(dists) >= 6:
                    current_code.append(int(w_new))
                    code_np = np.append(code_np, w_new)
                    if len(current_code) > best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)

    # Final verification
    if len(best_code) >= 32:
        if is_valid_code(best_code):
            return best_code
        else:
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
    else:
        return base_code if len(base_code) >= 32 else [0]
