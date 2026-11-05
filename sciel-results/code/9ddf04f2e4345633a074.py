import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    # Pre-compute popcount table for 17-bit integers
    popcount = [0] * (1 << 17)
    for i in range(1, 1 << 17):
        popcount[i] = popcount[i >> 1] + (i & 1)
    popcount_np = np.array(popcount, dtype=np.int8)

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
                    if g[k] != 0:
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

    base_code = list(set(words_17_8_6))
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

    # Step 2: Generate Candidates (weights 6 and 7)
    candidates = []
    for i in range(17):
        for j in range(i+1, 17):
            for k in range(j+1, 17):
                for l in range(k+1, 17):
                    for m in range(l+1, 17):
                        for n in range(m+1, 17):
                            w = (1 << i) | (1 << j) | (1 << k) | (1 << l) | (1 << m) | (1 << n)
                            candidates.append(w)
    for i in range(17):
        for j in range(i+1, 17):
            for k in range(j+1, 17):
                for l in range(k+1, 17):
                    for m in range(l+1, 17):
                        for n in range(m+1, 17):
                            for o in range(n+1, 17):
                                w = (1 << i) | (1 << j) | (1 << k) | (1 << l) | (1 << m) | (1 << n) | (1 << o)
                                candidates.append(w)
    
    base_set = set(current_code)
    candidates = [c for c in candidates if c not in base_set]
    
    cands_np = np.array(candidates, dtype=np.int64)
    code_np = np.array(current_code, dtype=np.int64)
    
    # Active candidates: list of indices into cands_np
    active_indices = list(range(len(cands_np)))
    random.shuffle(active_indices)

    # Phase 1: Greedy Add - add any candidate that fits
    while time.time() < deadline - 15.0 and len(active_indices) > 0:
        # Try to find a candidate that fits
        found = False
        for idx in active_indices:
            w_new = int(cands_np[idx])
            if np.all(popcount_np[w_new ^ code_np] >= 6):
                current_code.append(w_new)
                code_np = np.append(code_np, w_new)
                active_indices.remove(idx)
                found = True
                if len(current_code) > best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                break
        
        if not found:
            # No single candidate fits; break to local search
            break

    # Phase 2: Local Search (Remove 1, Add 2)
    # Rebuild active_indices to exclude words in current_code
    code_set = set(current_code)
    active_indices = [i for i in range(len(cands_np)) if int(cands_np[i]) not in code_set]
    random.shuffle(active_indices)

    while time.time() < deadline - 3.0 and len(active_indices) > 0:
        n_code = len(current_code)
        if n_code <= 32:
            break

        # Pick a random word to remove
        idx_rm = random.randint(0, n_code - 1)
        w_old = current_code[idx_rm]
        current_code.pop(idx_rm)
        code_np = np.array(current_code, dtype=np.int64)

        # Vectorized check: which active candidates fit with the reduced code?
        if len(active_indices) == 0:
            current_code.append(w_old)
            code_np = np.append(code_np, w_old)
            break

        active_cands = cands_np[active_indices]
        # Shape: (N_active, N_code)
        dists = popcount_np[active_cands[:, None] ^ code_np[None, :]]
        min_dists = np.min(dists, axis=1)
        valid_rel = np.where(min_dists >= 6)[0]

        if len(valid_rel) >= 2:
            # Add 2 words: net gain of +1
            # Pick two that are mutually compatible
            # First, check mutual distance between the valid candidates
            valid_cands = active_cands[valid_rel]
            # We need to find a pair with distance >= 6
            found_pair = False
            for i in range(min(len(valid_rel), 50)):
                for j in range(i + 1, min(len(valid_rel), 50)):
                    if popcount_fast(int(valid_cands[i] ^ valid_cands[j])) >= 6:
                        w1 = int(valid_cands[i])
                        w2 = int(valid_cands[j])
                        current_code.append(w1)
                        current_code.append(w2)
                        # Remove from active_indices
                        glob1 = active_indices[valid_rel[i]]
                        glob2 = active_indices[valid_rel[j]]
                        active_indices.remove(glob1)
                        active_indices.remove(glob2)
                        code_np = np.array(current_code, dtype=np.int64)
                        if len(current_code) > best_size:
                            best_size = len(current_code)
                            best_code = list(current_code)
                        found_pair = True
                        break
                if found_pair:
                    break
            
            if not found_pair:
                # Can't add 2 mutually compatible; try adding just 1
                if len(valid_rel) >= 1:
                    w1 = int(valid_cands[valid_rel[0]])
                    current_code.append(w1)
                    glob1 = active_indices[valid_rel[0]]
                    active_indices.remove(glob1)
                    code_np = np.array(current_code, dtype=np.int64)
                else:
                    current_code.append(w_old)
                    code_np = np.append(code_np, w_old)

        elif len(valid_rel) == 1:
            # Swap: add 1, size unchanged but may enable future improvements
            w1 = int(active_cands[valid_rel[0]])
            current_code.append(w1)
            glob1 = active_indices[valid_rel[0]]
            active_indices.remove(glob1)
            code_np = np.array(current_code, dtype=np.int64)
        else:
            # Restore
            current_code.append(w_old)
            code_np = np.append(code_np, w_old)

    # Final Verification
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
