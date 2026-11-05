import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    N = 17
    MIN_DIST = 6
    FULL_MASK = (1 << N) - 1

    # Pre-compute popcount table for 17-bit integers
    popcount = np.array([0] * (1 << N), dtype=np.uint8)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def pc(x):
        return int(popcount[x & FULL_MASK])

    def is_valid(words):
        n = len(words)
        if n < 2:
            return True
        arr = np.array(words, dtype=np.uint32)
        # Check all pairs using broadcasting in chunks to save memory
        chunk = 512
        for i in range(0, n, chunk):
            a = arr[i:i+chunk]
            for j in range(i + chunk, n):
                b = arr[j]
                d = popcount[a ^ b]
                if np.min(d) < MIN_DIST:
                    return False
            # Also check within chunk
            if len(a) > 1:
                # upper triangle
                for ii in range(len(a)):
                    for jj in range(ii + 1, len(a)):
                        if pc(a[ii] ^ a[jj]) < MIN_DIST:
                            return False
        return True

    # Step 1: Build the [17,8,6] linear code (256 words) via QR code construction
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

    # Find element of order 17 in GF(2^8)
    alpha = None
    for candidate in range(1, 256):
        if gf_pow(candidate, 17) == 1 and candidate != 1:
            alpha = candidate
            break

    if alpha is None:
        alpha = 3

    # Quadratic residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]

    # Build generator polynomial g(x) of degree 8
    g = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            new_g[j] ^= gf_mul(g[j], root)
            new_g[j + 1] ^= g[j]
        g = new_g

    # Generate all 512 words of the [17,9,5] cyclic code
    words_17_9_5 = []
    for m in range(512):
        c = [0] * 17
        for j in range(9):
            if (m >> j) & 1:
                for k in range(9):
                    if g[k]:
                        pos = (j + k) % 17
                        c[pos] ^= 1
        val = 0
        for bit in c:
            val = (val << 1) | bit
        words_17_9_5.append(val)

    # Extend to [18,9,6] by appending overall parity
    words_18_9_6 = []
    for w in words_17_9_5:
        parity = pc(w) % 2
        ext_val = w | (parity << 17)
        words_18_9_6.append(ext_val)

    # Shorten: keep words with parity bit = 0
    words_17_8_6 = []
    for w in words_18_9_6:
        if (w >> 17) & 1 == 0:
            w_short = w & FULL_MASK
            words_17_8_6.append(w_short)

    words_17_8_6 = list(set(words_17_8_6))
    base_code = words_17_8_6

    # Verify and fix if needed
    if not is_valid(base_code):
        valid_code = [base_code[0]]
        for w in base_code[1:]:
            ok = all(pc(w ^ v) >= MIN_DIST for v in valid_code)
            if ok:
                valid_code.append(w)
        base_code = valid_code

    current_code = list(base_code)
    best_code = list(base_code)
    best_size = len(current_code)
    if best_size < 32:
        return [0]

    # Step 2: Greedy expansion with multiple restarts and different orderings
    # Key insight: try to add words in order of weight, and use multiple random seeds
    
    # Pre-compute all words by weight
    words_by_weight = {w: [] for w in range(N + 1)}
    for w in range(1 << N):
        words_by_weight[pc(w)].append(w)

    # For distance 6 code, words of weight 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
    # but weight 0 (all zeros) is in the linear code
    # Focus on weights 6-11 for additions (these are most likely to fit)

    def greedy_expand(code_list, candidate_order, time_limit):
        """Try to add candidates in the given order, greedily."""
        code = list(code_list)
        code_set = set(code)
        code_arr = np.array(code, dtype=np.uint32)
        
        for w in candidate_order:
            if time.time() > time_limit:
                break
            if w in code_set:
                continue
            # Check compatibility with all existing codewords
            if len(code_arr) == 0:
                code.append(w)
                code_set.add(w)
                code_arr = np.array(code, dtype=np.uint32)
                continue
            xored = code_arr ^ w
            dists = popcount[xored & FULL_MASK]
            if np.min(dists) >= MIN_DIST:
                code.append(w)
                code_set.add(w)
                code_arr = np.array(code, dtype=np.uint32)
        
        return code

    # Strategy: Multiple rounds of greedy expansion with different candidate orderings
    # Round 1: Try all words of weight 6-11 in random order
    # Round 2: Try remaining words in random order
    # Repeat with different seeds

    all_candidates = []
    for wt in [6, 7, 8, 9, 10, 11]:
        all_candidates.extend(words_by_weight[wt])
    
    # Also include some higher weight words
    for wt in [5, 12]:
        all_candidates.extend(words_by_weight[wt])

    random.shuffle(all_candidates)

    # Run several rounds of greedy expansion
    for round_num in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Each round: start from best, try to add in a shuffled order
        # Use different shuffling strategies
        if round_num % 3 == 0:
            # Weight-ordered: try low weights first
            cands = []
            for wt in [6, 7, 8, 9, 10, 11, 5, 12]:
                cands.extend(words_by_weight[wt])
            random.shuffle(cands)
        elif round_num % 3 == 1:
            # Random order
            cands = list(all_candidates)
            random.shuffle(cands)
        else:
            # Weight-ordered descending
            cands = []
            for wt in [12, 11, 10, 9, 8, 7, 6, 5]:
                cands.extend(words_by_weight[wt])
            random.shuffle(cands)

        result = greedy_expand(best_code, cands, deadline - 10.0)
        if len(result) > best_size:
            best_size = len(result)
            best_code = result
            # If we improved, try to continue expanding
            # But don't spend too much time on one round

    # Step 3: Local search - try to swap out a codeword for a better one
    # This is more expensive but can break through local optima
    
    ls_time_limit = deadline - 5.0
    current_code = list(best_code)
    
    # Build a set for fast lookup
    code_set = set(current_code)
    
    # Pre-compute: for each codeword, which candidates are incompatible?
    # Too expensive. Instead, just try random swaps.
    
    # Candidate pool for swaps: words of weight 6-12
    swap_candidates = []
    for wt in [5, 6, 7, 8, 9, 10, 11, 12]:
        swap_candidates.extend(words_by_weight[wt])
    random.shuffle(swap_candidates)
    
    swap_idx = 0
    while time.time() < ls_time_limit:
        if swap_idx >= len(swap_candidates):
            random.shuffle(swap_candidates)
            swap_idx = 0
        
        w_new = swap_candidates[swap_idx]
        swap_idx += 1
        
        if w_new in code_set:
            continue
        
        # Try to find a codeword to remove that allows adding w_new
        # Strategy: for each codeword c in current_code, check if removing c
        # allows adding w_new. If so, do the swap.
        # This is O(n) per candidate, which is expensive.
        # Better: only try a few random codewords to remove.
        
        if len(current_code) <= best_size:
            continue  # Don't shrink
        
        # Try removing a random codeword and see if w_new fits
        n_trials = min(5, len(current_code))
        for _ in range(n_trials):
            idx_rm = random.randint(0, len(current_code) - 1)
            w_rm = current_code[idx_rm]
            
            # Temporarily remove w_rm
            code_without_rm = [c for i, c in enumerate(current_code) if i != idx_rm]
            
            # Check if w_new is compatible with code_without_rm
            if len(code_without_rm) == 0:
                compatible = True
            else:
                arr = np.array(code_without_rm, dtype=np.uint32)
                dists = popcount[(arr ^ w_new) & FULL_MASK]
                compatible = np.min(dists) >= MIN_DIST
            
            if compatible:
                # Do the swap
                current_code[idx_rm] = w_new
                code_set.discard(w_rm)
                code_set.add(w_new)
                
                if len(current_code) > best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                break
            else:
                # Try next random removal candidate
                continue

    # Final: try one more greedy pass from best
    if time.time() < deadline - 2.0:
        cands = list(all_candidates)
        random.shuffle(cands)
        result = greedy_expand(best_code, cands, deadline - 2.0)
        if len(result) > best_size:
            best_size = len(result)
            best_code = result

    # Final verification
    if len(best_code) >= 32:
        if is_valid(best_code):
            return best_code
        else:
            # Filter to valid subset
            final = []
            for w in best_code:
                ok = all(pc(w ^ v) >= MIN_DIST for v in final)
                if ok:
                    final.append(w)
            if len(final) >= 32:
                return final
            return base_code
    else:
        return base_code if len(base_code) >= 32 else [0]
