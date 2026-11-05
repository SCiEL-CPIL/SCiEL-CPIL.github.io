import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Pre-compute popcount table for 17-bit integers
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
    gf_poly = 0x1d  # x^8+x^4+x^3+x^2+1, primitive in GF(2^8)

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

    # Find element of order 17
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

    # Generate all 512 words of the [17,9,5] code
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

    # Step 2: Vectorized Greedy Extension
    # Precompute byte popcount table for vectorized operations
    byte_popcount = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    def vectorized_popcount(arr):
        """Compute popcount for an array of uint32 integers using byte lookup."""
        low = arr & 0xFF
        high = (arr >> 8) & 0xFF
        return byte_popcount[low] + byte_popcount[high]

    code_set = set(current_code)
    code_arr = np.array(current_code, dtype=np.uint32)
    
    # Generate all possible 17-bit words
    all_words = np.arange(1 << 17, dtype=np.uint32)
    np.random.shuffle(all_words)
    
    # We will iterate through all_words and try to add compatible ones
    # To avoid O(N^2) complexity in the worst case, we process in batches
    # and update the code array as we go.
    
    batch_size = 2048
    num_batches = len(all_words) // batch_size + 1
    
    for b in range(num_batches):
        if time.time() > deadline - 5.0:
            break
        
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, len(all_words))
        if start_idx >= len(all_words):
            break
            
        batch = all_words[start_idx:end_idx]
        
        # Filter out words already in the code
        # This is a bit slow in Python if done naively, but batch size is small
        # and code_set is a set.
        # We can vectorize the "not in set" check? No, sets are not vectorized.
        # But we can just check compatibility and then filter duplicates at the end?
        # No, if a word is in the code, it's compatible (dist 0 to itself, but we check pairwise).
        # Actually, if w is in code, d(w, w) = 0 < 6. So it will be rejected by the check.
        # So we don't strictly need to filter pre-check, but it saves computation.
        
        # Check compatibility with current code
        # xor_matrix shape: (batch_size, code_size)
        # This can be large: 2048 * 300 = 614k elements. Fine.
        
        # If code is very large, this might be slow. But code size is expected to grow to ~300-400.
        
        xor_matrix = batch[:, np.newaxis] ^ code_arr[np.newaxis, :]
        pcs = vectorized_popcount(xor_matrix)
        
        # A candidate is valid if min popcount in its row >= 6
        # Note: If the candidate is in the code, the popcount against itself is 0, so min will be 0.
        # So this naturally handles duplicates.
        
        min_pcs = np.min(pcs, axis=1)
        valid_mask = min_pcs >= 6
        
        if np.any(valid_mask):
            valid_words = batch[valid_mask]
            for w in valid_words:
                w_int = int(w)
                if w_int not in code_set:
                    code_set.add(w_int)
                    current_code.append(w_int)
            
            if len(current_code) > best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            
            # Update code array for next batch
            code_arr = np.array(current_code, dtype=np.uint32)
    
    # Step 3: Local Search / Random Polish
    # If we have time left, try random swaps or additions
    remaining_time = deadline - 5.0 - time.time()
    if remaining_time > 1.0 and len(current_code) < 1000: # Safety cap
        # Try adding more random words
        for _ in range(1000):
            if time.time() > deadline - 5.0:
                break
            w = random.randint(0, (1 << 17) - 1)
            if w in code_set:
                continue
            # Check compatibility
            ok = True
            for c in current_code:
                if popcount_fast(w ^ c) < 6:
                    ok = False
                    break
            if ok:
                code_set.add(w)
                current_code.append(w)
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
