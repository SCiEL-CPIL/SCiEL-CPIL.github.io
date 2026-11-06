import numpy as np
import time
import random
from itertools import combinations
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 17
    MIN_DIST = 6
    
    # GF(256) construction for the QR code.
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    
    def gf_mul(a, b, poly=0x11D):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n, poly=0x11D):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base, poly)
            base = gf_mul(base, base, poly)
            n >>= 1
        return res

    def get_primitive_17th_root():
        # 255 = 15 * 17
        # Primitive element alpha. Let's test 2.
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    current_code = None
    
    if beta:
        # Build the generator polynomial
        poly_coeffs = [1]
        current_beta = beta
        for _ in range(8):
            new_coeffs = [0] * (len(poly_coeffs) + 1)
            for i in range(len(poly_coeffs)):
                new_coeffs[i+1] = poly_coeffs[i]
            for i in range(len(poly_coeffs)):
                new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
            poly_coeffs = new_coeffs
            current_beta = gf_mul(current_beta, current_beta)
            
        g_bits = []
        for c in poly_coeffs:
            if c == 1:
                g_bits.append(1)
            elif c == 0:
                g_bits.append(0)
            else:
                g_bits.append(1)
        
        if len(g_bits) == 9 and g_bits[0] == 1:
            # Generate [17, 9] code
            codewords_17 = []
            for u in range(512):
                c = 0
                for k in range(9):
                    if (u >> k) & 1:
                        for j in range(9):
                            if g_bits[j]:
                                c ^= 1 << (j + k)
                codewords_17.append(c)
            
            # Extend to 18
            extended_codewords = []
            for c in codewords_17:
                parity = bin(c).count('1') % 2
                ec = c | (parity << 17)
                extended_codewords.append(ec)
            
            # Shorten (fix bit 17 to 0)
            shortened_codewords = []
            for ec in extended_codewords:
                if (ec >> 17) & 1 == 0:
                    shortened_codewords.append(ec & ((1 << 17) - 1))
            
            # Remove duplicates
            shortened_codewords = list(set(shortened_codewords))
            
            # Verify
            if len(shortened_codewords) >= 256:
                min_d = 100
                for i in range(len(shortened_codewords)):
                    for j in range(i+1, len(shortened_codewords)):
                        d = (shortened_codewords[i] ^ shortened_codewords[j]).bit_count()
                        if d < min_d:
                            min_d = d
                if min_d >= 6:
                    current_code = set(shortened_codewords)
    
    if current_code is None:
        # Fallback: Greedy from scratch
        current_code = set()
        current_code.add(0)
        
        random.seed(42)
        candidates = list(range(1, 1 << 17))
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline - 10.0:
                break
            valid = True
            for w in current_code:
                if (cand ^ w).bit_count() < 6:
                    valid = False
                    break
            if valid:
                current_code.add(cand)
    
    code_set = set(current_code)
    
    # Vectorized local search to add more codewords
    # Convert code_set to numpy array for fast distance computation
    code_array = np.array(list(code_set), dtype=np.uint32)
    code_bits = np.array([(c >> i) & 1 for c in code_array for i in range(17)], dtype=np.uint8).reshape(-1, 17)
    
    # Generate candidate pool: all words of weight 3-8 (likely where holes are)
    candidate_pool = []
    for w in range(3, 9):
        for comb in combinations(range(17), w):
            word = 0
            for i in comb:
                word |= (1 << i)
            candidate_pool.append(word)
    
    # Also add some random candidates
    random.seed(42)
    for _ in range(5000):
        candidate_pool.append(random.getrandbits(17))
    
    candidate_array = np.array(candidate_pool, dtype=np.uint32)
    candidate_bits = np.array([(c >> i) & 1 for c in candidate_array for i in range(17)], dtype=np.uint8).reshape(-1, 17)
    
    # Find candidates that are at distance >= 6 from ALL current codewords
    # Using vectorized XOR and bit count
    # Process in batches to manage memory
    batch_size = 10000
    valid_indices = []
    
    for start in range(0, len(candidate_array), batch_size):
        end = min(start + batch_size, len(candidate_array))
        batch_cands = candidate_bits[start:end]
        
        # Compute distances: batch_cands (batch, 17) vs code_bits (n_code, 17)
        # Use broadcasting: (batch, 1, 17) xor (1, n_code, 17) -> (batch, n_code, 17)
        # This might be too large, so use chunked approach
        n_code = len(code_bits)
        chunk_size = max(1, 5000000 // n_code)  # Keep intermediate array under ~5M elements
        
        for cstart in range(0, len(batch_cands), chunk_size):
            if time.time() > deadline - 10.0:
                break
            cend = min(cstart + chunk_size, len(batch_cands))
            chunk = batch_cands[cstart:cend]
            
            # Compute min distance from each candidate in chunk to any codeword
            # chunk: (c, 17), code_bits: (n, 17)
            # dist: (c, n)
            xor = chunk[:, np.newaxis, :] ^ code_bits[np.newaxis, :, :]
            dists = np.sum(xor, axis=2)
            min_dists = np.min(dists, axis=1)
            
            valid_in_chunk = np.where(min_dists >= 6)[0]
            for idx in valid_in_chunk:
                valid_indices.append(start + cstart + idx)
    
    if time.time() > deadline - 5.0:
        return list(code_set)
    
    # Now we have valid_indices: candidates at distance >= 6 from all current codewords
    # We need to select a maximum subset of these that are also mutually at distance >= 6
    # This is maximum independent set on the conflict graph
    
    # Greedy with multiple restarts using vectorized checks
    best_size = len(code_set)
    best_set = set(code_set)
    best_array = code_array.copy()
    best_bits = code_bits.copy()
    
    valid_cands = candidate_array[valid_indices]
    valid_cands_bits = candidate_bits[valid_indices]
    
    random.seed(42)
    n_valid = len(valid_cands)
    
    # Try multiple random orderings
    for trial in range(200):
        if time.time() > deadline - 3.0:
            break
        
        current_size = len(best_set)
        current_array = best_array.copy()
        current_bits = best_bits.copy()
        
        # Shuffle valid candidates
        order = np.random.permutation(n_valid)
        
        for idx in order:
            if time.time() > deadline - 3.0:
                break
            
            cand = valid_cands[idx]
            cand_bit = valid_cands_bits[idx]
            
            # Check if cand is already in current set
            # Quick check: compute min distance to current set
            if len(current_bits) > 0:
                xor = current_bits ^ cand_bit
                dists = np.sum(xor, axis=1)
                if np.min(dists) < 6:
                    continue
            
            # Add it
            current_array = np.append(current_array, cand)
            current_bits = np.vstack([current_bits, cand_bit])
            current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(best_array.tolist())
            best_array = current_array
            best_bits = current_bits
    
    # Final validation
    result = list(best_set)
    
    # Verify all pairs
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if (result[i] ^ result[j]).bit_count() < 6:
                # Invalid, fall back to original
                return list(code_set)
    
    return result
