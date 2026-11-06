import numpy as np
import time
import random
from itertools import combinations

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
    
    # Now try to improve via local search: swap-based optimization
    # We try to find codewords that can be swapped in to increase the total count
    # or maintain count while improving structure for future additions
    
    random.seed(42)
    code_list = list(current_code)
    code_set = set(code_list)
    
    # Precompute distances to speed up validation
    # Use numpy for batch distance computation
    code_array = np.array(code_list, dtype=np.uint32)
    
    def compute_distances_batch(candidates, existing):
        """Compute Hamming distances between candidates and existing codewords.
        Returns min distance for each candidate."""
        # candidates: list of ints, existing: np.array of uint32
        # For each candidate, compute distance to all existing
        min_dists = np.full(len(candidates), 100, dtype=np.uint8)
        for i, cand in enumerate(candidates):
            # Compute XOR with all existing
            xor_vals = np.array([cand ^ e for e in existing], dtype=np.uint32)
            # Count bits
            d = np.array([v.bit_count() for v in xor_vals], dtype=np.uint8)
            min_dists[i] = d.min()
        return min_dists
    
    # Try to add random candidates first
    added_count = 0
    max_add_attempts = 10000
    add_attempts = 0
    while time.time() < deadline - 30.0 and add_attempts < max_add_attempts:
        cand = random.randint(0, (1 << 17) - 1)
        if cand in code_set:
            add_attempts += 1
            continue
        
        # Quick check: distance to all existing
        valid = True
        for w in code_set:
            if (cand ^ w).bit_count() < 6:
                valid = False
                break
        if valid:
            code_set.add(cand)
            added_count += 1
        add_attempts += 1
    
    # Now try swap-based local search
    # For each codeword, try swapping it with a random candidate
    # Accept if the new set is still valid (all pairs have distance >= 6)
    
    swap_improvements = 0
    max_swap_rounds = 50
    for round_num in range(max_swap_rounds):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        # Shuffle codewords for randomization
        code_list = list(code_set)
        random.shuffle(code_list)
        
        for idx, old_word in enumerate(code_list):
            if time.time() > deadline - 10.0:
                break
            
            # Try a few swap candidates
            for _ in range(10):
                if time.time() > deadline - 10.0:
                    break
                
                new_word = random.randint(0, (1 << 17) - 1)
                if new_word == old_word or new_word in code_set:
                    continue
                
                # Check if new_word is valid against all other codewords (excluding old_word)
                valid = True
                for w in code_set:
                    if w == old_word:
                        continue
                    if (new_word ^ w).bit_count() < 6:
                        valid = False
                        break
                
                if valid:
                    # Perform the swap
                    code_set.remove(old_word)
                    code_set.add(new_word)
                    improved = True
                    swap_improvements += 1
                    break  # Move to next old_word
    
        if not improved:
            break
    
    return list(code_set)
