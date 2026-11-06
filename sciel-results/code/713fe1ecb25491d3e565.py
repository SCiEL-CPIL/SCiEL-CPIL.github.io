import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin

    # --- Part 1: Construct the 256-word linear [17, 9, 6] code ---
    
    # GF(256) parameters
    poly = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            n >>= 1
        return res

    def get_primitive_17th_root():
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    
    # Fallback seed if GF construction fails (should not happen)
    if beta is None:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Build generator polynomial G(x)
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
        
    g_bits = [0] * 9
    for i, c in enumerate(poly_coeffs):
        g_bits[i] = 1 if c == 1 else 0
        
    # Generate 512 codewords of the [17, 9, 6] code? 
    # Wait, the generator polynomial degree is 8?
    # The QR code of length 17 has dimension 9?
    # The roots are beta^1, ..., beta^8? No, the number of roots is the redundancy.
    # For [17, 9, 6], redundancy is 8. So degree of G(x) is 8.
    # Number of codewords is 2^(17-8) = 2^9 = 512?
    # Let's check the task description: "The 256-word linear reference is the [18,9,6] extended quadratic-residue code shortened by one coordinate".
    # [18, 9, 6] has 2^9 = 512 words.
    # Shortening by 1 coordinate:
    # Fix one coordinate to 0.
    # The number of words with that coordinate 0 is half, so 256.
    # So we generate 512 words, extend to 18 bits, filter for parity bit 0 (or fixed coord 0), resulting in 256 words.
    
    codewords_17 = []
    for u in range(512):
        c = 0
        for k in range(9):
            if (u >> k) & 1:
                for j in range(9):
                    if g_bits[j]:
                        c ^= 1 << (j + k)
        codewords_17.append(c)
        
    # Extend to 18 bits: add parity bit at pos 17
    # Shorten: keep words where bit 17 is 0, then drop bit 17.
    shortened_codewords = []
    for c in codewords_17:
        parity = bin(c).count('1') % 2
        ec = c | (parity << 17)
        if (ec >> 17) & 1 == 0:
            shortened_codewords.append(ec & ((1 << 17) - 1))
            
    # Remove duplicates
    code_set = set(shortened_codewords)
    if len(code_set) < 256:
         # Fallback
         return list(code_set)
         
    code_list = list(code_set)
    n_words = len(code_list)
    
    # --- Part 2: Local Search for Nonlinear Expansion ---
    
    # Helper to check if a word is compatible with a list of codewords
    def is_compatible(w, code_arr, min_dist=6):
        for c in code_arr:
            if (w ^ c).bit_count() < min_dist:
                return False
        return True

    # Pre-generate masks for ball of radius 5 around 0
    # We need all 17-bit integers with weight <= 5
    ball_masks = []
    for w in range(1 << 17):
        if w.bit_count() <= 5:
            ball_masks.append(w)
            
    # Shuffle for randomization? 
    # Actually, the order doesn't matter much for correctness, but for finding solutions faster.
    # We'll iterate through them.
    
    # We will try to perform 1-for-2 swaps.
    # We iterate through the current code words as potential removal candidates.
    # To avoid bias, we can shuffle the code list indices.
    
    indices = list(range(n_words))
    random.shuffle(indices)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Re-shuffle indices each pass? Or just iterate?
        # Let's iterate through a shuffled list of indices.
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
                
            c_rem = code_list[idx]
            
            # Find candidates in Ball(c_rem, 5) that are compatible with C \ {c_rem}
            candidates = []
            
            # Iterate through ball masks
            for mask in ball_masks:
                w = c_rem ^ mask
                # Check if w is already in the code (excluding c_rem)
                # If w is in the code, it's not a new word.
                # We can skip if w == c_rem (mask=0).
                if mask == 0:
                    continue
                    
                # Quick check: is w in code_set?
                # We maintain a set for O(1) lookup.
                if w in code_set:
                    continue
                    
                # Check compatibility with all other codewords
                compatible = True
                for j, c in enumerate(code_list):
                    if j == idx:
                        continue
                    if (w ^ c).bit_count() < 6:
                        compatible = False
                        break
                if compatible:
                    candidates.append(w)
                    # Limit candidates to avoid huge lists
                    if len(candidates) > 50:
                        break
            
            if len(candidates) < 2:
                continue
                
            # Find a pair in candidates with distance >= 6
            pair_found = None
            for i in range(len(candidates)):
                for j in range(i+1, len(candidates)):
                    if (candidates[i] ^ candidates[j]).bit_count() >= 6:
                        pair_found = (candidates[i], candidates[j])
                        break
                if pair_found:
                    break
                    
            if pair_found:
                w1, w2 = pair_found
                # Perform swap: remove c_rem, add w1, w2
                code_list.remove(c_rem)
                code_set.remove(c_rem)
                
                code_list.append(w1)
                code_set.add(w1)
                code_list.append(w2)
                code_set.add(w2)
                
                improved = True
                # Restart the loop to re-evaluate with the new code
                break # Break out of the for idx loop to restart the while loop
            
    return code_list
