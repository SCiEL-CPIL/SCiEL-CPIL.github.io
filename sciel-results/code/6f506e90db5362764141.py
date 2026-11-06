import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 17
    MIN_DIST = 6
    
    # GF(256) utilities
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

    # Find primitive element
    alpha = None
    for c in range(2, 256):
        if gf_pow(c, 255) == 1 and gf_pow(c, 17) != 1 and gf_pow(c, 15) != 1:
            # Check if it generates the field
            # 255 = 3 * 5 * 17
            if gf_pow(c, 255 // 3) != 1 and gf_pow(c, 255 // 5) != 1 and gf_pow(c, 255 // 17) != 1:
                alpha = c
                break
    if alpha is None:
        alpha = 2 # Fallback, likely works with 0x11D

    # Generate QR code [17, 9, 5]
    # Roots are alpha^i for i in {1, 2, 4, 8, 16, 15, 13, 9}
    roots = [gf_pow(alpha, i) for i in [1, 2, 4, 8, 16, 15, 13, 9]]
    
    # Build generator polynomial g(x)
    # g(x) = prod (x - root)
    # Coeffs in GF(256)
    g_coeffs = [1]
    for r in roots:
        new_coeffs = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_coeffs[i+1] = g_coeffs[i]
            new_coeffs[i] ^= gf_mul(g_coeffs[i], r)
        g_coeffs = new_coeffs
        
    # g_coeffs is [g_8, g_7, ..., g_0]
    # Convert to binary generator matrix
    # Each coefficient is 8 bits.
    # The code is binary, so we need to map GF(256) coeffs to binary rows.
    # This is a bit complex. Easier to generate codewords directly?
    # No, standard way:
    # The generator matrix G has 9 rows.
    # Row k is the binary representation of x^k g(x) mod x^17 - 1?
    # Actually, for cyclic codes over GF(2), the generator polynomial must be in GF(2)[x].
    # But the QR code is not a binary cyclic code in the simple sense?
    # Wait, the QR code of length 17 IS a binary cyclic code.
    # Its generator polynomial is in GF(2)[x].
    # The roots are in GF(256), but the polynomial coefficients are in GF(2).
    # So g_coeffs should all be 0 or 1.
    
    # Let's check if g_coeffs are 0 or 1.
    valid_g = all(c in [0, 1] for c in g_coeffs)
    
    if not valid_g:
        # Fallback to random/greedy if QR construction fails
        # But it should work.
        pass

    g_bits = [1 if c == 1 else 0 for c in g_coeffs]
    # g_bits length 9. g_bits[0] is coeff of x^8.
    
    # Generate all 512 codewords of the [17, 9, 5] code
    codewords_17 = []
    for u in range(512):
        c = 0
        for k in range(9):
            if (u >> k) & 1:
                # Add x^k * g(x)
                for j in range(9):
                    if g_bits[j]:
                        c ^= 1 << (j + k)
        codewords_17.append(c)
        
    # Extend to [18, 9, 6]
    extended_codewords = []
    for c in codewords_17:
        parity = bin(c).count('1') % 2
        ec = c | (parity << 17)
        extended_codewords.append(ec)
        
    # Shorten by fixing bit 17 (the parity bit) to 0
    shortened_codewords = []
    for ec in extended_codewords:
        if (ec >> 17) & 1 == 0:
            shortened_codewords.append(ec & ((1 << 17) - 1))
            
    # Remove duplicates (should be 256 unique)
    current_code = list(set(shortened_codewords))
    
    if len(current_code) < 256:
        # Fallback: if QR generation failed, use a smaller seed
        current_code = [0]
        
    # --- Greedy Additive Search ---
    
    # Precompute neighbors offsets: all words w with 0 < weight(w) <= 5
    offsets = []
    for w in range(1, 1 << 17):
        wt = w.bit_count()
        if 1 <= wt <= 5:
            offsets.append(w)
            
    # forbidden array
    forbidden = np.zeros(1 << 17, dtype=np.uint8)
    
    # Mark neighbors of initial code
    for c in current_code:
        for off in offsets:
            forbidden[c ^ off] = 1
    # Also mark the codewords themselves as forbidden (so we don't add them again)
    for c in current_code:
        forbidden[c] = 1
        
    # List of available candidates
    # We will iterate through all words. 
    # To improve performance, we can pre-filter or just iterate and check.
    # Iterating 131k words is fast.
    
    # We want to try multiple random orders to find a better packing
    best_code = current_code.copy()
    best_size = len(best_code)
    
    # Number of restarts based on time
    # Each restart:
    # 1. Reset forbidden (or copy from initial state)
    # 2. Shuffle candidates
    # 3. Greedy add
    
    # To speed up restarts, we can save the initial forbidden state
    initial_forbidden = forbidden.copy()
    
    # We will try to add words. 
    # Heuristic: 
    # Instead of just random order, we can prioritize words that are "central" or have low weight?
    # Or just random is often good enough for small improvements.
    
    # Let's do a few restarts with different random seeds
    for trial in range(20):
        if time.time() > deadline - 5.0:
            break
            
        forbidden = initial_forbidden.copy()
        code = best_code.copy()
        
        # Get list of all words not forbidden
        # This list is static for the start of the trial
        # We just iterate through a shuffled list of ALL words and check forbidden status
        all_words = list(range(1 << 17))
        random.seed(42 + trial)
        random.shuffle(all_words)
        
        added = 0
        for w in all_words:
            if forbidden[w]:
                continue
                
            # Check if w is already in code (should be forbidden, but just in case)
            # Add w
            code.append(w)
            forbidden[w] = 1
            
            # Mark neighbors
            for off in offsets:
                forbidden[w ^ off] = 1
                
            added += 1
            if added > 50: # Safety break, shouldn't happen
                break
                
        if len(code) > best_size:
            best_size = len(code)
            best_code = code
            
    # Final verification
    result = best_code
    # Ensure distinct
    result = list(set(result))
    
    # Verify distance
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if (result[i] ^ result[j]).bit_count() < MIN_DIST:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to QR code only
        result = list(set(current_code))
        
    return result

