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
        
    # Generate 512 codewords of the [17, 9, 6] code
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
    
    # Convert to numpy for vectorized distance checks
    def code_to_matrix(code_list):
        n = len(code_list)
        mat = np.zeros((n, 17), dtype=np.uint8)
        for i, c in enumerate(code_list):
            for b in range(17):
                mat[i, b] = (c >> b) & 1
        return mat
    
    def popcount_vec(vec):
        return vec.sum(axis=1)
    
    # Pre-generate masks for ball of radius 5 around 0
    # We need all 17-bit integers with weight <= 5
    ball_masks = []
    for w in range(1 << 17):
        if w.bit_count() <= 5:
            ball_masks.append(w)
    # Exclude zero mask
    ball_masks = [m for m in ball_masks if m != 0]
    
    # Shuffle for randomization
    random.shuffle(ball_masks)
    
    # Helper to check compatibility using numpy
    def is_compatible_fast(w, mat, n_rows):
        """Check if word w is at distance >= 6 from all codewords in mat[:n_rows]"""
        w_bits = np.array([(w >> b) & 1 for b in range(17)], dtype=np.uint8)
        if n_rows == 0:
            return True
        dists = popcount_vec(mat[:n_rows] ^ w_bits)
        return np.min(dists) >= 6
    
    # Greedy add phase: try to add words one at a time
    # Use a priority queue approach: try words in random order
    all_candidates = list(range(1 << 17))
    random.shuffle(all_candidates)
    
    # Remove existing codewords from candidates
    existing = set(code_list)
    candidates = [w for w in all_candidates if w not in existing]
    
    # Try to add as many as possible greedily
    mat = code_to_matrix(code_list)
    n_mat = len(code_list)
    
    added = 0
    for w in candidates:
        if time.time() > deadline - 10:
            break
        if is_compatible_fast(w, mat, n_mat):
            code_list.append(w)
            existing.add(w)
            # Update matrix
            new_mat = np.zeros((n_mat + 1, 17), dtype=np.uint8)
            new_mat[:n_mat] = mat
            for b in range(17):
                new_mat[n_mat, b] = (w >> b) & 1
            mat = new_mat
            n_mat += 1
            added += 1
            if added >= 100:  # Limit greedy adds to save time for swaps
                break
    
    # Now do 1-for-2 swaps to further improve
    # Rebuild matrix
    mat = code_to_matrix(code_list)
    n_mat = len(code_list)
    code_set = set(code_list)
    
    indices = list(range(n_mat))
    random.shuffle(indices)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
                
            c_rem = code_list[idx]
            
            # Find candidates in Ball(c_rem, 5) that are compatible with C \ {c_rem}
            candidates_found = []
            
            for mask in ball_masks:
                w = c_rem ^ mask
                if w in code_set:
                    continue
                    
                # Check compatibility with all other codewords
                # Build a temporary matrix without c_rem
                if is_compatible_fast(w, np.delete(mat, idx, axis=0), n_mat - 1):
                    candidates_found.append(w)
                    if len(candidates_found) > 30:
                        break
            
            if len(candidates_found) < 2:
                continue
                
            # Find a pair in candidates with distance >= 6
            pair_found = None
            for i in range(len(candidates_found)):
                for j in range(i+1, len(candidates_found)):
                    if (candidates_found[i] ^ candidates_found[j]).bit_count() >= 6:
                        pair_found = (candidates_found[i], candidates_found[j])
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
                
                # Rebuild matrix
                mat = code_to_matrix(code_list)
                n_mat = len(code_list)
                
                improved = True
                break  # Break out of the for idx loop to restart the while loop
            
    return code_list
