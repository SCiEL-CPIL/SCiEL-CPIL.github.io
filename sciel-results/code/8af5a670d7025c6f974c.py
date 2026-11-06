import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # safety margin before 90s kill
    
    N = 17
    MIN_DIST = 6
    
    def time_left():
        return deadline - time.time()
    
    # GF(2^8) arithmetic
    PRIM_POLY = 0x11D
    
    def gf_mul(a, b):
        res = 0
        for i in range(8):
            if (b >> i) & 1:
                res ^= a
            a <<= 1
            if a & 0x100:
                a ^= PRIM_POLY
        return res

    def gf_pow(a, p):
        res = 1
        for _ in range(p):
            res = gf_mul(res, a)
        return res

    # Find alpha (primitive element)
    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1 and gf_pow(a, 1) != 1:
            alpha = a
            break
            
    if alpha is None:
        return greedy_construct(deadline)

    # Quadratic residues mod 17
    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Calculate generator polynomial g(x) = prod_{q in Q} (x + alpha^q)
    g_coeffs = [1]
    
    for q in Q:
        root = gf_pow(alpha, q)
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i] ^= gf_mul(g_coeffs[i], root)
            new_g[i+1] ^= g_coeffs[i]
        g_coeffs = new_g
        
    # Convert to binary polynomial coefficients
    bin_g = [1 if c == 1 else 0 for c in g_coeffs]
    while len(bin_g) < 9:
        bin_g.append(0)
        
    # Convert bin_g to integer for fast shifting
    g_int = 0
    for i in range(9):
        if bin_g[i]:
            g_int |= (1 << i)
            
    # Generate all 512 codewords of the [17,9,6] code
    base_code = []
    
    for m in range(512):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= (g_int << i)
        
        # Check even weight
        if c.bit_count() % 2 == 0:
            base_code.append(c)
            
    code = base_code[:]
    code_set = set(code)
    
    rng = random.Random(42)
    
    # Precompute candidate pool: all words of weight 6-12 not already in code
    candidate_pool = []
    for w in range(6, 13):
        for combo in combinations(range(N), w):
            c = 0
            for p in combo:
                c |= (1 << p)
            if c not in code_set:
                candidate_pool.append(c)
    
    rng.shuffle(candidate_pool)
    
    # Maintain code as a list of integers, and a numpy matrix for fast distance checks
    code_matrix = np.array([[(c >> i) & 1 for i in range(N)] for c in code], dtype=np.uint8)
    
    def can_add(cand):
        """Check if cand has min distance >= MIN_DIST from all codewords."""
        if len(code_matrix) == 0:
            return True
        cand_bits = np.array([(cand >> i) & 1 for i in range(N)], dtype=np.uint8)
        diffs = code_matrix ^ cand_bits
        dists = diffs.sum(axis=1)
        return bool(dists.min() >= MIN_DIST)
    
    # Phase 1: Greedy addition from candidate pool
    added_count = 0
    for idx, cand in enumerate(candidate_pool):
        if time_left() < 5.0:
            break
        if cand in code_set:
            continue
        if can_add(cand):
            code.append(cand)
            code_set.add(cand)
            # Update matrix
            new_row = np.array([(cand >> i) & 1 for i in range(N)], dtype=np.uint8)
            code_matrix = np.vstack([code_matrix, new_row])
            added_count += 1
    
    # Phase 2: Random local search - try to add more words
    for iteration in range(5000):
        if time_left() < 5.0:
            break
        
        w = rng.randint(6, 12)
        positions = rng.sample(range(N), w)
        candidate = 0
        for p in positions:
            candidate |= (1 << p)
        
        if candidate in code_set:
            continue
        
        if can_add(candidate):
            code.append(candidate)
            code_set.add(candidate)
            new_row = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.uint8)
            code_matrix = np.vstack([code_matrix, new_row])
    
    # Phase 3: Try to swap - replace a codeword to allow adding more
    # This is a simple 1-swap local search
    if time_left() > 10.0:
        for _ in range(200):
            if time_left() < 5.0:
                break
            
            # Pick a random codeword to remove
            rem_idx = rng.randint(0, len(code) - 1)
            removed = code[rem_idx]
            
            # Try to find a replacement that allows at least one new word
            found_improvement = False
            for _ in range(200):
                if time_left() < 5.0:
                    break
                
                w = rng.randint(6, 12)
                positions = rng.sample(range(N), w)
                candidate = 0
                for p in positions:
                    candidate |= (1 << p)
                
                if candidate in code_set:
                    continue
                if candidate == removed:
                    continue
                
                # Temporarily remove 'removed' and check if candidate can be added
                code_set.remove(removed)
                # Rebuild matrix without removed for check
                mask = np.ones(len(code), dtype=bool)
                mask[rem_idx] = False
                temp_matrix = code_matrix[mask]
                
                cand_bits = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.uint8)
                diffs = temp_matrix ^ cand_bits
                dists = diffs.sum(axis=1)
                
                if len(dists) == 0 or dists.min() >= MIN_DIST:
                    # Candidate can be added after removing 'removed'
                    # Now check if we can also add another word
                    code_set.add(candidate)
                    temp_matrix2 = np.vstack([temp_matrix, cand_bits])
                    
                    # Try to add one more random word
                    improved = False
                    for _ in range(50):
                        if time_left() < 5.0:
                            break
                        w2 = rng.randint(6, 12)
                        pos2 = rng.sample(range(N), w2)
                        cand2 = 0
                        for p in pos2:
                            cand2 |= (1 << p)
                        if cand2 in code_set or cand2 == removed:
                            continue
                        cb2 = np.array([(cand2 >> i) & 1 for i in range(N)], dtype=np.uint8)
                        d2 = temp_matrix2 ^ cb2
                        dist2 = d2.sum(axis=1)
                        if len(dist2) == 0 or dist2.min() >= MIN_DIST:
                            # We can add cand2 too, so net gain of 1
                            code[rem_idx] = candidate
                            code_set.remove(candidate)
                            code_set.add(cand2)
                            code.append(cand2)
                            code_set.add(candidate)  # re-add candidate
                            code_matrix = np.vstack([temp_matrix2, cb2])
                            improved = True
                            break
                    
                    if improved:
                        found_improvement = True
                        break
                    else:
                        # Revert: remove candidate from set
                        code_set.remove(candidate)
                else:
                    # Can't add candidate, revert
                    code_set.add(removed)
                
                # If we didn't find improvement, restore
                if not found_improvement:
                    code_set.add(removed)
                    break
            
            if found_improvement:
                # Update code_matrix to reflect changes
                code_matrix = np.array([[(c >> i) & 1 for i in range(N)] for c in code], dtype=np.uint8)
    
    return code

def greedy_construct(deadline):
    code = [0]
    code_set = {0}
    rng = random.Random(42)
    while time.time() < deadline:
        w = rng.randint(3, 14)
        positions = rng.sample(range(17), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if c not in code_set:
            ok = True
            for w2 in code:
                if (c ^ w2).bit_count() < 6:
                    ok = False
                    break
            if ok:
                code.append(c)
                code_set.add(c)
    return code
