import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 17
    MIN_DIST = 6
    
    # 1. Construct the [17, 8, 6] linear code (256 words)
    # GF(2^8) arithmetic
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11d)
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11d
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # beta = 3 is a primitive element for GF(2^8) with 0x11d
    beta = 3
    alpha = gf_pow(beta, 15) # alpha^17 = 1
    
    # Find minimal polynomial of alpha^3
    a3 = gf_pow(alpha, 3)
    v = [1] * 9
    for i in range(1, 9):
        v[i] = gf_mul(v[i-1], a3)
        
    # Solve for coefficients c_0..c_7 of m_3(x) = x^8 + c_7 x^7 + ... + c_0
    M = [[0]*8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            M[j][i] = (v[i] >> j) & 1
            
    b = [(v[8] >> j) & 1 for j in range(8)]
    
    # Gaussian elimination over GF(2)
    A = [row[:] + [b[i]] for i, row in enumerate(M)]
    for col in range(8):
        pivot = -1
        for row in range(col, 8):
            if A[row][col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        A[col], A[pivot] = A[pivot], A[col]
        for row in range(8):
            if row != col and A[row][col] == 1:
                for k in range(col, 9):
                    A[row][k] ^= A[col][k]
                    
    c = [A[i][8] for i in range(8)]
    
    # Construct m_3(x)
    m3_poly = 0
    for i in range(8):
        if c[i]:
            m3_poly |= (1 << i)
    m3_poly |= (1 << 8)
    
    # h(x) = (x+1)m_3(x)
    def poly_mul(x, y):
        res = 0
        while y:
            if y & 1:
                res ^= x
            y >>= 1
            x <<= 1
        return res
        
    h_poly = poly_mul(0b11, m3_poly)
    
    # Generate codewords
    def poly_mod(p):
        # Reduce modulo x^17 - 1
        max_bit = p.bit_length()
        for i in range(max_bit - 1, 16, -1):
            if (p >> i) & 1:
                p ^= (1 << (i - 17))
        return p & ((1 << 17) - 1)

    base_words = set()
    for a in range(256):
        c = poly_mod(poly_mul(a, h_poly))
        base_words.add(c)
        
    base_list = list(base_words)
    
    # 2. Generate candidate words for nonlinear extension
    # We look for words with weight 8, 9, 10.
    # A word must be at distance >= 6 from all base words.
    
    candidates = []
    
    # Helper to check if a word is compatible with all base words
    def is_compatible_with_base(word):
        for bw in base_list:
            if (word ^ bw).bit_count() < MIN_DIST:
                return False
        return True

    # Generate candidates of weight 8 and 9
    # Using itertools.combinations for bit positions
    for w in [8, 9]:
        for pos in itertools.combinations(range(N), w):
            word = 0
            for p in pos:
                word |= (1 << p)
            if is_compatible_with_base(word):
                candidates.append(word)
                
    # Also try weight 10 and 7? 
    # Weight 7: distance to 0 is 7 (ok). But might be close to other codewords.
    # Weight 10: distance to 0 is 10 (ok).
    # Let's add 7 and 10 if time permits.
    if time.time() < deadline - 10.0:
        for w in [7, 10]:
            for pos in itertools.combinations(range(N), w):
                word = 0
                for p in pos:
                    word |= (1 << p)
                if is_compatible_with_base(word):
                    candidates.append(word)
                    
    # 3. Find a large clique among candidates
    # Clique members must be pairwise distance >= 6.
    # Greedy approach with multiple random orderings.
    
    best_clique = []
    
    # To speed up clique checking, we can pre-compute compatibility matrix?
    # Number of candidates might be a few thousand.
    # Let's check size.
    # If candidates is large, we need a faster heuristic.
    
    def find_clique(cand_list, max_time=5.0):
        if not cand_list:
            return []
        
        # Sort candidates by weight? Or random?
        # Let's try a greedy algorithm:
        # 1. Pick a candidate with max "compatibility score" (number of other candidates it's compatible with).
        # 2. Add to clique.
        # 3. Remove it and all incompatible candidates from the pool.
        # 4. Repeat.
        
        # Pre-computing degrees is O(V^2).
        # If V is ~5000, V^2 = 25M. This is okay in Python if optimized?
        # (5000*5000)/2 = 12.5M pairs.
        # Let's try it.
        
        V = len(cand_list)
        if V == 0:
            return []
        
        # Build adjacency list or just check on the fly?
        # On the fly is O(V^2) per clique finding step if we re-scan.
        # Better:
        # 1. Compute degree for each node.
        # 2. Pick node with highest degree.
        # 3. Remove neighbors.
        
        # Let's use a simpler iterative greedy:
        # Random shuffle candidates.
        # Iterate through shuffled list.
        # If candidate is compatible with all in current clique, add it.
        
        # This finds a clique, but not necessarily the largest.
        # To find a better one, we can run this multiple times with different shuffles.
        
        best_local = []
        rng = random.Random(42)
        
        # We can also try to order by weight or some heuristic.
        # For now, random shuffles.
        
        # Optimization: If V is small, we can be more exhaustive.
        # If V is large, we rely on the greedy.
        
        for _ in range(20): # 20 restarts
            shuffled = cand_list[:]
            rng.shuffle(shuffled)
            
            current_clique = []
            # To speed up checking, maintain a list of active candidates?
            # Just checking against current_clique is O(|C|) per candidate.
            # Total O(V * |C|).
            
            for word in shuffled:
                compatible = True
                for cw in current_clique:
                    if (word ^ cw).bit_count() < MIN_DIST:
                        compatible = False
                        break
                if compatible:
                    current_clique.append(word)
            
            if len(current_clique) > len(best_local):
                best_local = current_clique
                
            # Early stop if we hit a very good clique?
            # Theoretical max extension is ~80 (340-256).
            # If we find > 70, we can stop.
            if len(best_local) > 70:
                break
                
        return best_local

    # Run clique finding
    if candidates:
        # Time budget for clique finding: 20 seconds
        clique_start = time.time()
        extra_words = find_clique(candidates, max_time=20.0)
        clique_end = time.time()
    else:
        extra_words = []
        
    final_code = base_list + extra_words
    
    # Verify final code (safety check)
    # This is expensive for 300+ words. O(N^2).
    # 300^2 = 90,000 checks. Fast.
    def verify(code):
        n = len(code)
        for i in range(n):
            for j in range(i+1, n):
                if (code[i] ^ code[j]).bit_count() < MIN_DIST:
                    return False
        return True
        
    if not verify(final_code):
        # Fallback to base code if verification fails
        return base_list
        
    return final_code
