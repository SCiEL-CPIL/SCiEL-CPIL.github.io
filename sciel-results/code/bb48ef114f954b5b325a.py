import numpy as np
import time
from fractions import Fraction

START_TIME = time.time()
TIME_LIMIT = 115

def construct():
    n, m, p = 3, 5, 7

    def make_decomposition(terms):
        """Convert list of (U_dict, V_dict, W_dict) to matrix format."""
        rank = len(terms)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for r, (u, v, w) in enumerate(terms):
            for idx, val in u.items():
                U[idx][r] = val
            for idx, val in v.items():
                V[idx][r] = val
            for idx, val in w.items():
                W[idx][r] = val
        return {'U': U, 'V': V, 'W': W}

    def verify(U, V, W):
        """Verify decomposition correctness."""
        rank = len(U[0]) if U else 0
        # Check C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Actually the tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal delta (always 1 for valid (i,j,k))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook_terms(n_r, m_m, p_c, r_off, m_off, c_off):
        """Generate schoolbook rank-1 terms."""
        terms = []
        for i in range(n_r):
            for j in range(m_m):
                for k in range(p_c):
                    ai = r_off + i
                    aj = m_off + j
                    ak = c_off + k
                    if ai < n and aj < m and ak < p:
                        terms.append(({ai * m + aj: 1.0}, {aj * p + ak: 1.0}, {ak * n + ai: 1.0}))
        return terms

    def strassen_2x2x2_terms(r_off, m_off, c_off):
        """Generate 7 Strassen terms for 2x2x2 block at given offset."""
        # A = [[A00, A01], [A10, A11]], B = [[B00, B10], [B11, B21]]
        # Using standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B00-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # Each M is a scalar product of (sum of A entries) * (sum of B entries)
        # contributing to specific C entries.
        #
        # For the tensor decomposition, each rank-1 term is:
        # U factor: product of A coefficients for this term
        # V factor: product of B coefficients for this term
        # W factor: which C entries this term contributes to (with signs)
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11) -> C00: +1, C11: +1
        u = {(r_off)*m + m_off: 1.0, (r_off+1)*m + (m_off+1): 1.0}
        v = {m_off*p + c_off: 1.0, (m_off+1)*p + (c_off+1): 1.0}
        w = {c_off*n + r_off: 1.0, (c_off+1)*n + (r_off+1): 1.0}
        terms.append((u, v, w))
        
        # M2 = (A10+A11)B00 -> C10: +1, C11: -1
        u = {(r_off+1)*m + m_off: 1.0, (r_off+1)*m + (m_off+1): 1.0}
        v = {m_off*p + c_off: 1.0}
        w = {c_off*n + (r_off+1): 1.0, (c_off+1)*n + (r_off+1): -1.0}
        terms.append((u, v, w))
        
        # M3 = A00(B10-B11) -> C01: +1, C11: +1
        u = {r_off*m + m_off: 1.0}
        v = {(m_off+1)*p + c_off: 1.0, (m_off+1)*p + (c_off+1): -1.0}
        w = {(c_off+1)*n + r_off: 1.0, (c_off+1)*n + (r_off+1): 1.0}
        terms.append((u, v, w))
        
        # M4 = A11(B00-B10) -> C00: +1, C10: +1
        u = {(r_off+1)*m + (m_off+1): 1.0}
        v = {m_off*p + c_off: 1.0, (m_off+1)*p + c_off: -1.0}
        w = {c_off*n + r_off: 1.0, c_off*n + (r_off+1): 1.0}
        terms.append((u, v, w))
        
        # M5 = (A00+A01)B11 -> C00: -1, C01: +1
        u = {r_off*m + m_off: 1.0, r_off*m + (m_off+1): 1.0}
        v = {(m_off+1)*p + (c_off+1): 1.0}
        w = {c_off*n + r_off: -1.0, (c_off+1)*n + r_off: 1.0}
        terms.append((u, v, w))
        
        # M6 = (A10-A00)(B00+B01) -> C11: +1
        u = {(r_off+1)*m + m_off: 1.0, r_off*m + m_off: -1.0}
        v = {m_off*p + c_off: 1.0, m_off*p + (c_off+1): 1.0}
        w = {(c_off+1)*n + (r_off+1): 1.0}
        terms.append((u, v, w))
        
        # M7 = (A01-A11)(B10+B11) -> C00: +1
        u = {r_off*m + (m_off+1): 1.0, (r_off+1)*m + (m_off+1): -1.0}
        v = {(m_off+1)*p + c_off: 1.0, (m_off+1)*p + (c_off+1): 1.0}
        w = {c_off*n + r_off: 1.0}
        terms.append((u, v, w))
        
        return terms

    def decompose_block(n_r, m_m, p_c, r_off, m_off, c_off):
        """Decompose a block, using Strassen for 2x2x2, schoolbook otherwise."""
        if n_r <= 0 or m_m <= 0 or p_c <= 0:
            return []
        
        # Check bounds
        if r_off + n_r > n or m_off + m_m > m or c_off + p_c > p:
            # Clip to valid region
            n_r = min(n_r, n - r_off)
            m_m = min(m_m, m - m_off)
            p_c = min(p_c, p - c_off)
            if n_r <= 0 or m_m <= 0 or p_c <= 0:
                return []
        
        terms = []
        
        # Try to use Strassen for 2x2x2 blocks
        if n_r >= 2 and m_m >= 2 and p_c >= 2:
            # Use 2x2x2 Strassen for the top-left 2x2x2 corner
            # Then handle the remaining strips with schoolbook
            terms.extend(strassen_2x2x2_terms(r_off, m_off, c_off))
            # Remaining: row strip (n_r-2, m_m, p_c) at r_off+2
            #            col strip (n_r, m_m, p_c-2) at c_off+2
            #            mid strip (n_r, m_m-2, p_c) at m_off+2
            # But this double-counts. Better to split properly.
            
            # Actually, let me use a cleaner approach:
            # Split into 2x2x2 + remaining strips
            # The 2x2x2 handles rows [0,2), mids [0,2), cols [0,2)
            # Remaining:
            #   rows [2,n_r), all mids, all cols: (n_r-2) x m_m x p_c
            #   rows [0,2), mids [2,m_m), all cols: 2 x (m_m-2) x p_c
            #   rows [0,2), all mids, cols [2,p_c): 2 x m_m x (p_c-2)
            # But these overlap! I need to be more careful.
            
            # Better approach: just use schoolbook for non-2x2x2 cases
            # and only use Strassen when the block is exactly 2x2x2
            pass
        
        # For simplicity and correctness, use schoolbook for all blocks
        # except when we can cleanly apply Strassen
        if n_r == 2 and m_m == 2 and p_c == 2:
            return strassen_2x2x2_terms(r_off, m_off, c_off)
        
        return schoolbook_terms(n_r, m_m, p_c, r_off, m_off, c_off)

    # Strategy 1: Simple schoolbook
    terms1 = schoolbook_terms(n, m, p, 0, 0, 0)
    
    # Strategy 2: Split m=5 into 2+2+1, use Strassen for 2x2x2 blocks
    terms2 = []
    # m=5: split as 2+2+1
    # Block 1: 3x2x7 with mids [0,2)
    #   Split p=7 into 2+2+2+1
    #   Sub-blocks: 3x2x2, 3x2x2, 3x2x2, 3x2x1
    #   For 3x2x2: split n=3 into 2+1
    #     2x2x2 (Strassen) + 1x2x2 (schoolbook)
    for m_start in [0, 2, 4]:
        m_end = min(m_start + 2, m)
        m_width = m_end - m_start
        if m_width == 0:
            continue
        for p_start in range(0, p, 2):
            p_end = min(p_start + 2, p)
            p_width = p_end - p_start
            if p_width == 0:
                continue
            for n_start in range(0, n, 2):
                n_end = min(n_start + 2, n)
                n_width = n_end - n_start
                if n_width == 0:
                    continue
                if n_width == 2 and m_width == 2 and p_width == 2:
                    terms2.extend(strassen_2x2x2_terms(n_start, m_start, p_start))
                else:
                    terms2.extend(schoolbook_terms(n_width, m_width, p_width, n_start, m_start, p_start))
    
    # Strategy 3: More aggressive Strassen - split all dimensions
    terms3 = []
    # Split n=3 as 2+1, m=5 as 2+2+1, p=7 as 2+2+2+1
    n_splits = [(0, 2), (2, 3)]
    m_splits = [(0, 2), (2, 4), (4, 5)]
    p_splits = [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    for n_s, n_e in n_splits:
        for m_s, m_e in m_splits:
            for p_s, p_e in p_splits:
                nr, mr, pr = n_e - n_s, m_e - m_s, p_e - p_s
                if nr == 2 and mr == 2 and pr == 2:
                    terms3.extend(strassen_2x2x2_terms(n_s, m_s, p_s))
                else:
                    terms3.extend(schoolbook_terms(nr, mr, pr, n_s, m_s, p_s))
    
    # Strategy 4: Different split - n=3 as 1+2, m=5 as 1+2+2, p=7 as 1+2+2+2
    terms4 = []
    n_splits4 = [(0, 1), (1, 3)]
    m_splits4 = [(0, 1), (1, 3), (3, 5)]
    p_splits4 = [(0, 1), (1, 3), (3, 5), (5, 7)]
    
    for n_s, n_e in n_splits4:
        for m_s, m_e in m_splits4:
            for p_s, p_e in p_splits4:
                nr, mr, pr = n_e - n_s, m_e - m_s, p_e - p_s
                if nr == 2 and mr == 2 and pr == 2:
                    terms4.extend(strassen_2x2x2_terms(n_s, m_s, p_s))
                else:
                    terms4.extend(schoolbook_terms(nr, mr, pr, n_s, m_s, p_s))
    
    # Strategy 5: Optimize split points to maximize 2x2x2 Strassen blocks
    # n=3: can have at most one 2-row block (either rows 0-1 or 1-2)
    # m=5: can have at most two 2-mid blocks (e.g., 0-1,2-3 or 1-2,3-4)
    # p=7: can have at most three 2-col blocks (e.g., 0-1,2-3,4-5)
    
    # Try all combinations of where to place 2x2x2 blocks
    best_terms = None
    best_rank = n * m * p
    
    # For each strategy, verify and pick the best
    candidates = [
        ('schoolbook', terms1),
        ('strat2', terms2),
        ('strat3', terms3),
        ('strat4', terms4),
    ]
    
    for name, terms in candidates:
        if len(terms) < best_rank:
            result = make_decomposition(terms)
            if verify(result['U'], result['V'], result['W']):
                best_terms = terms
                best_rank = len(terms)
    
    # Try more systematic search for optimal splitting
    if time.time() - START_TIME < TIME_LIMIT - 5:
        # Systematic: try all ways to partition each dimension into blocks of size 1 or 2
        # n=3: partitions of 3 into 1s and 2s: [1,1,1], [1,2], [2,1]
        # m=5: [1,1,1,1,1], [1,1,1,2], [1,1,2,1], [1,2,1,1], [2,1,1,1], [1,2,2], [2,1,2], [2,2,1]
        # p=7: similar
        
        def get_partitions(total, max_part=2):
            """Get all partitions of total into parts of size 1 or max_part."""
            if total == 0:
                return [[]]
            results = []
            if total >= 1:
                for sub in get_partitions(total - 1, max_part):
                    results.append([1] + sub)
            if total >= max_part:
                for sub in get_partitions(total - max_part, max_part):
                    results.append([max_part] + sub)
            return results
        
        n_parts = get_partitions(n)
        m_parts = get_partitions(m)
        p_parts = get_partitions(p)
        
        # Limit search to avoid explosion
        max_combos = 500
        count = 0
        for np_ in n_parts:
            for mp_ in m_parts:
                for pp_ in p_parts:
                    if count >= max_combos:
                        break
                    count += 1
                    
                    terms = []
                    n_off = 0
                    for n_sz in np_:
                        m_off = 0
                        for m_sz in mp_:
                            p_off = 0
                            for p_sz in pp_:
                                if n_sz == 2 and m_sz == 2 and p_sz == 2:
                                    terms.extend(strassen_2x2x2_terms(n_off, m_off, p_off))
                                else:
                                    terms.extend(schoolbook_terms(n_sz, m_sz, p_sz, n_off, m_off, p_off))
                                p_off += p_sz
                            m_off += m_sz
                        n_off += n_sz
                    
                    if len(terms) < best_rank:
                        result = make_decomposition(terms)
                        if verify(result['U'], result['V'], result['W']):
                            best_terms = terms
                            best_rank = len(terms)
                    
                    if time.time() - START_TIME > TIME_LIMIT - 5:
                        break
                if time.time() - START_TIME > TIME_LIMIT - 5:
                    break
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
        
    if best_terms is None:
        best_terms = terms1
    
    return make_decomposition(best_terms)
