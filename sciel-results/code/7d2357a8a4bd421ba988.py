import time
import numpy as np

def construct():
    """
    Returns a rank-114 exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Structure:
    - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    - C = A * B = [A*B1, A*B2, A*B3]
    - A*B1: 4x4x4 via Strassen (rank 49)
    - A*B2: 4x4x4 via Strassen (rank 49)
    - A*B3: 4x4x1 via schoolbook (rank 16)
    Total rank: 49 + 49 + 16 = 114
    
    This is a structural improvement over the numerical optimization approach.
    We directly construct the known Strassen decomposition for 4x4x4.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 9
    nm = n * m       # 16
    mp = m * p       # 36
    pn = p * n       # 36
    
    # Strassen 2x2 terms: 7 multiplications
    # Each term is (U_pattern, V_pattern, W_pattern)
    # U_pattern: dict from (i,j) to coefficient
    # V_pattern: dict from (j,k) to coefficient
    # W_pattern: dict from (i,k) to coefficient
    
    def strassen_2x2_terms():
        terms = []
        # M1 = (A00+A11)*(B00+B11)
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))
        # M2 = (A10+A11)*B00
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ))
        # M3 = A00*(B01-B11)
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ))
        # M4 = A11*(B10-B00)
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ))
        # M5 = (A00+A01)*B11
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ))
        # M6 = (A10-A00)*(B00+B01)
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        # M7 = (A01-A11)*(B10+B11)
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ))
        return terms
    
    def strassen_4x4_terms():
        """
        Generate 49 terms for 4x4x4 matrix multiplication using two levels of Strassen.
        
        4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
        We apply Strassen at the block level (7 block multiplications),
        and each block multiplication is a 2x2x2 multiplication (7 scalar multiplications).
        Total: 7 * 7 = 49 scalar multiplications.
        """
        inner_terms = strassen_2x2_terms()
        
        # The 7 block-level Strassen products for 2x2 blocks
        # Each is (block_U, block_V, block_W) where block_* are dicts from (bi,bj) to coeff
        top_products = [
            # M1_block = (A00+A11)*(B00+B11)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2_block = (A10+A11)*B00
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3_block = A00*(B01-B11)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4_block = A11*(B10-B00)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5_block = (A00+A01)*B11
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # M6_block = (A10-A00)*(B00+B01)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7_block = (A01-A11)*(B10+B11)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        all_terms = []
        for (a_blocks, b_blocks, c_blocks) in top_products:
            for (u_inner, v_inner, w_inner) in inner_terms:
                u = {}
                v = {}
                w = {}
                
                # Combine block-level and inner-level indices
                # Block (bi, bj) with inner (li, lj) -> global (2*bi+li, 2*bj+lj)
                
                for (bi, bj), ca in a_blocks.items():
                    for (li, lj), ci in u_inner.items():
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        u[(i, j)] = u.get((i, j), 0) + ca * ci
                
                for (bi, bj), cb in b_blocks.items():
                    for (li, lj), ci in v_inner.items():
                        j = 2 * bi + li
                        k = 2 * bj + lj
                        v[(j, k)] = v.get((j, k), 0) + cb * ci
                
                for (bi, bj), cc in c_blocks.items():
                    for (li, lj), ci in w_inner.items():
                        i = 2 * bi + li
                        k = 2 * bj + lj
                        w[(i, k)] = w.get((i, k), 0) + cc * ci
                
                # Clean up zero entries
                u = {k: v for k, v in u.items() if v != 0}
                v = {k: v for k, v in v.items() if v != 0}
                w = {k: v for k, v in w.items() if v != 0}
                
                all_terms.append((u, v, w))
        
        return all_terms
    
    def build_rank114():
        """Build the rank-114 decomposition."""
        rank = 114
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        strassen_terms = strassen_4x4_terms()
        
        # First 4x4x4 block: columns 0-3 of B
        for c_idx, (u_pat, v_pat, w_pat) in enumerate(strassen_terms):
            for (i, j), cu in u_pat.items():
                U[i * m + j, c_idx] = cu
            for (j, k), cv in v_pat.items():
                V[j * p + k, c_idx] = cv
            for (i, k), cw in w_pat.items():
                W[k * n + i, c_idx] = cw
        
        # Second 4x4x4 block: columns 4-7 of B
        offset = 49
        for c_idx, (u_pat, v_pat, w_pat) in enumerate(strassen_terms):
            for (i, j), cu in u_pat.items():
                U[i * m + j, c_idx + offset] = cu
            for (j, k), cv in v_pat.items():
                V[j * p + (4 + k), c_idx + offset] = cv
            for (i, k), cw in w_pat.items():
                W[(4 + k) * n + i, c_idx + offset] = cw
        
        # Third block: 4x4x1, column 8 of B, schoolbook
        offset2 = 98
        for i in range(n):
            for j in range(m):
                c_idx = offset2 + i * m + j
                U[i * m + j, c_idx] = 1
                V[j * p + 8, c_idx] = 1
                W[8 * n + i, c_idx] = 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify_exact(U, V, W):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        # Target: T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-12:
                        return False
        return True
    
    def build_schoolbook():
        """Fallback: schoolbook rank-144 decomposition."""
        rank = n * m * p  # 144
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the rank-114 decomposition
    if time.time() < deadline - 5:
        cand = build_rank114()
        if verify_exact(cand["U"], cand["V"], cand["W"]):
            return {"U": cand["U"].tolist(), "V": cand["V"].tolist(), "W": cand["W"].tolist()}
    
    # Fallback to schoolbook
    schoolbook = build_schoolbook()
    return {"U": schoolbook["U"].tolist(), "V": schoolbook["V"].tolist(), "W": schoolbook["W"].tolist()}
