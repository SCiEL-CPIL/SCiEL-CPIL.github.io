import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    def verify_decomp(U, V, W, n=2, m=2, p=9):
        """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        R = U_arr.shape[1]
        U_r = U_arr.reshape(n, m, R)
        V_r = V_arr.reshape(m, p, R)
        W_r = W_arr.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-8)

    def build_strassen_rank32():
        """Build the known rank-32 decomposition: 4 Strassen blocks (rank 7) + schoolbook (rank 4)."""
        terms = []
        # Strassen block for columns k0, k1
        def add_strassen_block(k0, k1):
            # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            # Indices:
            # A: 00, 11 -> U indices 0, 3
            # B: 00, 11 -> V indices k0, 9+k1 (since B is 2x9, row 0 is 0..8, row 1 is 9..17)
            # Wait, V index is j*p+k. j=0 -> k, j=1 -> 9+k.
            # C: 00, 11 -> W indices 2*k0, 2*k1+1
            # W index is k*n+i. i=0 -> 2k, i=1 -> 2k+1.
            
            # Term 1: M1
            terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
            
            # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
            # A: 10, 11 -> U indices 2, 3
            # B: 00 -> V index k0
            # C: 10, 11 -> W indices 2*k0+1, 2*k1+1
            terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
            
            # M3 = A00(B10-B11) -> C00 -= M3, C01 += M3
            # A: 00 -> U index 0
            # B: 10, 11 -> V indices 9+k0, 9+k1
            # C: 00, 01 -> W indices 2*k0, 2*k1
            terms.append(({0: 1}, {9+k0: 1, 9+k1: -1}, {2*k0: -1, 2*k1: 1}))
            
            # M4 = A11(B01-B00) -> C10 += M4, C11 -= M4
            # A: 11 -> U index 3
            # B: 01, 00 -> V indices k1, k0
            # C: 10, 11 -> W indices 2*k0+1, 2*k1+1
            terms.append(({3: 1}, {k1: 1, k0: -1}, {2*k0+1: 1, 2*k1+1: -1}))
            
            # M5 = (A00+A01)B11 -> C00 += M5, C01 += M5
            # A: 00, 01 -> U indices 0, 1
            # B: 11 -> V index 9+k1
            # C: 00, 01 -> W indices 2*k0, 2*k1
            terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: 1, 2*k1: 1}))
            
            # M6 = (A10-A00)(B00+B10) -> C10 += M6, C11 += M6
            # A: 10, 00 -> U indices 2, 0
            # B: 00, 10 -> V indices k0, 9+k0
            # C: 10, 11 -> W indices 2*k0+1, 2*k1+1
            terms.append(({2: 1, 0: -1}, {k0: 1, 9+k0: 1}, {2*k0+1: 1, 2*k1+1: 1}))
            
            # M7 = (A01-A11)(B11+B01) -> C01 += M7, C11 += M7
            # A: 01, 11 -> U indices 1, 3
            # B: 11, 01 -> V indices 9+k1, k1
            # C: 01, 11 -> W indices 2*k1, 2*k1+1
            terms.append(({1: 1, 3: -1}, {9+k1: 1, k1: 1}, {2*k1: 1, 2*k1+1: 1}))

        # Add 4 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7)
        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            add_strassen_block(k0, k1)
        
        # Schoolbook block for column 8
        # C00 += A00 * B08
        # C01 += A01 * B18
        # C10 += A10 * B08
        # C11 += A11 * B18
        k = 8
        terms.append(({0: 1}, {k: 1}, {2*k: 1}))
        terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
        terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
        terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
        
        return terms

    def terms_to_matrices(terms, n=2, m=2, p=9):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        return U, V, W

    # Build the base rank-32 decomposition
    terms = build_strassen_rank32()
    
    # Convert to matrices
    U, V, W = terms_to_matrices(terms)
    
    # Verify
    if not verify_decomp(U, V, W):
        # Fallback to schoolbook rank 36 if something went wrong (should not happen)
        terms36 = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms36.append(({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1}))
        U, V, W = terms_to_matrices(terms36)
    
    return {"U": U, "V": V, "W": W}
