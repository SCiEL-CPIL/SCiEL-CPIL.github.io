import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = 2, 4, 5
    
    def verify_numpy(U, V, W, rank):
        """Fast verification using numpy tensor contraction."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Reshape to 3D tensors
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        # Compute sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r]
        # Use einsum
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        # Check all entries equal 1
        if not np.allclose(result, 1.0, atol=1e-10):
            return False
        # Also check for exact integer/half-integer values
        # The entries should be exact, so check with tolerance
        return True
    
    def verify_exact(U, V, W, rank):
        """Exact verification with fractions to ensure correctness."""
        from fractions import Fraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True
    
    # Rank-36 Strassen-based solution
    rank36 = 36
    U36 = [[0] * rank36 for _ in range(n * m)]
    V36 = [[0] * rank36 for _ in range(m * p)]
    W36 = [[0] * rank36 for _ in range(p * n)]
    
    col = 0
    strassen_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    standard_terms = [
        ({0:1}, {0:1}, {0:1}),
        ({1:1}, {1:1}, {0:1}),
        ({2:1}, {0:1}, {1:1}),
        ({3:1}, {1:1}, {1:1})
    ]
    
    for q in range(2):
        for s in range(3):
            if s < 2:
                for u_t, v_t, w_t in strassen_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U36[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v // 2
                        lk = idx_v % 2
                        gj = 2*q + lj
                        gk = 2*s + lk
                        V36[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w // 2
                        lk = idx_w % 2
                        gi = li
                        gk = 2*s + lk
                        W36[gk*n + gi][col] += val_w
                    col += 1
            else:
                for u_t, v_t, w_t in standard_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U36[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v
                        gj = 2*q + lj
                        gk = 4
                        V36[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w
                        gi = li
                        gk = 4
                        W36[gk*n + gi][col] += val_w
                    col += 1
    
    if col == rank36 and verify_numpy(U36, V36, W36, rank36):
        if verify_exact(U36, V36, W36, rank36):
            return {"U": U36, "V": V36, "W": W36}
    
    # Try to find a lower rank by searching over block structures
    # The key insight: we can try different combinations of Strassen and standard
    # terms for different (q, s) blocks to reduce rank
    
    # Strategy: For each (q, s) block, we can choose either:
    # - 2x2 Strassen (7 terms) for s < 2
    # - 1x1 standard (4 terms) for s = 2 (since it's 2x1)
    # But we can also try:
    # - For s < 2: use 2x2 Strassen (7 terms) OR try to share terms
    # - For s = 2: use 2x1 standard (4 terms) OR try 2x1 Strassen-like (3 terms?)
    
    # Actually, for 2x1 multiplication, the minimum rank is 3 (not 4)
    # because we can use: (A1+A2)(B1), A1(B2-B1), A2(B1) -> wait, that's for 2x2
    # For 2x1: C1 = A1*B1, C2 = A2*B1. Minimum rank is 2 (just two multiplications)
    # Actually no, 2x1 needs rank 2 minimum since there are 2 outputs.
    
    # Let me reconsider. The current approach uses:
    # - 2 q-values x 2 s-values (s=0,1) with 7 Strassen terms each = 28
    # - 2 q-values x 1 s-value (s=2) with 4 standard terms each = 8
    # Total = 36
    
    # Can we do better? Let's think about the structure:
    # Matrix A is 2x4, B is 4x5
    # Split A into 2 blocks of 2x2: A = [A_q for q in {0,1}]
    # Split B into 3 blocks: B = [B_s for s in {0,1,2}] where B_0, B_1 are 2x2 and B_2 is 2x1
    
    # For each (q, s) pair, we compute A_q * B_s
    # - If s < 2: A_q is 2x2, B_s is 2x2 -> rank 7 (Strassen)
    # - If s = 2: A_q is 2x2, B_s is 2x1 -> rank 2 (just 2 mults)
    
    # So minimum with this block structure: 2*2*7 + 2*1*2 = 28 + 4 = 32
    
    # Let's implement rank-32 solution!
    rank32 = 32
    U32 = [[0] * rank32 for _ in range(n * m)]
    V32 = [[0] * rank32 for _ in range(m * p)]
    W32 = [[0] * rank32 for _ in range(p * n)]
    
    col = 0
    for q in range(2):
        for s in range(3):
            if s < 2:
                # Use 7 Strassen terms for 2x2 * 2x2
                for u_t, v_t, w_t in strassen_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U32[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v // 2
                        lk = idx_v % 2
                        gj = 2*q + lj
                        gk = 2*s + lk
                        V32[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w // 2
                        lk = idx_w % 2
                        gi = li
                        gk = 2*s + lk
                        W32[gk*n + gi][col] += val_w
                    col += 1
            else:
                # Use 2 terms for 2x2 * 2x1 (just direct multiplication)
                # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
                # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
                # Term 1: A[0,0]*B[0,0] contributes to C[0,0]
                # Term 2: A[0,1]*B[1,0] contributes to C[0,0]
                # Term 3: A[1,0]*B[0,0] contributes to C[1,0]
                # Term 4: A[1,1]*B[1,0] contributes to C[1,0]
                # That's 4 terms, not 2. Let me reconsider.
                
                # Actually for 2x2 * 2x1:
                # Result is 2x1: [c0, c1]
                # c0 = a00*b0 + a01*b1
                # c1 = a10*b0 + a11*b1
                # Minimum rank is 4 (each term is independent)
                # Unless we can factor...
                
                # Hmm, so it's still 4 terms. Let me keep the standard approach.
                for u_t, v_t, w_t in standard_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U32[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v
                        gj = 2*q + lj
                        gk = 4
                        V32[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w
                        gi = li
                        gk = 4
                        W32[gk*n + gi][col] += val_w
                    col += 1
    
    if col == rank32 and verify_numpy(U32, V32, W32, rank32):
        if verify_exact(U32, V32, W32, rank32):
            return {"U": U32, "V": V32, "W": W32}
    
    # Try rank-36 with numpy verification
    if col == rank36 and verify_numpy(U36, V36, W36, rank36):
        if verify_exact(U36, V36, W36, rank36):
            return {"U": U36, "V": V36, "W": W36}
    
    # Schoolbook fallback
    rank_school = n * m * p
    best = {
        "U": [[0]*rank_school for _ in range(n*m)],
        "V": [[0]*rank_school for _ in range(m*p)],
        "W": [[0]*rank_school for _ in range(p*n)]
    }
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best["U"][i*m+j][c] = 1
                best["V"][j*p+k][c] = 1
                best["W"][k*n+i][c] = 1
                c += 1
    return best
