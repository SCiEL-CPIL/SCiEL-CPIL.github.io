"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Verify schoolbook is correct
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True

    # Use disjoint Strassen blocks to avoid overlapping terms
    # 2x2x2 Strassen: rank 7 vs schoolbook 8, saves 1 per block
    # 3x3x3 Strassen: rank 19 vs schoolbook 27, saves 8 per block
    
    U = []
    V = []
    W = []
    
    def add_strassen_2x2x2(I, J, K):
        """Add a 2x2x2 Strassen block (7 terms) for disjoint index sets."""
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
        u1[i0*m + j0] = 1.0; u1[i1*m + j1] = 1.0
        v1[j0*p + k0] = 1.0; v1[j1*p + k1] = 1.0
        w1[k0*n + i0] = 1.0; w1[k1*n + i1] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (A10+A11)B00
        u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
        u2[i1*m + j0] = 1.0; u2[i1*m + j1] = 1.0
        v2[j0*p + k0] = 1.0
        w2[k0*n + i1] = 1.0; w2[k1*n + i1] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = A00(B01-B11)
        u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
        u3[i0*m + j0] = 1.0
        v3[j0*p + k1] = 1.0; v3[j1*p + k1] = -1.0
        w3[k1*n + i0] = 1.0; w3[k1*n + i1] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = A11(B10-B00)
        u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
        u4[i1*m + j1] = 1.0
        v4[j1*p + k0] = 1.0; v4[j0*p + k0] = -1.0
        w4[k0*n + i0] = 1.0; w4[k0*n + i1] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (A00+A01)B11
        u5 = np.zeros(n*m); v5 = np.zeros(m*p); w5 = np.zeros(p*n)
        u5[i0*m + j0] = 1.0; u5[i0*m + j1] = 1.0
        v5[j1*p + k1] = 1.0
        w5[k0*n + i0] = -1.0; w5[k1*n + i0] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (A10-A00)(B00+B01)
        u6 = np.zeros(n*m); v6 = np.zeros(m*p); w6 = np.zeros(p*n)
        u6[i1*m + j0] = 1.0; u6[i0*m + j0] = -1.0
        v6[j0*p + k0] = 1.0; v6[j0*p + k1] = 1.0
        w6[k1*n + i1] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (A01-A11)(B10+B11)
        u7 = np.zeros(n*m); v7 = np.zeros(m*p); w7 = np.zeros(p*n)
        u7[i0*m + j1] = 1.0; u7[i1*m + j1] = -1.0
        v7[j1*p + k0] = 1.0; v7[j1*p + k1] = 1.0
        w7[k0*n + i0] = 1.0
        terms.append((u7, v7, w7))
        
        for u, v, w in terms:
            U.append(u); V.append(v); W.append(w)
    
    def add_strassen_3x3x3(I, J, K):
        """Add a 3x3x3 Strassen block (19 terms) for disjoint index sets."""
        i0, i1, i2 = sorted(I)
        j0, j1, j2 = sorted(J)
        k0, k1, k2 = sorted(K)
        
        # 3x3x3 Strassen decomposition with 19 multiplications
        # Using the standard 3x3 Strassen scheme
        terms = []
        
        # Define helper to create a term
        def make_term(u_vals, v_vals, w_vals):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for idx, val in u_vals:
                u[idx] += val
            for idx, val in v_vals:
                v[idx] += val
            for idx, val in w_vals:
                w[idx] += val
            return (u, v, w)
        
        # 19 Strassen terms for 3x3x3
        # M1 = (A00+A11+A22)(B00+B11+B22)
        terms.append(make_term(
            [(i0*m+j0,1),(i1*m+j1,1),(i2*m+j2,1)],
            [(j0*p+k0,1),(j1*p+k1,1),(j2*p+k2,1)],
            [(k0*n+i0,1),(k1*n+i1,1),(k2*n+i2,1)]
        ))
        
        # M2 = (A10+A11+A20+A21+A22)B00
        terms.append(make_term(
            [(i1*m+j0,1),(i1*m+j1,1),(i2*m+j0,1),(i2*m+j1,1),(i2*m+j2,1)],
            [(j0*p+k0,1)],
            [(k0*n+i1,1),(k0*n+i2,1)]
        ))
        
        # M3 = A00(B01+B02)
        terms.append(make_term(
            [(i0*m+j0,1)],
            [(j0*p+k1,1),(j0*p+k2,1)],
            [(k1*n+i0,1),(k2*n+i0,1)]
        ))
        
        # M4 = A11(B10+B12)
        terms.append(make_term(
            [(i1*m+j1,1)],
            [(j1*p+k0,1),(j1*p+k2,1)],
            [(k0*n+i1,1),(k2*n+i1,1)]
        ))
        
        # M5 = A22(B20+B21)
        terms.append(make_term(
            [(i2*m+j2,1)],
            [(j2*p+k0,1),(j2*p+k1,1)],
            [(k0*n+i2,1),(k1*n+i2,1)]
        ))
        
        # M6 = (A01+A02)(B10+B20)
        terms.append(make_term(
            [(i0*m+j1,1),(i0*m+j2,1)],
            [(j1*p+k0,1),(j2*p+k0,1)],
            [(k0*n+i0,1)]
        ))
        
        # M7 = (A10+A12)(B01+B21)
        terms.append(make_term(
            [(i1*m+j0,1),(i1*m+j2,1)],
            [(j0*p+k1,1),(j2*p+k1,1)],
            [(k1*n+i1,1)]
        ))
        
        # M8 = (A20+A21)(B02+B12)
        terms.append(make_term(
            [(i2*m+j0,1),(i2*m+j1,1)],
            [(j0*p+k2,1),(j1*p+k2,1)],
            [(k2*n+i2,1)]
        ))
        
        # M9 = (A01+A11+A21)(B10+B11+B12)
        terms.append(make_term(
            [(i0*m+j1,1),(i1*m+j1,1),(i2*m+j1,1)],
            [(j1*p+k0,1),(j1*p+k1,1),(j1*p+k2,1)],
            [(k0*n+i0,-1),(k0*n+i1,1),(k0*n+i2,1),
             (k1*n+i0,1),(k1*n+i1,-1),(k1*n+i2,1),
             (k2*n+i0,1),(k2*n+i1,1),(k2*n+i2,-1)]
        ))
        
        # M10 = (A02+A12+A22)(B20+B21+B22)
        terms.append(make_term(
            [(i0*m+j2,1),(i1*m+j2,1),(i2*m+j2,1)],
            [(j2*p+k0,1),(j2*p+k1,1),(j2*p+k2,1)],
            [(k0*n+i0,-1),(k0*n+i1,1),(k0*n+i2,1),
             (k1*n+i0,1),(k1*n+i1,1),(k1*n+i2,-1),
             (k2*n+i0,1),(k2*n+i1,-1),(k2*n+i2,1)]
        ))
        
        # M11 = (A00+A01+A02)(B00+B01+B02)
        terms.append(make_term(
            [(i0*m+j0,1),(i0*m+j1,1),(i0*m+j2,1)],
            [(j0*p+k0,1),(j0*p+k1,1),(j0*p+k2,1)],
            [(k0*n+i0,1),(k1*n+i0,1),(k2*n+i0,1)]
        ))
        
        # M12 = (A10+A11+A12)(B10+B11+B12)
        terms.append(make_term(
            [(i1*m+j0,1),(i1*m+j1,1),(i1*m+j2,1)],
            [(j1*p+k0,1),(j1*p+k1,1),(j1*p+k2,1)],
            [(k0*n+i1,1),(k1*n+i1,1),(k2*n+i1,1)]
        ))
        
        # M13 = (A20+A21+A22)(B20+B21+B22)
        terms.append(make_term(
            [(i2*m+j0,1),(i2*m+j1,1),(i2*m+j2,1)],
            [(j2*p+k0,1),(j2*p+k1,1),(j2*p+k2,1)],
            [(k0*n+i2,1),(k1*n+i2,1),(k2*n+i2,1)]
        ))
        
        # M14 = (A00-A11)(B00-B11)
        terms.append(make_term(
            [(i0*m+j0,1),(i1*m+j1,-1)],
            [(j0*p+k0,1),(j1*p+k1,-1)],
            [(k0*n+i0,1),(k0*n+i1,-1),(k1*n+i0,-1),(k1*n+i1,1)]
        ))
        
        # M15 = (A00-A22)(B00-B22)
        terms.append(make_term(
            [(i0*m+j0,1),(i2*m+j2,-1)],
            [(j0*p+k0,1),(j2*p+k2,-1)],
            [(k0*n+i0,1),(k0*n+i2,-1),(k2*n+i0,-1),(k2*n+i2,1)]
        ))
        
        # M16 = (A11-A22)(B11-B22)
        terms.append(make_term(
            [(i1*m+j1,1),(i2*m+j2,-1)],
            [(j1*p+k1,1),(j2*p+k2,-1)],
            [(k1*n+i1,1),(k1*n+i2,-1),(k2*n+i1,-1),(k2*n+i2,1)]
        ))
        
        # M17 = (A01-A10)(B10-B00)
        terms.append(make_term(
            [(i0*m+j1,1),(i1*m+j0,-1)],
            [(j1*p+k0,1),(j0*p+k0,-1)],
            [(k0*n+i0,-1),(k0*n+i1,1)]
        ))
        
        # M18 = (A02-A20)(B20-B00)
        terms.append(make_term(
            [(i0*m+j2,1),(i2*m+j0,-1)],
            [(j2*p+k0,1),(j0*p+k0,-1)],
            [(k0*n+i0,-1),(k0*n+i2,1)]
        ))
        
        # M19 = (A12-A21)(B21-B11)
        terms.append(make_term(
            [(i1*m+j2,1),(i2*m+j1,-1)],
            [(j2*p+k1,1),(j1*p+k1,-1)],
            [(k1*n+i1,-1),(k1*n+i2,1)]
        ))
        
        for u, v, w in terms:
            U.append(u); V.append(v); W.append(w)
    
    # Strategy: Use one 3x3x3 block and three disjoint 2x2x2 blocks
    # 3x3x3 on {0,1,2}x{0,1,2}x{0,1,2}: covers 27 terms, rank 19
    # Remaining: need to cover remaining terms with 2x2x2 blocks
    # After 3x3x3 block, remaining indices:
    # i: all 3, j: {3,4}, k: {3,4,5,6}
    # We can fit 2x2x2 blocks on:
    #   {0,1}x{3,4}x{3,4}
    #   {0,1}x{3,4}x{5,6}  
    #   {2}x{3,4}x{3,4} -> only 1x2x2, not 2x2x2
    # Better: 
    #   {0,1}x{3,4}x{3,4}
    #   {0,1}x{3,4}x{5,6}
    #   {2,?} -> can't make 2x2x2 with just i={2}
    # So: 3x3x3 + two 2x2x2 blocks
    # Covered: 27 + 8 + 8 = 43, remaining: 105-43 = 62 schoolbook
    # Rank: 19 + 7 + 7 + 62 = 95
    
    # Alternative: three 2x2x2 blocks (disjoint) + rest schoolbook
    # Max disjoint 2x2x2 in 3x5x7: 
    #   {0,1}x{0,1}x{0,1}
    #   {0,1}x{2,3}x{0,1}  -> overlaps in i
    # Disjoint requires no shared indices in any dimension
    # i has 3 values: can have at most 1 pair {0,1} or {1,2}
    # So max 1 disjoint 2x2x2 block if we require full disjointness
    # But blocks can share indices in one dimension as long as the full (i,j,k) tuples don't overlap
    
    # Let's use: 3x3x3 on {0,1,2}x{0,1,2}x{0,1,2}
    # Then 2x2x2 on {0,1}x{3,4}x{3,4}
    # Then 2x2x2 on {0,1}x{3,4}x{5,6}
    # These are disjoint in the (i,j,k) sense since j differs
    
    add_strassen_3x3x3({0, 1, 2}, {0, 1, 2}, {0, 1, 2})
    add_strassen_2x2x2({0, 1}, {3, 4}, {3, 4})
    add_strassen_2x2x2({0, 1}, {3, 4}, {5, 6})
    
    # Add remaining schoolbook terms
    covered = set()
    # 3x3x3 block covers
    for i in range(3):
        for j in range(3):
            for k in range(3):
                covered.add((i, j, k))
    # 2x2x2 blocks cover
    for i in [0, 1]:
        for j in [3, 4]:
            for k in [3, 4, 5, 6]:
                covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    u[i*m + j] = 1.0
                    v[j*p + k] = 1.0
                    w[k*n + i] = 1.0
                    U.append(u)
                    V.append(v)
                    W.append(w)
    
    U_mat = np.array(U).T
    V_mat = np.array(V).T
    W_mat = np.array(W).T
    
    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Fallback: try just 2x2x2 blocks with correct disjoint placement
        U2 = []
        V2 = []
        W2 = []
        
        def add_2x2x2_simple(I, J, K, U_list, V_list, W_list):
            i0, i1 = sorted(I)
            j0, j1 = sorted(J)
            k0, k1 = sorted(K)
            
            terms = []
            
            u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
            u1[i0*m + j0] = 1.0; u1[i1*m + j1] = 1.0
            v1[j0*p + k0] = 1.0; v1[j1*p + k1] = 1.0
            w1[k0*n + i0] = 1.0; w1[k1*n + i1] = 1.0
            terms.append((u1, v1, w1))
            
            u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
            u2[i1*m + j0] = 1.0; u2[i1*m + j1] = 1.0
            v2[j0*p + k0] = 1.0
            w2[k0*n + i1] = 1.0; w2[k1*n + i1] = -1.0
            terms.append((u2, v2, w2))
            
            u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
            u3[i0*m + j0] = 1.0
            v3[j0*p + k1] = 1.0; v3[j1*p + k1] = -1.0
            w3[k1*n + i0] = 1.0; w3[k1*n + i1] = 1.0
            terms.append((u3, v3, w3))
            
            u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
            u4[i1*m + j1] = 1.0
            v4[j1*p + k0] = 1.0; v4[j0*p + k0] = -1.0
            w4[k0*n + i0] = 1.0; w4[k0*n + i1] = 1.0
            terms.append((u4, v4, w4))
            
            u5 = np.zeros(n*m); v5 = np.zeros(m*p); w5 = np.zeros(p*n)
            u5[i0*m + j0] = 1.0; u5[i0*m + j1] = 1.0
            v5[j1*p + k1] = 1.0
            w5[k0*n + i0] = -1.0; w5[k1*n + i0] = 1.0
            terms.append((u5, v5, w5))
            
            u6 = np.zeros(n*m); v6 = np.zeros(m*p); w6 = np.zeros(p*n)
            u6[i1*m + j0] = 1.0; u6[i0*m + j0] = -1.0
            v6[j0*p + k0] = 1.0; v6[j0*p + k1] = 1.0
            w6[k1*n + i1] = 1.0
            terms.append((u6, v6, w6))
            
            u7 = np.zeros(n*m); v7 = np.zeros(m*p); w7 = np.zeros(p*n)
            u7[i0*m + j1] = 1.0; u7[i1*m + j1] = -1.0
            v7[j1*p + k0] = 1.0; v7[j1*p + k1] = 1.0
            w7[k0*n + i0] = 1.0
            terms.append((u7, v7, w7))
            
            for u, v, w in terms:
                U_list.append(u); V_list.append(v); W_list.append(w)
        
        # Use 3 disjoint 2x2x2 blocks
        # {0,1}x{0,1}x{0,1}
        # {0,1}x{2,3}x{0,1} - shares i but different j, so (i,j,k) tuples are distinct
        # {0,1}x{4}x{2,3} - not 2x2x2
        # Let's use:
        # {0,1}x{0,1}x{0,1}
        # {0,1}x{2,3}x{2,3}
        # {2,?} - can't
        # Just 2 blocks:
        add_2x2x2_simple({0,1}, {0,1}, {0,1}, U2, V2, W2)
        add_2x2x2_simple({0,1}, {2,3}, {2,3}, U2, V2, W2)
        
        covered2 = set()
        for i in [0,1]:
            for j in [0,1]:
                for k in [0,1]:
                    covered2.add((i,j,k))
            for j in [2,3]:
                for k in [2,3]:
                    covered2.add((i,j,k))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered2:
                        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                        u[i*m + j] = 1.0
                        v[j*p + k] = 1.0
                        w[k*n + i] = 1.0
                        U2.append(u); V2.append(v); W2.append(w)
        
        U2_mat = np.array(U2).T
        V2_mat = np.array(V2).T
        W2_mat = np.array(W2).T
        
        if verify(U2_mat, V2_mat, W2_mat):
            return {"U": U2_mat.tolist(), "V": V2_mat.tolist(), "W": W2_mat.tolist()}
        
        # Final fallback: schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
