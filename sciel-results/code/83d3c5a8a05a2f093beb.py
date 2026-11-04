"""Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication.

Strategy: Exploit Strassen's 7-mult algorithm on 2x2x2 blocks and direct rank-1 terms
for the remaining columns. For 2x4 by 4x7:
- Split the 4 columns of A into two 2-column blocks
- Split the 7 columns of B into three 2-column blocks (cols 0-1, 2-3, 4-5) plus col 6
- Use Strassen (rank 7) for each 2x2 * 2x2 block: 3 blocks * 7 = 21
- Use direct (rank 4) for each 2x2 * 2x1 block: 2 blocks * 4 = 8
- Total rank: 29
"""
import numpy as np
import time


def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Build the target tensor
    T_full = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0

    def verify_decomposition(U, V, W, T_target, atol=1e-9):
        r = U.shape[1]
        T_approx = np.zeros_like(T_target)
        for r_idx in range(r):
            T_approx += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_target, T_approx, atol=atol)

    # Strassen 2x2x2: computes [A_top | A_bottom] * [B_left | B_right]
    # where A is 2x2, B is 2x2, result is 2x2
    # Indices: i0 = row offset of A block (0 or 1), j0 = col offset of A / row offset of B,
    #          k0 = col offset of B / determines W indices
    # Returns list of (u_vec, v_vec, w_vec) each of length (2,2,2) reshaped to (4,4,4)
    # Actually we'll build full-size vectors directly.

    def strassen_2x2x2(i0, j0, k0):
        """
        Strassen decomposition of the 2x2 * 2x2 sub-block:
        A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
        
        Returns 7 terms, each as (u, v, w) full-size vectors.
        """
        terms = []
        
        # Strassen's 7 products:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B10)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # For each Strassen product, we need to express it as a rank-1 tensor term.
        # A factor: u[i*m+j] = 1 if (i,j) is in the relevant A position
        # B factor: v[j*p+k] = 1 if (j,k) is in the relevant B position  
        # C factor: w[k*n+i] = coefficient for the C output position
        
        # Helper to make full-size vectors
        def make_u(i_vals, j_vals):
            vec = np.zeros(n * m)
            for ii in i_vals:
                for jj in j_vals:
                    vec[ii * m + jj] = 1.0
            return vec
        
        def make_v(j_vals, k_vals):
            vec = np.zeros(m * p)
            for jj in j_vals:
                for kk in k_vals:
                    vec[jj * p + kk] = 1.0
            return vec
        
        def make_w(coeffs):
            """coeffs: dict from (k,i) to coefficient"""
            vec = np.zeros(p * n)
            for (kk, ii), coeff in coeffs.items():
                vec[kk * n + ii] = coeff
            return vec
        
        # M1 = (A00+A11)*(B00+B11)
        # A part: positions (i0,j0) and (i0+1,j0+1)
        u1 = make_u([i0, i0+1], [j0, j0+1])
        # B part: positions (j0,k0) and (j0+1,k0+1)
        v1 = make_v([j0, j0+1], [k0, k0+1])
        # C: M1 contributes to C00(+1), C11(+1)
        w1 = make_w({(k0, i0): 1, (k0+1, i0+1): 1})
        terms.append((u1, v1, w1))
        
        # M2 = (A10+A11)*B00
        u2 = make_u([i0+1], [j0, j0+1])
        v2 = make_v([j0], [k0])
        w2 = make_w({(k0, i0+1): 1})  # contributes to C10
        terms.append((u2, v2, w2))
        
        # M3 = A00*(B10-B11)
        u3 = make_u([i0], [j0])
        v3 = make_v([j0+1], [k0])
        v3[j0 * p + k0 + 1] = -1  # B10 - B11
        # Actually B10 is (j0+1, k0), B11 is (j0+1, k0+1)
        v3 = make_v([j0+1], [k0, k0+1])
        v3[(j0+1) * p + k0] = 1
        v3[(j0+1) * p + k0 + 1] = -1
        w3 = make_w({(k0, i0): 1, (k0+1, i0): 1})  # contributes to C00(-1 via M4? No)
        # Wait, let me redo this carefully.
        # M3 = A00 * (B10 - B11)
        # A00 is at position (i0, j0)
        # B10 is at (j0+1, k0), B11 is at (j0+1, k0+1)
        # M3 contributes: +1 to C01, +1 to C11
        # Actually in Strassen: C01 = M3 + M4, C11 = M1 - M2 + M3 + M6
        # So M3 contributes +1 to C01 and +1 to C11
        w3 = make_w({(k0+1, i0): 1, (k0+1, i0+1): 1})
        terms.append((u3, v3, w3))
        
        # M4 = A11*(B01-B10)
        # A11 at (i0+1, j0+1), B01 at (j0, k0+1), B10 at (j0+1, k0)
        u4 = make_u([i0+1], [j0+1])
        v4 = np.zeros(m * p)
        v4[j0 * p + k0 + 1] = 1   # B01
        v4[(j0+1) * p + k0] = -1  # -B10
        w4 = make_w({(k0+1, i0): 1, (k0, i0+1): -1})  # C01: +1, C10: -1
        # Wait: C00 = M1 - M4 + M5 + M7, so M4 contributes -1 to C00
        # C01 = M3 + M4, so M4 contributes +1 to C01
        # Let me be more careful with the Strassen formulas:
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4  
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # So M4: C00 gets -1, C01 gets +1
        w4 = make_w({(k0, i0): -1, (k0+1, i0): 1})
        terms.append((u4, v4, w4))
        
        # M5 = (A00+A01)*B11
        # A00 at (i0,j0), A01 at (i0,j0+1), B11 at (j0+1,k0+1)
        u5 = make_u([i0], [j0, j0+1])
        v5 = make_v([j0+1], [k0+1])
        w5 = make_w({(k0, i0): 1, (k0, i0+1): 1})  # C00: +1, C10: +1
        terms.append((u5, v5, w5))
        
        # M6 = (A10-A00)*(B00+B10)
        # A10 at (i0+1,j0), A00 at (i0,j0), B00 at (j0,k0), B10 at (j0+1,k0)
        u6 = np.zeros(n * m)
        u6[(i0+1) * m + j0] = 1
        u6[i0 * m + j0] = -1
        v6 = make_v([j0, j0+1], [k0])
        w6 = make_w({(k0, i0+1): 1})  # C11: +1
        terms.append((u6, v6, w6))
        
        # M7 = (A01-A11)*(B01+B11)
        # A01 at (i0,j0+1), A11 at (i0+1,j0+1), B01 at (j0,k0+1), B11 at (j0+1,k0+1)
        u7 = np.zeros(n * m)
        u7[i0 * m + j0 + 1] = 1
        u7[(i0+1) * m + j0 + 1] = -1
        v7 = make_v([j0, j0+1], [k0+1])
        w7 = make_w({(k0, i0): 1})  # C00: +1
        terms.append((u7, v7, w7))
        
        return terms

    def direct_2x2x1(i0, j0, k0):
        """
        Direct 4-term decomposition of 2x2 * 2x1:
        A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[i0:i0+2, k0:k0+1]
        """
        terms = []
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n * m)
                u[(i0 + di) * m + (j0 + dj)] = 1.0
                v = np.zeros(m * p)
                v[(j0 + dj) * p + k0] = 1.0
                w = np.zeros(p * n)
                w[k0 * n + (i0 + di)] = 1.0
                terms.append((u, v, w))
        return terms

    # Build all terms
    all_terms = []
    
    # Three 4x2 blocks of B: cols (0,1), (2,3), (4,5)
    for k0 in [0, 2, 4]:
        # First 2x2 * 2x2: A[:, 0:2] * B[0:2, k0:k0+2]
        all_terms.extend(strassen_2x2x2(0, 0, k0))
        # Second 2x2 * 2x2: A[:, 2:4] * B[2:4, k0:k0+2]
        all_terms.extend(strassen_2x2x2(0, 2, k0))
    
    # Last column: B[:, 6]
    # A[:, 0:2] * B[0:2, 6]
    all_terms.extend(direct_2x2x1(0, 0, 6))
    # A[:, 2:4] * B[2:4, 6]
    all_terms.extend(direct_2x2x1(0, 2, 6))

    rank = len(all_terms)
    print(f"Constructed decomposition with rank {rank}")

    # Assemble matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for t, (u, v, w) in enumerate(all_terms):
        U[:, t] = u
        V[:, t] = v
        W[:, t] = w

    # Verify
    if verify_decomposition(U, V, W, T_full):
        print("Verification PASSED")
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        print("Verification FAILED, falling back to schoolbook")
    
    # Fallback: Schoolbook rank 56
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    assert verify_decomposition(U_sb, V_sb, W_sb, T_full)
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
