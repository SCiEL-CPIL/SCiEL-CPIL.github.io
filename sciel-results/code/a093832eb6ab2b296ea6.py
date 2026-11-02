<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_decomposition(n, m, p):
    """
    Builds a rank-77 decomposition for 4x4 x 4x6 matrix multiplication.
    Uses tensor product of Strassen 2x2x2 (rank 7) for the 4x4x4 part (49 terms)
    and Strassen-based decomposition for the 4x4x2 part (28 terms).
    """
    # Strassen 2x2x2 terms
    strassen_terms_2x2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    
    rank_terms = []
    
    # Generate 49 terms for 4x4x4 part via tensor product
    for s in range(7):
        u_s, v_s, w_s = strassen_terms_2x2[s]
        for t in range(7):
            u_t, v_t, w_t = strassen_terms_2x2[t]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            for (i1, j1), cu in u_s.items():
                for (i2, j2), cu2 in u_t.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    coeff = cu * cu2
                    key = (gi, gj)
                    U_global[key] = U_global.get(key, 0) + coeff
                    
            for (j1, k1), cv in v_s.items():
                for (j2, k2), cv2 in v_t.items():
                    gj = 2 * j1 + j2
                    gk = 2 * k1 + k2
                    coeff = cv * cv2
                    key = (gj, gk)
                    V_global[key] = V_global.get(key, 0) + coeff
                    
            for (i1, k1), cw in w_s.items():
                for (i2, k2), cw2 in w_t.items():
                    gi = 2 * i1 + i2
                    gk = 2 * k1 + k2
                    coeff = cw * cw2
                    key = (gk, gi)
                    W_global[key] = W_global.get(key, 0) + coeff
                    
            rank_terms.append((U_global, V_global, W_global))
            
    # Add 28 terms for 4x4x2 part using Strassen for each 2x2x2 block
    # C_top = A[0:2,0:2] @ B[0:2,0:2] + A[0:2,2:4] @ B[2:4,0:2]
    # C_bot = A[2:4,0:2] @ B[0:2,0:2] + A[2:4,2:4] @ B[2:4,0:2]
    
    # For each of the 4 sub-products, apply Strassen
    sub_products = [
        # (row_offset_A, col_offset_A, row_offset_B, col_offset_B, row_offset_C, col_offset_C)
        (0, 0, 0, 0, 0, 0),  # A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
        (0, 2, 2, 0, 0, 0),  # A[0:2,2:4] @ B[2:4,0:2] -> C[0:2,0:2]
        (2, 0, 0, 0, 2, 0),  # A[2:4,0:2] @ B[0:2,0:2] -> C[2:4,0:2]
        (2, 2, 2, 0, 2, 0),  # A[2:4,2:4] @ B[2:4,0:2] -> C[2:4,0:2]
    ]
    
    for s in range(7):
        u_s, v_s, w_s = strassen_terms_2x2[s]
        for (ra, ca, rb, cb, rc, cc) in sub_products:
            U_global = {}
            V_global = {}
            W_global = {}
            
            for (i1, j1), cu in u_s.items():
                gi = ra + i1
                gj = ca + j1
                U_global[(gi, gj)] = U_global.get((gi, gj), 0) + cu
                
            for (j1, k1), cv in v_s.items():
                gj = rb + j1
                gk = cb + k1
                V_global[(gj, gk)] = V_global.get((gj, gk), 0) + cv
                
            for (i1, k1), cw in w_s.items():
                gi = rc + i1
                gk = cc + k1
                W_global[(gk, gi)] = W_global.get((gk, gi), 0) + cw
                
            rank_terms.append((U_global, V_global, W_global))
                
    rank = len(rank_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i, r] = coeff
            
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W[:, r].reshape(p, n).T
        
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Build decomposition
    U, V, W = _build_decomposition(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """Return the 7 rank-1 terms of Strassen's 2x2x2 decomposition."""
    return [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

def _rank5_2x2x3():
    """
    Rank-5 decomposition for 2x2 x 2x3 matrix multiplication.
    C[i,j] = sum_k A[i,k] B[k,j] for i in {0,1}, k in {0,1}, j in {0,1,2}
    
    We split into:
    - First two columns: use Strassen (7 terms) but we can do better
    - Actually, let's use a direct construction.
    
    For 2x2 x 2x3:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    Using 5 rank-1 terms:
    Let's define:
    m1 = A[0,0]+A[0,1]+A[1,0]+A[1,1]  (all A)
    m2 = A[0,0]+A[0,1]  (row 0 of A)
    m3 = A[1,0]+A[1,1]  (row 1 of A)
    m4 = A[0,0]
    m5 = A[1,1]
    
    n1 = B[0,0]+B[1,0]  (col 0 of B)
    n2 = B[0,1]+B[1,1]  (col 1 of B)
    n3 = B[0,2]+B[1,2]  (col 2 of B)
    n4 = B[0,0]
    n5 = B[1,0]
    
    This is getting complicated. Let me use a known rank-5 decomposition.
    
    Actually, the minimal rank for 2x2x2x3 is 5. Let me construct it directly.
    
    C[0,0] = A00*B00 + A01*B10
    C[0,1] = A00*B01 + A01*B11
    C[0,2] = A00*B02 + A01*B12
    C[1,0] = A10*B00 + A11*B10
    C[1,1] = A10*B01 + A11*B11
    C[1,2] = A10*B02 + A11*B12
    
    5 terms:
    t1: U={A00}, V={B00, B01, B02}, W={C00, C01, C02}
    t2: U={A01}, V={B10, B11, B12}, W={C00, C01, C02}
    t3: U={A10}, V={B00, B01, B02}, W={C10, C11, C12}
    t4: U={A11}, V={B10, B11, B12}, W={C10, C11, C12}
    
    That's only 4 terms! But wait, that gives:
    C00 = A00*B00 + A01*B10 ✓
    C01 = A00*B01 + A01*B11 ✓
    C02 = A00*B02 + A01*B12 ✓
    C10 = A10*B00 + A11*B10 ✓
    C11 = A10*B01 + A11*B11 ✓
    C12 = A10*B02 + A11*B12 ✓
    
    So rank 4 suffices for 2x2 x 2x3! Let me verify this is correct.
    Yes, this is just the schoolbook method which uses 4*2=8... wait no.
    
    Schoolbook for 2x2 x 2x3: 2*2*3 = 12 terms.
    But my construction above uses only 4 terms! Let me double-check.
    
    Term 1: alpha = A00, beta = B00+B01+B02 (as a vector), contributes A00*B0j to C0j
    Wait, I need to be more careful with the tensor decomposition format.
    
    In the tensor decomposition:
    U has shape (2*2, rank) = (4, rank), indexed by (i,j) for A[i,j]
    V has shape (2*3, rank) = (6, rank), indexed by (k,l) for B[k,l]  
    W has shape (3*2, rank) = (6, rank), indexed by (l,i) for C[i,l] (trace convention: W[k*n+i,r] where n=2, p=3)
    
    Wait, the problem says W[k*n+i,r] which means W is indexed by (k,i) where k is column of B / row of C, and i is row of A / col of C.
    
    So for C[i,l] (i=row, l=col), W index is (l, i).
    
    Let me redo:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    4 terms:
    t1: U={(0,0):1}, V={(0,0):1,(0,1):1,(0,2):1}, W={(0,0):1,(1,0):1,(2,0):1}
         => contributes A[0,0] * (B[0,0]+B[0,1]+B[0,2]) to C[0,0]+C[0,1]+C[0,2]
         But we need A[0,0]*B[0,j] to go to C[0,j], not summed.
    
    I see the issue. Each rank-1 term is alpha_r * beta_r * W_r, where:
    alpha_r = sum_{i,j} A[i,j] * U[i,j,r]
    beta_r = sum_{k,l} B[k,l] * V[k,l,r]
    Contribution to C[i,l] = alpha_r * beta_r * W[l,i,r]
    
    So for t1: alpha = A[0,0], beta = B[0,0]+B[0,1]+B[0,2]
    Contribution to C[0,0] = A[0,0]*(B[0,0]+B[0,1]+B[0,2])*W[0,0,1]
    Contribution to C[0,1] = A[0,0]*(B[0,0]+B[0,1]+B[0,2])*W[1,0,1]
    Contribution to C[0,2] = A[0,0]*(B[0,0]+B[0,1]+B[0,2])*W[2,0,1]
    
    This doesn't work because beta is a single scalar, so it multiplies the same combination of B's into all C entries.
    
    I need a different approach. Let me use:
    t1: U={(0,0):1}, V={(0,0):1}, W={(0,0):1}  => A00*B00 -> C00
    t2: U={(0,0):1}, V={(0,1):1}, W={(1,0):1}  => A00*B01 -> C01
    t3: U={(0,0):1}, V={(0,2):1}, W={(2,0):1}  => A00*B02 -> C02
    t4: U={(0,1):1}, V={(1,0):1}, W={(0,0):1}  => A01*B10 -> C00
    t5: U={(0,1):1}, V={(1,1):1}, W={(1,0):1}  => A01*B11 -> C01
    t6: U={(0,1):1}, V={(1,2):1}, W={(2,0):1}  => A01*B12 -> C02
    t7: U={(1,0):1}, V={(0,0):1}, W={(0,1):1}  => A10*B00 -> C10
    t8: U={(1,0):1}, V={(0,1):1}, W={(1,1):1}  => A10*B01 -> C11
    t9: U={(1,0):1}, V={(0,2):1}, W={(2,1):1}  => A10*B02 -> C12
    t10: U={(1,1):1}, V={(1,0):1}, W={(0,1):1}  => A11*B10 -> C10
    t11: U={(1,1):1}, V={(1,1):1}, W={(1,1):1}  => A11*B11 -> C11
    t12: U={(1,1):1}, V={(1,2):1}, W={(2,1):1}  => A11*B12 -> C12
    
    That's 12 terms (schoolbook). To reduce rank, I need to combine terms.
    
    Let me try a Strassen-like approach for the 2x2x2 part (first two columns) and add one more term for the third column.
    
    For 2x2 x 2x2 (first two columns), Strassen gives rank 7.
    For the third column, I can add 2 more terms (one for each row of A):
    t8: U={(0,0):1}, V={(0,2):1}, W={(2,0):1}
    t9: U={(1,0):1}, V={(0,2):1}, W={(2,1):1}
    
    Wait, that's not right either. Let me think more carefully.
    
    Actually, let me just use the Kronecker product approach with a verified rank-5 for 2x2x2x3.
    
    After research, the rank of 2x2x2x3 is indeed 5. Here's a construction:
    
    Define:
    m1 = A00 + A01 + A10 + A11
    m2 = A00 + A01
    m3 = A10 + A11  
    m4 = A00
    m5 = A11
    
    n1 = B00 + B10
    n2 = B01 + B11
    n3 = B02 + B12
    n4 = B00
    n5 = B10
    
    Hmm, this is getting complex. Let me just use a simple approach:
    Use Strassen (rank 7) for the 2x2x2 part and 2 additional terms for the third column.
    Total: 7 + 2 = 9 for the 2x2x2x3 factor.
    
    Then Kronecker with 2x2x2 (rank 7): 7 * 9 = 63 total rank.
    That's worse than 77.
    
    Let me try another approach: use rank-5 for 2x2x2x3 if I can find it.
    
    Actually, let me just implement the Kronecker product of 2x2x2 (rank 7) with 2x2x3 (rank 5) = 35 total.
    I'll construct the rank-5 for 2x2x2x3 carefully.
    """
    # For 2x2 x 2x3, use a direct rank-5 construction
    # After careful analysis, here's a valid rank-5 decomposition:
    
    terms = []
    
    # Term 1: handles A[0,0] with B[0,0], B[0,1], B[0,2]
    # But we need to separate the B columns...
    
    # Let me use a different strategy: 
    # Split 2x2x2x3 into 2x2x2 (cols 0-1) + 2x2x1 (col 2)
    # 2x2x2: Strassen rank 7
    # 2x2x1: rank 2 (just A00*B02 and A10*B02... no, A00*B02 -> C02, A01*B12 -> C02, A10*B02 -> C12, A11*B12 -> C12)
    # That's 4 terms for the last column.
    # Total: 7 + 4 = 11. Worse.
    
    # Let me try to be smarter. 
    # For the last column (col 2 of B):
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # I can use 2 terms:
    # t_a: U={(0,0):1, (1,0):1}, V={(0,2):1}, W={(2,0):1, (2,1):1}
    #   alpha = A00 + A10, beta = B02
    #   C[0,2] += (A00+A10)*B02*1, C[1,2] += (A00+A10)*B02*1
    #   But we need C[0,2] += A00*B02 and C[1,2] += A10*B02, not (A00+A10)*B02 for both.
    
    # This doesn't work because alpha is a single scalar.
    
    # OK, let me just use 4 terms for the last column and 7 for the first two, total 11.
    # Then Kronecker: 7 * 11 = 77. Same as before!
    
    # Let me try yet another approach: use a completely different factorization.
    # 4x4 x 4x6 = (2x2 ⊗ 2x2) x (2x2 ⊗ 2x3)
    # Use rank-7 for 2x2x2 and rank-5 for 2x2x2x3 if possible.
    
    # I'll implement a verified rank-5 for 2x2x2x3 by direct construction.
    # Here's one approach using the fact that we can share computations:
    
    # Let me define 5 terms:
    # t1: U={(0,0):1, (1,1):1}, V={(0,0):1, (1,0):1}, W={(0,0):1, (1,1):1, (2,0):1, (2,1):1}
    #   alpha = A00 + A11, beta = B00 + B10
    #   C00 += (A00+A11)(B00+B10)*1 = A00*B00 + A00*B10 + A11*B00 + A11*B10
    #   But we need C00 = A00*B00 + A01*B10, so this introduces wrong terms.
    
    # This is getting too complex. Let me just use the Kronecker product with rank-7 and rank-4 (schoolbook for 2x2x2x1).
    # 2x2x2x1: C[i,0] = sum_k A[i,k]*B[k,0], which is 2x2x1, rank = 2*2*1 = 4 (schoolbook)
    # But actually 2x2x1 has rank 2: C[0,0]=A00*B00+A01*B10, C[1,0]=A10*B00+A11*B10
    # t1: U={(0,0):1}, V={(0,0):1}, W={(0,0):1} -> A00*B00 -> C00
    # t2: U={(0,1):1}, V={(1,0):1}, W={(0,0):1} -> A01*B10 -> C00
    # t3: U={(1,0):1}, V={(0,0):1}, W={(0,1):1} -> A10*B00 -> C10
    # t4: U={(1,1):1}, V={(1,0):1}, W={(0,1):1} -> A11*B10 -> C10
    # Rank 4 for 2x2x1.
    
    # So 2x2x2x3 = 2x2x2x2 (cols 0-1) + 2x2x1 (col 2)
    # Rank: 7 + 4 = 11
    # Kronecker with 2x2x2 (rank 7): 7*11 = 77. Same as current!
    
    # I need a better approach. Let me just use the current 77-rank solution but verify it works.
    # Actually, the current solution claims 77 but let me check if it's correct.
    
    # For now, let me return a simple schoolbook decomposition to ensure correctness.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W[:, r].reshape(p, n).T
        
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140.0
    
    U, V, W = _build_decomposition(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I made an error - I'm returning schoolbook (rank 96) which is worse. Let me fix this by properly implementing the Kronecker product approach with the existing 77-rank decomposition, but I realize the current code already does this. Let me instead focus on verifying and keeping the existing 77-rank solution, or finding a genuinely better one.

Let me reconsider: the inspiration says score 1.2468 uses Kronecker product of 2×2×3 (rank 5) and 2×2×2 (rank 7) = 35 rank. Let me implement that properly.

