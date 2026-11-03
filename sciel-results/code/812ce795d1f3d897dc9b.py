import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Build rank-35 decomposition using Kronecker product of Strassen (2x2x2, rank 7)
    # and a 2x1x5 linear convolution block (rank 5)
    # 
    # The 2x4x5 tensor T[i,j,k] = sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # Strategy: Decompose as Kronecker product
    # - Strassen 2x2x2: indices (i, j_mod2, k_mod2) with 7 rank-1 terms
    # - 2x1x5 block: handles the remaining dimensions with rank 5
    #
    # Actually, let's use a direct construction.
    # 
    # For 2x2x5 matmul, the rank is 10. We can construct it as:
    # Use Strassen for the 2x2 part (7 mults) giving 7 intermediate values,
    # each of which is a 1x5 vector that needs to be "multiplied" with B's rows.
    # But each Strassen mult gives a scalar, and we need to combine with 5 B values.
    #
    # Alternative: The 2x2xn tensor has rank 7 + (n-2) for n >= 2.
    # For n=5: rank = 7 + 3 = 10.
    #
    # The 10 rank-1 terms for 2x2x5:
    # Terms 0-6: Strassen-like for the first 2 k-values, adapted
    # Terms 7-9: Direct terms for k=2,3,4
    #
    # For 2x4x5: split j into two pairs (j0, j1) and (j2, j3)
    # Each pair gives a 2x2x5 sub-tensor of rank 10
    # Total rank = 2 * 10 = 20? But A entries are shared...
    #
    # Actually, the A matrix has entries A[i,j] for i in {0,1}, j in {0,1,2,3}
    # The B matrix has entries B[j,k] for j in {0,1,2,3}, k in {0,1,2,3,4}
    # 
    # For pair 0 (j=0,1): uses A[:,0:2] and B[0:2,:]
    # For pair 1 (j=2,3): uses A[:,2:4] and B[2:4,:]
    # These are independent, so we can just concatenate!
    # Total rank = 10 + 10 = 20
    #
    # But wait, that gives rank 20, which is way below 36. Let me verify this is correct.
    #
    # The tensor is T[i,j,k] = 1 (the matmul tensor).
    # For pair 0: T[i, j, k] for j in {0,1}
    # For pair 1: T[i, j, k] for j in {2,3}
    #
    # If I can decompose each 2x2x5 sub-tensor with rank 10, total rank is 20.
    #
    # Let me construct the 2x2x5 rank-10 decomposition.
    
    # First, let me build the 2x2x5 rank-10 decomposition
    # Then use it twice (for j pairs 0 and 1)
    
    def build_2x2x5_rank10():
        """Build rank-10 decomposition for 2x2x5 matmul tensor.
        Returns U (2x2=4 rows x 10 cols), V (2x5=10 rows x 10 cols), W (5x2=10 rows x 10 cols)
        such that for all i in {0,1}, j in {0,1}, k in {0,1,2,3,4}:
        sum_r U[i*2+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1
        """
        n_sub, m_sub, p_sub = 2, 2, 5
        rank = 10
        U = np.zeros((n_sub * m_sub, rank))
        V = np.zeros((m_sub * p_sub, rank))
        W = np.zeros((p_sub * n_sub, rank))
        
        # Strassen 2x2x2 terms for k=0,1 (adapted to 2x2x5)
        # M1 = (A00+A11)(B00+B11) -> contributes to C[0,0] and C[1,1]
        # M2 = (A10+A11)B00 -> contributes to C[1,0] and C[1,1]  
        # M3 = A00(B01-B11) -> contributes to C[0,1] and C[1,1]
        # M4 = A11(B10-B00) -> contributes to C[0,0] and C[1,0]
        # M5 = (A00+A01)B11 -> contributes to C[0,1] and C[1,1]
        # M6 = (A10-A00)(B00+B01) -> contributes to C[1,1]
        # M7 = (A01-A11)(B10+B11) -> contributes to C[0,0]
        
        # For 2x2x5, we need to handle k=0,1,2,3,4
        # The Strassen terms naturally handle k=0,1 (the 2x2x2 part)
        # For k=2,3,4, we need additional terms
        
        # Actually, the standard 2x2xn decomposition:
        # Use 7 Strassen terms that work for the first 2 columns of B
        # Then add (n-2) direct terms for the remaining columns
        
        # Strassen terms (r=0 to 6): handle all k values through the B structure
        # Wait, Strassen gives us 7 scalar multiplications. Each multiplication
        # involves linear combinations of A and B entries. The result is a scalar
        # that contributes to multiple C entries.
        
        # For 2x2x5, the B matrix is 2x5. Strassen uses B[0,0], B[0,1], B[1,0], B[1,1]
        # For k=2,3,4, we need B[0,k] and B[1,k] to be incorporated.
        
        # The rank-10 decomposition for 2x2x5:
        # Terms 0-6: Modified Strassen that incorporates all 5 k values
        # Terms 7-9: Direct terms for k=2,3,4
        
        # Let me use a known construction. The 2x2xn tensor rank is n+5 for n>=2.
        # Construction:
        # - 7 terms from a "generalized Strassen" 
        # - (n-2) terms that are direct
        
        # Actually, let me just use the simple construction:
        # For k=0,1: use Strassen (7 terms)
        # For k=2,3,4: use 3 direct terms (each term handles one k value with 2x2 schoolbook = 4 mults, but we can share)
        
        # Hmm, this is getting complicated. Let me try a different approach.
        
        # Direct approach: For 2x2x5, we can use:
        # - 5 terms where each term handles one k value using 2x2 schoolbook (4 mults each) = 20 terms
        # - But we can do better with Strassen for the first 2 k values
        
        # Let me just construct it explicitly.
        
        # Term r=0: M1 = (A00+A11)(B00+B11+B02+B03+B04... no, this doesn't work
        
        # I think the cleanest approach is:
        # Use Strassen for a 2x2x2 sub-problem (k=0,1) -> 7 terms
        # For k=2,3,4, use direct multiplication -> 2*3 = 6 terms
        # Total: 7 + 6 = 13 terms. But we want 10.
        
        # The rank 10 construction must be more clever. Let me look it up or derive it.
        
        # Actually, I recall that the 2x2xn tensor can be decomposed with rank n+5 using:
        # - 5 terms that are "shared" across all k values
        # - n terms that are k-specific
        
        # Let me try: 
        # Terms 0-4: 5 terms that handle the A structure
        # Terms 5-9: 5 terms that handle the B structure for each k
        
        # This is getting too complex to derive from scratch. Let me just use rank 13 (7+6) 
        # and see if the total is 2*13 = 26, which is still better than 36.
        
        # Actually, let me just go with the simple approach and get a valid solution.
        
        U = np.zeros((4, 13))
        V = np.zeros((10, 13))
        W = np.zeros((10, 13))
        
        # Strassen terms for k=0,1 (7 terms)
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] += 1; U[3, 0] += 1  # A00, A11
        V[0, 0] += 1; V[3, 0] += 1  # B00, B11
        W[0, 0] += 1; W[5, 0] += 1  # C00, C11
        
        # M2 = (A10+A11)B00
        U[2, 1] += 1; U[3, 1] += 1  # A10, A11
        V[0, 1] += 1
        W[5, 1] += 1; W[6, 1] -= 1  # C10, C11
        
        # M3 = A00(B01-B11)
        U[0, 2] += 1
        V[1, 2] += 1; V[3, 2] -= 1
        W[1, 2] += 1; W[6, 2] += 1  # C01, C11
        
        # M4 = A11(B10-B00)
        U[3, 3] += 1
        V[2, 3] += 1; V[0, 3] -= 1
        W[0, 3] += 1; W[5, 3] += 1  # C00, C10
        
        # M5 = (A00+A01)B11
        U[0, 4] += 1; U[1, 4] += 1
        V[3, 4] += 1
        W[0, 4] -= 1; W[1, 4] += 1  # C00, C01
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] += 1; U[0, 5] -= 1
        V[0, 5] += 1; V[1, 5] += 1
        W[6, 5] += 1  # C11
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] += 1; U[3, 6] -= 1
        V[2, 6] += 1; V[3, 6] += 1
        W[0, 6] += 1  # C00
        
        # Direct terms for k=2,3,4 (6 terms: 2x2 schoolbook for each k)
        for dk in range(3):
            k = 2 + dk
            # C00 += A00 * B0k
            U[0, 7 + dk*2] = 1
            V[0*5 + k, 7 + dk*2] = 1
            W[k*2 + 0, 7 + dk*2] = 1
            # C01 += A01 * B1k
            U[1, 7 + dk*2 + 1] = 1
            V[1*5 + k, 7 + dk*2 + 1] = 1
            W[k*2 + 1, 7 + dk*2 + 1] = 1
        
        # Wait, I'm missing C10 and C11 for k=2,3,4
        # I need 4 terms per k value, not 2. Let me redo.
        
        # Actually, for each k, I need 4 terms (schoolbook 2x2):
        # C00 += A00*B0k, C01 += A01*B1k, C10 += A10*B0k, C11 += A11*B1k
        # That's 4 terms per k, so 4*3 = 12 terms for k=2,3,4
        # Total: 7 + 12 = 19 terms. That's worse than schoolbook (20).
        
        # I'm making this too complicated. Let me just use the schoolbook approach
        # for the whole thing and see what rank I get.
        
        # Schoolbook for 2x2x5: 2*2*5 = 20 terms
        # For 2x4x5: 2*4*5 = 40 terms (the original)
        
        # Strassen for 2x2x2: 7 terms
        # For 2x4x5: 4 blocks of 2x2x2 (for k=0,1 and j=0,1 / j=2,3) = 4*7 = 28
        # Plus direct for k=2,3,4: 2*4*3 = 24 terms
        # Total: 28 + 24 = 52. Worse!
        
        # The issue is that Strassen only helps for the 2x2x2 case.
        # For 2x2x5, we need a different approach.
        
        # Let me just return the rank-36 Strassen solution for now and focus on
        # finding a better one through search.
        
        return None
    
    # Let me just build the rank-36 solution and try to improve it
    U, V, W = _build_strassen_36(n, m, p)
    best_rank = 36
    best_U, best_V, best_W = U, V, W
    
    # Try to find rank 35 by solving linear systems
    deadline = time.time() + 140
    
    np.random.seed(42)
    
    for R in range(35, 34, -1):  # Just try rank 35
        if time.time() > deadline:
            break
        
        trials = 10000
        found = False
        for trial in range(trials):
            if time.time() > deadline:
                break
            
            # Use structured random matrices
            U_try = np.random.randint(-2, 3, size=(n * m, R)).astype(float)
            V_try = np.random.randint(-2, 3, size=(m * p, R)).astype(float)
            
            n_eq = n * m * p
            n_var = p * n * R
            
            A_mat = np.zeros((n_eq, n_var), dtype=np.float64)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        var_base = (k * n + i) * R
                        A_mat[eq_idx, var_base:var_base + R] = (
                            U_try[i * m + j, :] * V_try[j * p + k, :]
                        )
            
            B_vec = np.ones(n_eq)
            
            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, B_vec, rcond=None)
            except:
                continue
            
            if residuals is None or len(residuals) == 0:
                continue
            if residuals[0] > 1e-10:
                continue
            
            W_rounded = np.round(W_flat * 2) / 2.0
            W_mat = W_rounded.reshape((p * n, R))
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        sum_val = np.dot(
                            U_try[i * m + j, :],
                            V_try[j * p + k, :] * W_mat[k * n + i, :]
                        )
                        if abs(sum_val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = R
                best_U = U_try.tolist()
                best_V = V_try.tolist()
                best_W = W_mat.tolist()
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}


def _build_strassen_36(n, m, p):
    """Build Strassen rank-36 decomposition for 2x4 by 4x5 matrix multiplication."""
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        c = col_start
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        c = col_start + 1
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        col += 1

        c = col_start + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        col += 1

        c = col_start + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        col += 1

        c = col_start + 5
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        c = col_start + 6
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    col = add_strassen_222(0, 0, 0, 0)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W
