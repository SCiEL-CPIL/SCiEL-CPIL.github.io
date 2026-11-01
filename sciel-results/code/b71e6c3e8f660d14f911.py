"""Matrix multiplication 2x2 by 2x3 rank-minimal exact decomposition."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = None

def _now():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 140  # 10s safety margin before 150s kill
    return time.time()

def _time_left():
    return _DEADLINE - time.time()

def _check_deadline():
    if time.time() >= _DEADLINE - 2:
        return True
    return False

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor is T[i, j, k] for C[i,k] = sum_j A[i,j] * B[j,k]
    # In flattened form:
    # U[i*m + j, r] = u_{ijr}
    # V[j*p + k, r] = v_{jkr}
    # W[k*n + i, r] = w_{kir}
    # Product: sum_r u_{ijr} * v_{jkr} * w_{kir} should equal T[i,j,k] = delta (which is 1 for valid indices)
    # Actually the bilinear form: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] = 1 for all valid i,j,k (since it's just the multiplication map)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Verify with random matrices
    rng = np.random.default_rng(42)
    for _ in range(20):
        A = rng.integers(-10, 11, size=(n, m)).astype(float)
        B = rng.integers(-10, 11, size=(m, p)).astype(float)
        C = A @ B
        # Compute via decomposition
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_vec = np.array(U)[:, r]  # (n*m,)
            v_vec = np.array(V)[:, r]  # (m*p,)
            w_vec = np.array(W)[:, r]  # (p*n,)
            # u reshaped to (n, m), v reshaped to (m, p), w reshaped to (p, n)
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            # The contribution: for each (i,k), sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is like a trace of u_mat * v_mat * w_mat appropriately
            # Actually: sum_j u_mat[i,j] * v_mat[j,k] = (u_mat @ v_mat)[i,k]
            # Then multiply by w_mat[k,i] and sum over k? No, w_mat[k,i] is per (k,i)
            # The formula: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                    C_decomp[i, k] += s
        if np.max(np.abs(C - C_decomp)) > tol:
            return False
    return True

def _try_11_decomposition():
    """
    Try to construct a rank-11 decomposition by combining:
    - Strassen's 7-term decomposition for 2x2*2x2 (for the first 2 columns of B)
    - 4 schoolbook terms for the third column of B
    
    Total: 7 + 4 = 11 terms.
    
    The key insight: for B = [B1, B2] where B1 is 2x2 (cols 1,2) and B2 is 2x1 (col 3),
    C = [C1, C2, C3] where C1 = A*B1_cols12 (2x2), C3 = A*B2 (2x1).
    
    For the 2x2 part, use Strassen (rank 7).
    For the 2x1 part (A * column vector), we need rank 2 (schoolbook) since
    (A @ v)[i] = sum_j A[i,j] v[j], which is a 2x2x1 tensor of rank 2.
    
    Wait, but we can potentially share terms. Let me think more carefully.
    
    Actually, the standard approach: the 2x2x3 matrix multiplication tensor has
    known rank bounds. The trivial rank is 2*2*3 = 12. Can we do better?
    
    One approach: use the fact that multiplying by a 2x3 matrix can be seen as
    two separate 2x2 multiplications with some sharing. But the cleanest known
    result is that the rank of 2x2x3 is at most 12 (schoolbook) and it's not
    immediately obvious we can go below 12.
    
    However, there's a known result: the rank of matrix multiplication [2,2,3] is 
    exactly 12? Or can it be 11?
    
    Let me try a different approach: search for a rank-11 decomposition by
    starting from the 12-term schoolbook and trying to merge/simplify.
    
    Actually, let me just try to verify if a rank-11 solution exists by doing
    a numerical search.
    """
    n, m, p = 2, 2, 3
    target_rank = 11
    
    # The tensor T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a 3-way tensor decomposition. Let's set up the problem.
    # Flatten the tensor: T_flat of size n*m*p = 12
    # Each term r contributes a rank-1 tensor which when flattened gives:
    # u_r (size n*m) outer v_r (size m*p) outer w_r (size p*n) -> but these share indices
    
    # Actually, the constraint is:
    # For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's parameterize and solve. We have 12 equations (one per (i,j,k) triple)
    # and 3 * rank * (appropriate size) unknowns.
    
    # For rank 11: 11 * (4 + 6 + 6) = 11 * 16 = 176 unknowns, 12 equations.
    # This is underdetermined, so solutions should exist if the structure allows.
    
    # Let me try a specific structured approach:
    # Start with schoolbook (rank 12) and try to find a linear dependence.
    
    # Schoolbook terms: for each (i,j,k), one term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # The 12 terms correspond to the 12 (i,j,k) triples.
    
    # Can we express one of these as a linear combination of the others?
    # In general, no, because each term is "unique" in its index pattern.
    
    # Let me try a numerical approach: set up the system and solve.
    
    # Alternative: use the known result that [2,2,3] multiplication has rank at most 12,
    # and check if 11 is achievable.
    
    # Let me try a direct numerical search for rank 11.
    
    # Parameterize:
    # U: 4 x 11, V: 6 x 11, W: 6 x 11
    # Constraints: for each (i,j,k) in {(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),
    #                                   (1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,1),(1,1,2)}:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 12 equations in 176 unknowns (with bilinear/trilinear structure).
    
    # Let me try to fix W to be the identity-like pattern and solve for U, V.
    
    # Actually, let me try a simpler approach: just use the schoolbook rank-12 solution
    # and return it. The task says "find a rank-minimal exact tensor decomposition" but
    # if we can't find rank < 12, we return rank 12.
    
    # But first, let me try to see if rank 11 is possible with a structured search.
    
    # Try: use Strassen-like ideas. For the 2x2x2 part (columns 0,1 of B), use rank 7.
    # For column 2 of B, use rank 2 (since A @ b for b in R^2 is a 2x1x2 tensor, rank 2).
    # Total: 7 + 2 = 9? But wait, the indices don't fully separate.
    
    # Let me be more careful. The output C is 2x3. C[:, 0:2] depends on A and B[:, 0:2] (2x2x2, rank 7 via Strassen).
    # C[:, 2] depends on A and B[:, 2] (2x2x1, rank 2).
    # But the U factors for these two parts both depend on A, so we can share the U structure.
    
    # Strassen for 2x2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For C[:, 2] = A @ B[:, 2]:
    # C13 = a11*b13 + a12*b23
    # C23 = a21*b13 + a22*b23
    
    # So total terms: 7 (Strassen) + 2 (for C13, C23) = 9 terms.
    # But we need to express all in the U, V, W framework.
    
    # U has 4 rows (for A's 4 entries) x rank columns
    # V has 6 rows (for B's 6 entries) x rank columns  
    # W has 6 rows (for C's 6 entries in k*n+i order) x rank columns
    
    # C index order: W[k*n+i, r] where k=0,1 and i=0,1
    # So C[0,0] -> W[0*2+0]=W[0], C[0,1] -> W[1], C[0,2] -> W[2]
    # C[1,0] -> W[1*2+0]=W[2], C[1,1] -> W[3], C[1,2] -> W[4]
    
    # Wait, p*n = 3*2 = 6, so W has 6 rows.
    # k ranges 0..2, i ranges 0..1
    # W[k*2+i, r]:
    # k=0,i=0 -> W[0] = C[0,0]
    # k=0,i=1 -> W[1] = C[0,1]
    # k=1,i=0 -> W[2] = C[1,0]
    # k=1,i=1 -> W[3] = C[1,1]
    # k=2,i=0 -> W[4] = C[0,2]
    # k=2,i=1 -> W[5] = C[1,2]
    
    # A index: U[i*m+j, r], i=0,1, j=0,1
    # U[0] = a00, U[1] = a01, U[2] = a10, U[3] = a11
    
    # B index: V[j*p+k, r], j=0,1, k=0,1,2
    # V[0] = b00, V[1] = b01, V[2] = b02, V[3] = b10, V[4] = b11, V[5] = b12
    
    # Now let me write out the 9-term decomposition:
    
    # Strassen terms (for C[0:2, 0:2]):
    # Term 0: M1 = (a00+a11)(b00+b11)
    #   U: [1, 0, 0, 1], V: [1, 0, 0, 0, 1, 0], W: contributes to C[0,0] and C[1,1]
    # Term 1: M2 = (a10+a11)b00
    #   U: [0, 0, 1, 1], V: [1, 0, 0, 0, 0, 0], W: contributes to C[1,0] and C[1,1]
    # Term 2: M3 = a00(b01-b11)
    #   U: [1, 0, 0, 0], V: [0, 1, 0, 0, -1, 0], W: contributes to C[0,1] and C[1,1]
    # Term 3: M4 = a11(b10-b00)
    #   U: [0, 0, 0, 1], V: [0, 0, 0, 1, 0, 0] - [1,0,0,0,0,0] = [-1, 0, 0, 1, 0, 0], W: C[0,0], C[1,0]
    # Term 4: M5 = (a00+a01)b11
    #   U: [1, 1, 0, 0], V: [0, 0, 0, 0, 1, 0], W: C[0,0], C[0,1]
    # Term 5: M6 = (a10-a00)(b00+b01)
    #   U: [-1, 0, 1, 0], V: [1, 1, 0, 0, 0, 0], W: C[1,1]
    # Term 6: M7 = (a01-a11)(b10+b11)
    #   U: [0, 1, 0, -1], V: [0, 0, 0, 1, 1, 0], W: C[0,0], C[1,1]
    
    # Let me define the W contributions more carefully.
    # W[k*2+i, r] is the coefficient for C[i,k] in term r.
    
    # For Strassen:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # W indices: C[0,0]->W[0], C[0,1]->W[1], C[1,0]->W[2], C[1,1]->W[3]
    
    # Term 0 (M1): W[0]=1, W[1]=0, W[2]=0, W[3]=1, W[4]=0, W[5]=0
    # Term 1 (M2): W[0]=0, W[1]=0, W[2]=1, W[3]=-1, W[4]=0, W[5]=0
    # Term 2 (M3): W[0]=0, W[1]=1, W[2]=0, W[3]=1, W[4]=0, W[5]=0
    # Term 3 (M4): W[0]=1, W[1]=0, W[2]=1, W[3]=0, W[4]=0, W[5]=0
    # Term 4 (M5): W[0]=-1, W[1]=1, W[2]=0, W[3]=0, W[4]=0, W[5]=0
    # Term 5 (M6): W[0]=0, W[1]=0, W[2]=0, W[3]=1, W[4]=0, W[5]=0
    # Term 6 (M7): W[0]=1, W[1]=0, W[2]=0, W[3]=1, W[4]=0, W[5]=0
    
    # For C[0,2] and C[1,2]:
    # C[0,2] = a00*b02 + a01*b12 -> Term 7
    # C[1,2] = a10*b02 + a11*b12 -> Term 8
    
    # Term 7: U: [1, 1, 0, 0], V: [0, 0, 1, 0, 0, 1], W: W[4]=1, rest 0
    # Term 8: U: [0, 0, 1, 1], V: [0, 0, 1, 0, 0, 1], W: W[5]=1, rest 0
    
    # Total: 9 terms! Let me verify this.
    
    U = [
        [1, 0, 1, 0, 1, -1, 0, 1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1, 1, 0],  # a01
        [0, 1, 0, 0, 0, 1, 0, 0, 1],  # a10
        [1, 1, 0, 1, 0, 0, -1, 0, 1],  # a11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0, 0, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0, 0, 0],  # b01
        [0, 0, 0, 0, 0, 0, 0, 1, 1],  # b02
        [0, 0, 0, 1, 0, 0, 1, 0, 0],  # b10
        [1, 0, -1, 0, 1, 0, 1, 0, 0],  # b11
        [0, 0, 0, 0, 0, 0, 0, 1, 1],  # b12
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1, 0, 0],  # C[0,0]
        [0, 0, 1, 0, 1, 0, 0, 0, 0],  # C[0,1]
        [0, 1, 0, 1, 0, 0, 0, 0, 0],  # C[1,0]
        [1, -1, 1, 0, 0, 1, 1, 0, 0],  # C[1,1]
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # C[0,2]
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # C[1,2]
    ]
    
    if _verify(U, V, W, 2, 2, 3):
        return U, V, W, 9
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 9 first
    result = _try_11_decomposition()
    if result is not None:
        U, V, W, rank = result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook rank 12
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
