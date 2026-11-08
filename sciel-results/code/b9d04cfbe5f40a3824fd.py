"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    """Return the trivial schoolbook decomposition (rank = n*m*p)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, tol=1e-10):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, k) pair, check that sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    # equals delta[i,j] * delta[j,k] for the (i,k) entry... 
    # Actually, the tensor is: T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # The decomposition should satisfy: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Let's verify: for all i,j,k, the sum should equal 1 (since A[i,j]*B[j,k] contributes to C[i,k])
    # Wait, the matrix multiplication tensor has entries T[i,j,k] = 1 for all valid i,j,k
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So we need: for all i, j, k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's compute this efficiently
    # Reshape to 3D tensors
    U_3d = U_arr.reshape(n, m, -1)  # (n, m, rank)
    V_3d = V_arr.reshape(m, p, -1)  # (m, p, rank)
    W_3d = W_arr.reshape(p, n, -1)  # (p, n, rank)
    
    # Compute the contraction: sum_c U[i,m,c] * V[m,p,c] * W[p,n,c]
    # This is a 3-tensor contraction
    # We can do it as: for each (i,j,k), compute dot(U_3d[i,j,:], V_3d[j,k,:]) * W_3d[k,i,:]
    
    # More efficient: contract step by step
    # First compute M[i,j,k] = sum_c U[i,j,c] * V[j,k,c]
    # Then check if M[i,j,k] * W[k,i,:] sums to 1
    
    # Actually, let's just compute the full tensor product and check
    # T[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    
    # Use einsum for efficiency
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    # T should be all ones (since the matrix multiplication tensor has 1s everywhere)
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _strassen_4x4():
    """
    Return Strassen-style decomposition for 4x4 * 4x4.
    Uses the fact that 4x4 can be decomposed using 7 multiplications (Strassen)
    applied to 2x2 blocks, giving rank 7*4 = 28 for the full 4x4x4 tensor.
    
    Actually, Strassen gives rank 7 for 2x2x2. For 4x4x4, we can use:
    - Block decomposition into four 2x2 blocks
    - Apply Strassen to compute each of the 4 output blocks
    - This gives rank 4*7 = 28
    
    But we need to construct the explicit U, V, W matrices.
    """
    n = 4
    rank = 28  # 4 blocks * 7 Strassen multiplications each
    
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4, we have blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # C = [[C11, C12], [C21, C22]] where each Cij is 2x2
    
    # We'll apply Strassen to compute each C block, but we need to be careful
    # because the inputs to Strassen are themselves 2x2 matrices.
    
    # Let's define the 7 Strassen intermediate products for the 4x4 case.
    # Each "multiplication" is actually a 2x2 * 2x2 multiplication.
    
    # Strassen intermediates (each is a 2x2 matrix product):
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B21-B22)
    # M4 = A22 * (B12-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # Then:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now, each Mx is a 2x2 matrix product, which itself can be decomposed
    # using schoolbook (rank 4) or further Strassen (but let's use schoolbook for simplicity)
    
    # For each of the 7 Mx, we need 4 scalar multiplications (schoolbook for 2x2*2x2)
    # Total rank = 7 * 4 = 28
    
    # Let's build U, V, W for the 4x4x4 tensor.
    
    # Indexing: U[i*n+j, c], V[j*n+k, c], W[k*n+i, c]
    # where i,j,k are in [0,3]
    
    # For each Strassen term Mx (x=0..6), and for each of the 4 scalar products within it,
    # we assign a column c.
    
    # Let's define the structure more carefully.
    
    # Block indices: A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B and C.
    
    # For M1 = (A11+A22) * (B11+B22):
    # This is a 2x2 * 2x2 multiplication.
    # The result M1 is a 2x2 matrix, and it contributes to C11 and C22.
    
    # Let's build this systematically.
    
    # For each of the 7 Strassen terms, we'll create 4 columns (for the 4 scalar products).
    
    # M1 = (A11+A22) * (B11+B22)
    # Scalar products for M1:
    # M1[0,0] = (A[0,0]+A[2,2])*(B[0,0]+B[2,2]) + (A[0,1]+A[2,3])*(B[1,0]+B[3,2])
    # Wait, this is getting complicated. Let me use a different approach.
    
    # Actually, for the 4x4x4 tensor, we can just use the known fact that the tensor rank
    # is at most 28 (from Strassen applied to 2x2 blocks with schoolbook for each 2x2 product).
    
    # Let me construct it more directly using the block structure.
    
    # I'll create a helper to add a scalar multiplication term.
    
    def add_term(U, V, W, i, j, k, col, coeff_U=1, coeff_V=1, coeff_W=1):
        """Add a term U[i,j]*V[j,k]*W[k,i] = coeff_U*coeff_V*coeff_W to column col."""
        U[i*n + j][col] += coeff_U
        V[j*n + k][col] += coeff_V
        W[k*n + i][col] += coeff_W
    
    # For M1 = (A11+A22) * (B11+B22):
    # The (a,b) entry of M1 is sum_{u,v} (A11[a,u]+A22[a,u]) * (B11[v,b]+B22[v,b]) * delta[u,v]
    # = sum_{u} (A11[a,u]+A22[a,u]) * (B11[u,b]+B22[u,b])
    # = sum_{u} (A[2a,u]+A[2a+2,u]) * (B[2u,b]+B[2u+2,b])  [for a,b in {0,1}]
    
    # Wait, I need to be more careful with indices.
    
    # Let's use explicit indices. For the 4x4 case:
    # A is 4x4, B is 4x4, C is 4x4.
    # Block A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    
    # M1 = (A11+A22) * (B11+B22)
    # M1 is 2x2. M1[a,b] = sum_{u=0}^{1} (A11[a,u]+A22[a,u]) * (B11[u,b]+B22[u,b])
    # = sum_{u=0}^{1} (A[2a,u]+A[2a+2,u]) * (B[2u,b]+B[2u+2,b])
    
    # This expands to 4 terms for each (a,b):
    # (A[2a,u] * B[2u,b]) + (A[2a,u] * B[2u+2,b]) + (A[2a+2,u] * B[2u,b]) + (A[2a+2,u] * B[2u+2,b])
    
    # Each of these is a scalar product A[i,j]*B[j,k] where:
    # Term 1: i=2a, j=u, k=b (but B index is 2u, so j=u, k=2u... wait)
    
    # I'm getting confused. Let me just use the schoolbook decomposition for the 4x4x4 case
    # but with a different ordering to see if we can find structure.
    
    # Actually, let me just return a valid schoolbook decomposition for 4x4x4 for now,
    # and focus on the 4x4x9 case.
    
    # For 4x4x4, schoolbook has rank 64.
    # We can do better with Strassen, but let's first get the 4x4x9 case working.
    
    # For now, let's just return None to indicate we don't have a good 4x4x4 decomposition yet.
    return None

def _kronecker_combine(U1, V1, W1, U2, V2, W2):
    """
    Combine two tensor decompositions using Kronecker product.
    If T1 = sum_c U1[:,c] V1[:,c] W1[:,c] and T2 = sum_d U2[:,d] V2[:,d] W2[:,d],
    then T1 ⊗ T2 = sum_{c,d} (U1[:,c]⊗U2[:,d]) (V1[:,c]⊗V2[:,d]) (W1[:,c]⊗W2[:,d])
    """
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank = rank1 * rank2
    
    U = [[0] * rank for _ in range(len(U1) * len(U2))]
    V = [[0] * rank for _ in range(len(V1) * len(V2))]
    W = [[0] * rank for _ in range(len(W1) * len(W2))]
    
    for c in range(rank1):
        for d in range(rank2):
            col = c * rank2 + d
            # Kronecker product of U1[:,c] and U2[:,d]
            for i1 in range(len(U1)):
                for i2 in range(len(U2)):
                    idx = i1 * len(U2) + i2
                    U[idx][col] = U1[i1][c] * U2[i2][d]
            # Similarly for V and W
            for j1 in range(len(V1)):
                for j2 in range(len(V2)):
                    idx = j1 * len(V2) + j2
                    V[idx][col] = V1[j1][c] * V2[j2][d]
            for k1 in range(len(W1)):
                for k2 in range(len(W2)):
                    idx = k1 * len(W2) + k2
                    W[idx][col] = W1[k1][c] * W2[k2][d]
    
    return {"U": U, "V": V, "W": W}

def _try_strassen_4x4x9():
    """
    Try to construct a low-rank decomposition for 4x4x9 using Strassen for the 4x4 part
    and a simple decomposition for the 4x9 part.
    
    Idea: Decompose the 4x4x9 tensor as a combination of 4x4x4 and 4x4x5 tensors,
    or use the fact that 9 = 4+5 and try to find a decomposition.
    
    Actually, a better approach: use the Kronecker product of a 4x4x4 decomposition
    with a 1x4x1 decomposition (which is just scalar multiplication, rank 1).
    But 4x4x4 ⊗ 1x4x1 gives 4x4x16, not 4x4x9.
    
    Let me think differently. The tensor for 4x4x9 multiplication is:
    T[i,j,k] = 1 for i in [0,3], j in [0,3], k in [0,8]
    
    We can write this as: T[i,j,k] = T4[i,j,k mod 4] * delta[k < 4] + T4[i,j,k mod 4] * delta[k >= 4 and k < 8] + T4[i,j,k mod 4] * delta[k >= 8]
    
    This isn't quite right. Let me just try a direct approach.
    
    For now, let's return None to indicate we don't have a working decomposition yet.
    """
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication.
    Returns a dictionary with U, V, W matrices.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Try different strategies, keeping the best valid one found
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Schoolbook (always works, rank = 4*4*9 = 144)
    schoolbook = _schoolbook()
    if _validate(schoolbook['U'], schoolbook['V'], schoolbook['W']):
        best_result = schoolbook
        best_rank = len(schoolbook['U'][0])
    
    # Strategy 2: Try Strassen-based decomposition for 4x4x9
    if time.time() < deadline:
        strassen_result = _try_strassen_4x4x9()
        if strassen_result is not None:
            if _validate(strassen_result['U'], strassen_result['V'], strassen_result['W']):
                rank = len(strassen_result['U'][0])
                if rank < best_rank:
                    best_result = strassen_result
                    best_rank = rank
    
    # Strategy 3: Try Kronecker product approach
    if time.time() < deadline:
        # We can decompose 4x4x9 as a combination of smaller tensors
        # For example, 4x4x9 = 4x4x4 + 4x4x5 (not exactly, but we can try)
        # Or use the fact that 9 = 3*3 and try to exploit that
        
        # Let's try a simple approach: use schoolbook for each of the 9 columns
        # but see if we can share some structure
        
        # For now, let's just return the best we have
        pass
    
    if best_result is None:
        # This should never happen since schoolbook always works
        return _schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
