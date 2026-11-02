"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_factors():
    """
    Return Strassen's rank-7 factors for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of 7 elements, each a flat 4-vector (2x2 matrix).
    """
    # Strassen's algorithm for 2x2 matrices:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # 
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e  
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-d)(f+h)
    # p7 = (b-d)(e+g)
    #
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # In tensor decomposition form, for each factor r:
    # U[r] is the matrix that multiplies A, V[r] multiplies B, W[r] is the output weight
    # (A*B)[i,k] = sum_r sum_j (A*U[r])_{i,j} * (V[r]*B)_{j,k} * W[r]_{k,i}
    # Actually, the standard form is:
    # (A*B)[i,k] = sum_r (sum_j A[i,j] U[r][j,i']) * (sum_j' V[r][j',k] B[j',k]) ... 
    
    # Let me use the explicit linear forms. For factor r:
    # U[r][i,j] = coefficient of A[i,j] in the linear form for factor r
    # V[r][j,k] = coefficient of B[j,k] in the linear form for factor r
    # W[r][k,i] = coefficient for output C[k,i]... wait, C[i,k]
    
    # The tensor is T[i,j,k] = 1 for all i,j,k (since (A*B)[i,k] = sum_j A[i,j]B[j,k])
    # Decomposition: T[i,j,k] = sum_r U[r][i,j] * V[r][j,k] * W[r][k,i]
    
    # For Strassen:
    # Factor 1: p1 = (a+d)(e+h)
    #   U[1] = [[1,0],[1,0]] (a+d), V[1] = [[1,1],[0,0]] (e+h), W[1] = [[1,0],[0,1]] (C11 += p1, C22 += p1)
    #   Wait, W[1][k,i] means W[1][0,0]=1 (C[0,0]), W[1][1,1]=1 (C[1,1])
    
    # Let me define more carefully. C[i,k] gets contribution from factor r:
    # C[i,k] += W[r][k,i] * (sum_j A[i,j]*U[r][i,j]) * (sum_j B[j,k]*V[r][j,k])
    
    # Factor 1: p1 = (a+d)(e+h) contributes to C[0,0] and C[1,1]
    #   U[1][0,:] = [1,0], U[1][1,:] = [1,0]  (a+d for both rows)
    #   V[1][:,0] = [1,0], V[1][:,1] = [1,0]  (e+h for both cols)
    #   W[1][0,0] = 1, W[1][1,1] = 1
    
    # Factor 2: p2 = (c+d)e contributes to C[1,0] and C[1,1]
    #   U[2][0,:] = [0,0], U[2][1,:] = [1,1]  (c+d for row 1)
    #   V[2][:,0] = [1,0], V[2][:,1] = [0,0]  (e for col 0)
    #   W[2][0,1] = 1, W[2][1,1] = -1  (C[1,0] += p2, C[1,1] -= p2)
    
    # Factor 3: p3 = a(f-h) contributes to C[0,1] and C[1,1]
    #   U[3][0,:] = [1,0], U[3][1,:] = [0,0]  (a for row 0)
    #   V[3][:,0] = [0,0], V[3][:,1] = [1,-1]  (f-h for col 1)
    #   W[3][1,0] = 1, W[3][1,1] = 1  (C[0,1] += p3, C[1,1] += p3)
    
    # Factor 4: p4 = d(g-e) contributes to C[0,0] and C[1,0]
    #   U[4][0,:] = [0,0], U[4][1,:] = [0,1]  (d for row 1)
    #   V[4][:,0] = [0,1], V[4][:,1] = [-1,0]  (g-e)
    #   W[4][0,0] = 1, W[4][0,1] = 1  (C[0,0] += p4, C[1,0] += p4)
    
    # Factor 5: p5 = (a+b)h contributes to C[0,0] and C[0,1]
    #   U[5][0,:] = [1,1], U[5][1,:] = [0,0]  (a+b for row 0)
    #   V[5][:,0] = [0,0], V[5][:,1] = [0,1]  (h for col 1)
    #   W[5][0,0] = -1, W[5][1,0] = 1  (C[0,0] -= p5, C[0,1] += p5)
    
    # Factor 6: p6 = (c-d)(f+h) contributes to C[1,1]
    #   U[6][0,:] = [0,0], U[6][1,:] = [1,-1]  (c-d for row 1)
    #   V[6][:,0] = [0,0], V[6][:,1] = [1,1]  (f+h for col 1)
    #   W[6][1,1] = 1  (C[1,1] += p6)
    
    # Factor 7: p7 = (b-d)(e+g) contributes to C[0,0]
    #   U[7][0,:] = [0,1], U[7][1,:] = [0,-1]  (b-d)
    #   V[7][:,0] = [1,0], V[7][:,1] = [0,1]  (e+g)
    #   W[7][0,0] = 1  (C[0,0] += p7)
    
    U = []
    V = []
    W = []
    
    # Factor 1
    U.append([1,0,1,0])  # [[1,0],[1,0]]
    V.append([1,1,0,0])  # [[1,1],[0,0]]
    W.append([1,0,0,1])  # [[1,0],[0,1]]
    
    # Factor 2
    U.append([0,0,1,1])  # [[0,0],[1,1]]
    V.append([1,0,0,0])  # [[1,0],[0,0]]
    W.append([0,1,-1,0])  # [[0,-1],[1,0]] -> W[k,i]: W[0,1]=-1, W[1,0]=1
    
    # Factor 3
    U.append([1,0,0,0])  # [[1,0],[0,0]]
    V.append([0,0,1,-1])  # [[0,1],[0,-1]] -> wait, V[j,k], so V[0,1]=1, V[1,1]=-1
    V[-1] = [0,1,0,-1]  # [[0,1],[0,-1]]
    W.append([0,1,1,0])  # [[0,1],[1,0]] -> W[1,0]=1, W[1,1]=1
    
    # Factor 4
    U.append([0,0,0,1])  # [[0,0],[0,1]]
    V.append([0,-1,1,0])  # [[0,1],[1,0]] -> V[0,0]=0, V[0,1]=-1, V[1,0]=1, V[1,1]=0
    W.append([1,0,1,0])  # [[1,1],[0,0]] -> W[0,0]=1, W[0,1]=1
    
    # Factor 5
    U.append([1,1,0,0])  # [[1,1],[0,0]]
    V.append([0,0,0,1])  # [[0,0],[0,1]]
    W.append([-1,0,0,1])  # [[-1,0],[0,1]] -> W[0,0]=-1, W[1,0]=1
    
    # Factor 6
    U.append([0,0,1,-1])  # [[0,0],[1,-1]]
    V.append([0,0,1,1])  # [[0,1],[0,1]] -> wait, V[j,k], V[0,1]=1, V[1,1]=1
    V[-1] = [0,1,0,1]  # [[0,1],[0,1]]
    W.append([0,0,0,1])  # [[0,0],[0,1]] -> W[1,1]=1
    
    # Factor 7
    U.append([0,1,0,-1])  # [[0,1],[0,-1]]
    V.append([1,0,0,1])  # [[1,0],[0,1]]
    W.append([1,0,0,0])  # [[1,0],[0,0]] -> W[0,0]=1
    
    return U, V, W

def _build_4x4x4_strassen(n=4, m=4, p=4, deadline=None):
    """
    Build a rank-49 decomposition for 4x4x4 using Strassen on 2x2 blocks.
    4x4 matrix = 2x2 blocks of 2x2 matrices.
    A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    C = A*B = [[A11*B11+A12*B21, A11*B12+A12*B22],
               [A21*B11+A22*B21, A21*B12+A22*B22]]
    
    Each block product is a 2x2x2 multiplication, which can be done in rank 7.
    So we need 4 block products * 7 = 28 for the 2x2 block structure,
    but each block product involves 2x2 matrices, so the total rank is 4*7=28?
    No, the total rank should be 7*7=49 if we use Kronecker product.
    
    Actually, the Kronecker product of two rank-7 decompositions gives a rank-49
    decomposition for 4x4x4.
    """
    if deadline and time.time() > deadline:
        return None, None, None, None
    
    U2, V2, W2 = _strassen_factors()
    rank = 7 * 7  # 49
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U[r] = U2[r1] ⊗ U2[r2]
            # V[r] = V2[r1] ⊗ V2[r2]
            # W[r] = W2[r1] ⊗ W2[r2]
            
            U_flat = np.array(U2[r1]) * np.array(U2[r2])
            V_flat = np.array(V2[r1]) * np.array(V2[r2])
            W_flat = np.array(W2[r1]) * np.array(W2[r2])
            
            U[:, r] = U_flat
            V[:, r] = V_flat
            W[:, r] = W_flat
    
    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), 4, 4, 4):
        return U.tolist(), V.tolist(), W.tolist(), rank
    
    return None, None, None, None

def _build_4x4x7(deadline):
    """
    Build a decomposition for 4x4x7 using Strassen for the first 4 columns
    and schoolbook for the last 3 columns.
    """
    if time.time() > deadline:
        return None, None, None, None
    
    # First, get the 4x4x4 Strassen decomposition
    U4, V4, W4, rank4 = _build_4x4x4_strassen(4, 4, 4, deadline)
    if U4 is None:
        return None, None, None, None
    
    U4 = np.array(U4)
    V4 = np.array(V4)
    W4 = np.array(W4)
    
    # Now handle the last 3 columns of B (columns 4, 5, 6)
    # For each of these columns, we need rank 16 (schoolbook for 4x4x1)
    rank_extra = 3 * 16  # 48
    rank_total = rank4 + rank_extra
    
    U = np.zeros((16, rank_total))
    V = np.zeros((28, rank_total))
    W = np.zeros((28, rank_total))
    
    # Copy the 4x4x4 part
    U[:, :rank4] = U4
    V[:16, :rank4] = V4[:, :16]  # Only first 16 rows (for columns 0-3 of B)
    W[:16, :rank4] = W4[:, :16]  # Only first 16 rows (for columns 0-3 of B)
    
    # Add schoolbook for columns 4, 5, 6
    col_offset = rank4
    for k_extra in range(3):
        k = 4 + k_extra  # Actual column index in B
        for i in range(4):
            for j in range(4):
                r = col_offset + k_extra * 16 + i * 4 + j
                U[i * 4 + j, r] = 1
                V[j * 7 + k, r] = 1
                W[k * 4 + i, r] = 1
    
    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), 4, 4, 7):
        return U.tolist(), V.tolist(), W.tolist(), rank_total
    
    return None, None, None, None

def _random_search(n, m, p, deadline, max_rank=100, min_rank=50, n_trials_per_rank=30):
    """Random search for low-rank decomposition."""
    rng = np.random.default_rng(42)
    best = None
    
    for rank in range(max_rank, min_rank - 1, -1):
        if time.time() > deadline:
            break
        for trial in range(n_trials_per_rank):
            if time.time() > deadline:
                break
            density = 0.2
            U = rng.standard_normal((n * m, rank)) * (rng.random((n * m, rank)) < density)
            V = rng.standard_normal((m * p, rank)) * (rng.random((m * p, rank)) < density)
            W = rng.standard_normal((p * n, rank)) * (rng.random((p * n, rank)) < density)
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                if best is None or rank < best[3]:
                    best = (U.tolist(), V.tolist(), W.tolist(), rank)
                break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try Kronecker/Strassen decomposition
    result = _build_4x4x7(deadline)
    if result[0] is not None:
        U, V, W, rank = result
        best_rank = rank

    # Try random search if we have time
    if time.time() < deadline - 20:
        result = _random_search(n, m, p, deadline)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
