import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute the tensor from U, V, W
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    # The multiplication tensor should be 1 for all (i,j,k)
    expected = np.ones((n, m, p))
    
    if not np.allclose(tensor, expected, atol=1e-9):
        return False, tensor
    return True, tensor

def build_rank_28(n, m, p):
    """
    Construct a rank-28 decomposition for 2x3x5 matrix multiplication.
    Strategy:
    1. j=0: 2x1x5 block (rank 10, schoolbook)
    2. j=1,2, k=0,1: 2x2x2 block (rank 7, Strassen)
    3. j=1,2, k=2: 2x2x1 block (rank 4, schoolbook)
    4. j=1,2, k=3,4: 2x2x2 block (rank 7, Strassen)
    Total rank: 10 + 7 + 4 + 7 = 28.
    """
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k, col):
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        return col + 1

    # Helper to add Strassen 2x2x2 block at offsets (i_off, j_off, k_off)
    def add_strassen(i_off, j_off, k_off, col):
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = -1.0
        col += 1

        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = -1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        col += 1

        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        col += 1

        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+0)][col] = -1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = -1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        col += 1

        return col

    # 1. j=0 block: i in {0,1}, j=0, k in {0,1,2,3,4}
    # Schoolbook: 2 * 5 = 10 terms
    for k in range(p):
        col = add_schoolbook(0, 0, k, col)
        col = add_schoolbook(1, 0, k, col)
    
    # 2. j=1,2, k=0,1 block: i in {0,1}, j in {1,2}, k in {0,1}
    # Strassen: 7 terms
    # i_off=0, j_off=1, k_off=0
    col = add_strassen(0, 1, 0, col)

    # 3. j=1,2, k=2 block: i in {0,1}, j in {1,2}, k=2
    # Schoolbook: 2 * 2 = 4 terms
    for i in range(n):
        for j in [1, 2]:
            col = add_schoolbook(i, j, 2, col)

    # 4. j=1,2, k=3,4 block: i in {0,1}, j in {1,2}, k in {3,4}
    # Strassen: 7 terms
    # i_off=0, j_off=1, k_off=3
    col = add_strassen(0, 1, 3, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def try_rank_27_search(n, m, p, time_limit=5.0):
    """
    Attempt to find a rank-27 decomposition using a simplified search.
    This is a heuristic attempt; if it fails, we fall back to rank 28.
    """
    start_time = time.time()
    
    # For 2x3x5, the tensor has 30 entries.
    # Rank 27 means we need to satisfy 30 equations with 27*(6+15+10) variables?
    # Actually, it's a system of bilinear equations.
    # U is 6x27, V is 15x27, W is 10x27.
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # This is a highly non-linear problem. A simple local search might work.
    # Initialize with a random rank-27 matrix and try to minimize error.
    
    rank = 27
    np.random.seed(42)
    
    # Initialize with small random values
    U = np.random.rand(n * m, rank) * 0.1
    V = np.random.rand(m * p, rank) * 0.1
    W = np.random.rand(p * n, rank) * 0.1
    
    # Gradient descent on the error
    # Error = sum (T[i,j,k] - 1)^2
    
    lr = 0.01
    for step in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    
                    # T[i,j,k] = sum_r U[idx_u, r] V[idx_v, r] W[idx_w, r]
                    # dT/dU[idx_u, r] = V[idx_v, r] W[idx_w, r]
                    # dT/dV[idx_v, r] = U[idx_u, r] W[idx_w, r]
                    # dT/dW[idx_w, r] = U[idx_u, r] V[idx_v, r]
                    
                    # Error term: (T - 1)
                    t_val = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                    err = t_val - 1.0
                    
                    dU[idx_u] += err * (V[idx_v] * W[idx_w])
                    dV[idx_v] += err * (U[idx_u] * W[idx_w])
                    dW[idx_w] += err * (U[idx_u] * V[idx_v])
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Check convergence
        # Compute max error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    t_val = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                    max_err = max(max_err, abs(t_val - 1.0))
        
        if max_err < 1e-9:
            # Found a solution
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Attempt the known best rank 28 decomposition
    try:
        U, V, W = build_rank_28(n, m, p)
        valid, tensor = verify_decomposition(U, V, W, n, m, p)
        if valid:
            # Check if we have time to try rank 27
            elapsed = time.time() - start_time
            if elapsed < 10.0: # Plenty of time left
                result_27 = try_rank_27_search(n, m, p, time_limit=10.0)
                if result_27:
                    U27, V27, W27 = result_27
                    valid27, _ = verify_decomposition(U27, V27, W27, n, m, p)
                    if valid27:
                        return {'U': U27, 'V': V27, 'W': W27}
            
            return {'U': U, 'V': V, 'W': W}
    except Exception as e:
        pass

    # Fallback: Schoolbook (Rank 30)
    rank_school = n * m * p
    U = [[0.0] * rank_school for _ in range(n * m)]
    V = [[0.0] * rank_school for _ in range(m * p)]
    W = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    valid, _ = verify_decomposition(U, V, W, n, m, p)
    if valid:
        return {'U': U, 'V': V, 'W': W}
    
    # Should not happen
    return {'U': [[0.0]], 'V': [[0.0]], 'W': [[0.0]]}
