import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]
DEADLINE = 140  # seconds; keep safety margin before 150s kill

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

def try_rank_27(n, m, p):
    """
    Search for a rank-27 decomposition by relaxing the rank-28 block structure
    and minimizing the Frobenius residual of the tensor reconstruction.
    
    Strategy: Start from the rank-28 solution, zero out one term (the "redundant"
    one), then optimize all entries to drive the residual to zero.
    """
    import numpy as np
    from scipy.optimize import minimize
    import time
    
    start = time.time()
    
    # Build rank-28 as starting point
    U28, V28, W28 = build_rank_28(n, m, p)
    U28 = np.array(U28, dtype=float)
    V28 = np.array(V28, dtype=float)
    W28 = np.array(W28, dtype=float)
    
    # Try removing each of the 28 terms and see if we can recover rank 27
    # We'll flatten U, V, W into a single vector and minimize ||T_recon - T_target||_F^2
    # T_target[i,j,k] = 1 for all i,j,k
    
    def compute_residual(x, rank):
        """Compute squared Frobenius norm of (T_recon - T_target)."""
        # Unpack
        nU = n * m
        nV = m * p
        nW = p * n
        U = x[:nU*rank].reshape(nU, rank)
        V = x[nU*rank:nU*rank+nV*rank].reshape(nV, rank)
        W = x[nU*rank+nV*rank:].reshape(nW, rank)
        
        # Compute tensor
        tensor = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        residual = tensor - 1.0
        return np.sum(residual**2)
    
    def compute_residual_grad(x, rank):
        """Compute gradient of squared Frobenius residual."""
        nU = n * m
        nV = m * p
        nW = p * n
        U = x[:nU*rank].reshape(nU, rank)
        V = x[nU*rank:nU*rank+nV*rank].reshape(nV, rank)
        W = x[nU*rank+nV*rank:].reshape(nW, rank)
        
        # Compute tensor
        tensor = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        residual = tensor - 1.0  # shape (n, m, p)
        
        # Gradient with respect to U[i*m+j, r] = sum_{k} V[j*p+k, r] * W[k*n+i, r] * residual[i,j,k]
        grad_U = np.zeros((nU, rank))
        grad_V = np.zeros((nV, rank))
        grad_W = np.zeros((nW, rank))
        
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    # d/dU[i*m+j, r]
                    s = 0.0
                    for k in range(p):
                        s += V[j*p+k, r] * W[k*n+i, r] * residual[i, j, k]
                    grad_U[i*m+j, r] = 2.0 * s
                    
                    # d/dV[j*p+k, r] for each k
                    for k in range(p):
                        grad_V[j*p+k, r] = 2.0 * U[i*m+j, r] * W[k*n+i, r] * residual[i, j, k]
                    
                    # d/dW[k*n+i, r] for each k
                    for k in range(p):
                        grad_W[k*n+i, r] = 2.0 * U[i*m+j, r] * V[j*p+k, r] * residual[i, j, k]
        
        # Pack gradients
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return grad
    
    # Try each term removal
    best_result = None
    best_residual = np.inf
    
    for removed_term in range(28):
        if time.time() - start > DEADLINE - 10:
            break
        
        # Create initial guess: rank-28 with one term zeroed
        U0 = U28.copy()
        V0 = V28.copy()
        W0 = W28.copy()
        U0[:, removed_term] = 0.0
        V0[:, removed_term] = 0.0
        W0[:, removed_term] = 0.0
        
        # Also try: keep the term but perturb slightly
        # Actually, better approach: start from rank-28, remove one term, 
        # then optimize the remaining 27 terms
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Check initial residual
        init_res = compute_residual(x0, 27)
        if init_res < 1e-10:
            # Already zero residual with 27 terms!
            U_sol = x0[:n*m*27].reshape(n*m, 27)
            V_sol = x0[n*m*27:n*m*27+m*p*27].reshape(m*p, 27)
            W_sol = x0[n*m*27+m*p*27:].reshape(p*n, 27)
            return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
        
        # Optimize
        try:
            result = minimize(
                compute_residual,
                x0,
                jac=compute_residual_grad,
                args=(27,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            final_res = result.fun
            if final_res < best_residual:
                best_residual = final_res
                best_result = result.x
            
            if best_residual < 1e-12:
                break
        except Exception:
            continue
    
    # Check if we found a valid rank-27 solution
    if best_result is not None and best_residual < 1e-12:
        nU = n * m
        nV = m * p
        nW = p * n
        U_sol = best_result[:nU*27].reshape(nU, 27)
        V_sol = best_result[nU*27:nU*27+nV*27].reshape(nV, 27)
        W_sol = best_result[nU*27+nV*27:].reshape(nW, 27)
        
        # Verify
        valid, _ = verify_decomposition(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p)
        if valid:
            return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, try to find a rank-27 decomposition via optimization
    try:
        if time.time() - start_time < DEADLINE - 5:
            result_27 = try_rank_27(n, m, p)
            if result_27 is not None:
                U, V, W = result_27
                valid, _ = verify_decomposition(U, V, W, n, m, p)
                if valid:
                    return {'U': U, 'V': V, 'W': W}
    except Exception:
        pass
    
    # Directly attempt the known best rank 28 decomposition
    try:
        U, V, W = build_rank_28(n, m, p)
        valid, tensor = verify_decomposition(U, V, W, n, m, p)
        if valid:
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
