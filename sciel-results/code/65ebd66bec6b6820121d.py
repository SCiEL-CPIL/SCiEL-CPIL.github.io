"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m + j, j*p + k, k*n + i] = 1 for all i, j, k, else 0.
    The convention is W[k*n + i, r].
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _decompose_tensor(T, rank, method='CP'):
    """Attempt to decompose T into rank factors.
    Returns (U, V, W) each of shape (dim, rank) as lists, or None on failure.
    """
    shape = T.shape
    # For a 3-way tensor of shape (I, J, K), a CP rank-r decomposition is:
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # Here I=n*m, J=m*p, K=p*n
    I, J, K = shape
    # Matrix unfoldings:
    # T_I (I x J*K): reshape T to (I, J*K)
    # T_J (J x I*K): reshape to (J, I*K) with appropriate transposition
    # T_K (K x I*J): reshape to (K, I*J)
    
    T_I = T.reshape(I, J * K)
    # For J unfolding: T_J[j, i*K + k] = T[i, j, k]
    T_J = np.zeros((J, I * K), dtype=np.int64)
    T_K = np.zeros((K, I * J), dtype=np.int64)
    for i in range(I):
        for j in range(J):
            for k in range(K):
                T_J[j, i * K + k] = T[i, j, k]
                T_K[k, i * J + j] = T[i, j, k]
    
    # Rank-r factorization via SVD of T_I:
    # U * (V * W)^T where (V*W) has shape (J*K, r)
    # This is a rank-r approximation, but we need exact.
    
    # Let's try: use SVD of T_I, take top r components, then check if the full tensor is exactly recovered.
    # For exact CP, we need T_i = U * (V ⊙ W)^T where ⊙ is Khatri-Rao.
    # This is hard. Let's try a different approach.
    
    # Use the fact that the schoolbook is a CP decomposition. For lower rank, we can try random rank-r factors and check.
    # But exact recovery by chance is unlikely.
    
    # Instead, let's try to solve the system of equations.
    # We have I*J*K equations, and 3*I*rank unknowns.
    # This is a polynomial system, hard to solve exactly.
    
    # For now, return the schoolbook decomposition which is always valid.
    return None

def _verify(U, V, W, T):
    """Check if U, V, W exactly decompose T."""
    I, J, K = T.shape
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.array_equal(reconstructed, T)

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130.0  # 20s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the tensor
    T = _build_tensor(n, m, p)
    
    # Schoolbook decomposition (rank = n*m*p = 48)
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    import numpy as np
    
    # Precompute tensor slices for efficient ALS updates
    # T[i*m+j, j*p+k, k*n+i] = 1
    # For fixed i: M_i[j,k] = T[i*m+j, j*p+k, k*n+i]
    # For fixed j: M_j[i,k] = T[i*m+j, j*p+k, k*n+i]
    # For fixed (k,i): M_w[j] = T[i*m+j, j*p+k, k*n+i]
    
    # Precompute all M_i, M_j, M_w
    M_i_list = []
    for i in range(n):
        M_i = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                M_i[j, k] = T[i * m + j, j * p + k, k * n + i]
        M_i_list.append(M_i)
    
    M_j_list = []
    for j in range(m):
        M_j = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                M_j[i, k] = T[i * m + j, j * p + k, k * n + i]
        M_j_list.append(M_j)
    
    M_w_list = {}
    for k in range(p):
        for i in range(n):
            M_w = np.zeros(m)
            for j in range(m):
                M_w[j] = T[i * m + j, j * p + k, k * n + i]
            M_w_list[(k, i)] = M_w
    
    # Try ranks from a reasonable lower bound up to schoolbook-1
    # Lower bound: at least max(n, m, p) * 2 = 12, but realistically higher
    # For 2x4x6, known results suggest rank around 30-40
    # Try a wider range with more restarts at promising ranks
    
    # Known lower bounds for matrix multiplication rank:
    # For (n,m,p) = (2,4,6), the rank is at least 2*4 + 4*6 + 6*2 - something
    # Actually, the Strassen-like lower bound is harder
    # Let's try ranks from 30 to 47
    
    start_rank = 30
    for rank in range(start_rank, rank_schoolbook):
        if time.time() > deadline:
            break
        
        # Use ALS (Alternating Least Squares) to try to find exact decomposition
        # Initialize with random values, then iterate
        
        best_solution = None
        n_restarts = 8  # More restarts for better coverage
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with different scales
            seed = 42 + restart * 1000 + rank * 7
            np.random.seed(seed)
            
            # Try different initialization strategies
            if restart % 3 == 0:
                # Small random
                U_curr = np.random.randn(n * m, rank) * 0.5
                V_curr = np.random.randn(m * p, rank) * 0.5
                W_curr = np.random.randn(p * n, rank) * 0.5
            elif restart % 3 == 1:
                # Sparse random with some structure
                U_curr = np.zeros((n * m, rank))
                V_curr = np.zeros((m * p, rank))
                W_curr = np.zeros((p * n, rank))
                for c in range(rank):
                    # Randomly select entries
                    for _ in range(3):
                        idx = np.random.randint(0, n * m)
                        U_curr[idx, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    for _ in range(3):
                        idx = np.random.randint(0, m * p)
                        V_curr[idx, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    for _ in range(3):
                        idx = np.random.randint(0, p * n)
                        W_curr[idx, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            else:
                # Integer-like initialization
                U_curr = np.random.randint(-2, 3, (n * m, rank)) * 0.5
                V_curr = np.random.randint(-2, 3, (m * p, rank)) * 0.5
                W_curr = np.random.randint(-2, 3, (p * n, rank)) * 0.5
            
            # ALS iterations
            n_iterations = 80
            for iteration in range(n_iterations):
                if time.time() > deadline:
                    break
                
                # Update U: For each i
                for i in range(n):
                    # D_i[j*p+k, c] = V_curr[j, c] * W_curr[k*n+i, c]
                    D_i = np.zeros((m * p, rank))
                    for j in range(m):
                        for k in range(p):
                            D_i[j * p + k, :] = V_curr[j, :] * W_curr[k * n + i, :]
                    
                    # Solve: M_i.flatten() = D_i.T @ U[i,:]
                    try:
                        U_curr[i * m + j, :] = np.linalg.lstsq(D_i.T, M_i_list[i].flatten(), rcond=None)[0]
                    except:
                        pass
                
                # Update V: For each j
                for j in range(m):
                    # D_j[i*p+k, c] = U_curr[i*m+j, c] * W_curr[k*n+i, c]
                    D_j = np.zeros((n * p, rank))
                    for i in range(n):
                        for k in range(p):
                            D_j[i * p + k, :] = U_curr[i * m + j, :] * W_curr[k * n + i, :]
                    
                    # Solve: M_j.flatten() = D_j.T @ V[j,:]
                    try:
                        V_curr[j, :] = np.linalg.lstsq(D_j.T, M_j_list[j].flatten(), rcond=None)[0]
                    except:
                        pass
                
                # Update W: For each (k,i) pair
                for k in range(p):
                    for i in range(n):
                        # D_w[j, c] = U_curr[i*m+j, c] * V_curr[j, c]
                        D_w = np.zeros((m, rank))
                        for j in range(m):
                            D_w[j, :] = U_curr[i * m + j, :] * V_curr[j, :]
                        
                        # Solve: M_w = D_w.T @ W[k*n+i,:]
                        try:
                            W_curr[k * n + i, :] = np.linalg.lstsq(D_w.T, M_w_list[(k, i)], rcond=None)[0]
                        except:
                            pass
                
                # Early termination check: if residual is very small, break
                if iteration % 10 == 0:
                    recon = np.einsum('ic,jc,kc->ijk', U_curr, V_curr, W_curr)
                    residual = np.max(np.abs(recon - T))
                    if residual < 1e-8:
                        break
            
            # Check if solution is exact
            reconstructed = np.einsum('ic,jc,kc->ijk', U_curr, V_curr, W_curr)
            if np.allclose(reconstructed, T, atol=1e-6):
                # Check if entries are close to integers or half-integers
                max_dev_int = max(np.max(np.abs(U_curr - np.round(U_curr))),
                                 np.max(np.abs(V_curr - np.round(V_curr))),
                                 np.max(np.abs(W_curr - np.round(W_curr))))
                max_dev_half = max(np.max(np.abs(2*U_curr - np.round(2*U_curr))),
                                  np.max(np.abs(2*V_curr - np.round(2*V_curr))),
                                  np.max(np.abs(2*W_curr - np.round(2*W_curr))))
                
                if max_dev_int < 1e-4:
                    # Convert to integers
                    U_int = np.round(U_curr).astype(int)
                    V_int = np.round(V_curr).astype(int)
                    W_int = np.round(W_curr).astype(int)
                    
                    # Verify exactness with integers
                    recon_int = np.einsum('ic,jc,kc->ijk', U_int, V_int, W_int)
                    if np.array_equal(recon_int, T):
                        best_solution = (U_int, V_int, W_int)
                        break
                elif max_dev_half < 1e-4:
                    # Convert to half-integers (multiply by 2, round, divide by 2)
                    U_half = np.round(2 * U_curr) / 2
                    V_half = np.round(2 * V_curr) / 2
                    W_half = np.round(2 * W_curr) / 2
                    
                    # Verify exactness with half-integers
                    recon_half = np.einsum('ic,jc,kc->ijk', U_half, V_half, W_half)
                    if np.allclose(recon_half, T, atol=1e-10):
                        # Store as integers by multiplying by 2
                        U_int = (2 * U_half).astype(int)
                        V_int = (2 * V_half).astype(int)
                        W_int = (2 * W_half).astype(int)
                        
                        # Verify
                        recon_int = np.einsum('ic,jc,kc->ijk', U_int, V_int, W_int)
                        # The tensor should be doubled if we used half-integers
                        # Actually, we need to check the convention
                        # If we store 2*value, then the product is 8*original
                        # So we need to check if recon_int == 8*T
                        if np.array_equal(recon_int, 8 * T):
                            best_solution = (U_int, V_int, W_int)
                            break
                        else:
                            # Try with the actual half-integers
                            U_list = U_half.tolist()
                            V_list = V_half.tolist()
                            W_list = W_half.tolist()
                            best_solution = (U_list, V_list, W_list)
                            break
        
        if best_solution is not None:
            U_best, V_best, W_best = best_solution
            best_rank = rank
            # Convert to lists if they're arrays
            if isinstance(U_best, np.ndarray):
                best_U = U_best.tolist()
                best_V = V_best.tolist()
                best_W = W_best.tolist()
            else:
                best_U = U_best
                best_V = V_best
                best_W = W_best
            break  # Found a solution, stop searching
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
