"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # The matrix multiplication tensor T[i,j,k] where:
    # i indexes A (n*m = 9), j indexes B (m*p = 18), k indexes C (p*n = 18)
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[k]
    # A row i//m, col i%m; B row j//p, col j%p
    # C row j%p, col i//m (using W[k*n+i] convention where k = j%p, i = i//m)
    # W[k*n+i, r] where output is C[row, col]
    # C[row, col] = sum_j A[row, j] * B[j, col]
    # row = 0..n-1, col = 0..p-1
    # A index: i = row * m + j  (row-major for A)
    # B index: j * p + col
    # C index: W[k*n + i] where k = col, i = row
    
    # Build the tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for row in range(n):
        for col in range(p):
            for j in range(m):
                a_idx = row * m + j
                b_idx = j * p + col
                c_idx = col * n + row  # W[k*n+i] with k=col, i=row
                T[a_idx, b_idx, c_idx] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U @ diag @ V @ diag @ W reconstructs T"""
        T_approx = np.einsum('is,js,ks->ijk', U, V, W)
        return np.max(np.abs(T_approx - T)) < tol
    
    def generate_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook
    U, V, W, schoolbook_rank = generate_schoolbook()
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), schoolbook_rank
    
    # Strategy: Use Strassen-Winograd for 3x3x3 (rank 23) and decompose 3x3x6
    # into two independent 3x3x3 multiplications
    # B is 3x6, so we can view it as two 3x3 blocks: B_left (cols 0-2) and B_right (cols 3-5)
    # C is 3x6, so C_left = A * B_left and C_right = A * B_right
    # Each 3x3x3 multiplication has a known rank-23 decomposition (Strassen-Winograd)
    
    # Let's try to find a rank-23 decomposition for 3x3x3 first
    # Then combine two such decompositions for 3x3x6
    
    # For 3x3x3, the tensor T_3[i,j,k] where i in 0..8, j in 0..8, k in 0..8
    # T_3[i,j,k] = 1 if A[row, col_a] * B[col_a, col_b] contributes to C[row, col_b]
    # where i = row*3 + col_a, j = col_a*3 + col_b, k = col_b*3 + row
    
    # Let's use a known Strassen-Winograd decomposition for 3x3x3
    # The Winograd algorithm for 3x3 convolution/matrix multiplication uses rank 23
    
    # Actually, let's use a more systematic approach:
    # Try to find low-rank decompositions using optimization
    
    def optimize_decomposition(rank, max_iter=5000, tol=1e-8, seed=42):
        """Try to find a rank-r decomposition using iterative optimization"""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Use Adam-like optimization with momentum
        lr = 0.05
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        t = 0
        best_err = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                return None
            
            t += 1
            
            # Compute gradient
            T_approx = np.einsum('is,js,ks->ijk', U, V, W)
            err = T - T_approx
            
            grad_U = np.einsum('ijk,js,ks->is', err, V, W)
            grad_V = np.einsum('ijk,is,ks->js', err, U, W)
            grad_W = np.einsum('ijk,is,js->ks', err, U, V)
            
            # Update momentum
            m_U = beta1 * m_U + (1 - beta1) * grad_U
            v_U = beta2 * v_U + (1 - beta2) * grad_U**2
            m_hat_U = m_U / (1 - beta1**t)
            v_hat_U = v_U / (1 - beta2**t)
            
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V**2
            m_hat_V = m_V / (1 - beta1**t)
            v_hat_V = v_V / (1 - beta2**t)
            
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W**2
            m_hat_W = m_W / (1 - beta1**t)
            v_hat_W = v_W / (1 - beta2**t)
            
            # Update parameters
            U -= lr * m_hat_U / (np.sqrt(v_hat_U) + eps)
            V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
            W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
            
            # Check convergence periodically
            if it % 100 == 0:
                max_err = np.max(np.abs(T_approx - T))
                if max_err < best_err:
                    best_err = max_err
                if max_err < tol:
                    break
        
        # Final check
        T_approx = np.einsum('is,js,ks->ijk', U, V, W)
        max_err = np.max(np.abs(T_approx - T))
        
        if max_err < 1e-4:
            # Clean up small values
            U_clean = np.where(np.abs(U) < 1e-6, 0, U)
            V_clean = np.where(np.abs(V) < 1e-6, 0, V)
            W_clean = np.where(np.abs(W) < 1e-6, 0, W)
            
            T_clean = np.einsum('is,js,ks->ijk', U_clean, V_clean, W_clean)
            if np.max(np.abs(T_clean - T)) < 1e-3:
                return U_clean, V_clean, W_clean, max_err
        return None
    
    # Try ranks from 30 up to 54
    for rank in range(30, 55):
        if time.time() > deadline - 20:
            break
        
        # Try multiple seeds for this rank
        for seed in [42, 123, 456]:
            if time.time() > deadline - 20:
                break
            
            result = optimize_decomposition(rank, max_iter=3000, seed=seed)
            if result is not None:
                U_new, V_new, W_new, err = result
                if err < 1e-4:
                    best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank
                    break  # Found a valid lower rank
        
        if best_rank < 54:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
