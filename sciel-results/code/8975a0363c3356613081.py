"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Construct Schoolbook Decomposition
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
                
    best_rank = rank_schoolbook
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def verify_decomposition(U, V, W, rank):
        """
        Verify that U, V, W exactly decompose the matrix multiplication tensor.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
        """
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    def compute_error(U, V, W, rank):
        """Compute the maximum absolute error from the target tensor (all ones)."""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.max(np.abs(T - 1.0))

    def compute_gradients(U, V, W, rank):
        """
        Compute gradients of the squared error w.r.t. U, V, W.
        Error = sum_{i,j,k} (T[i,j,k] - 1)^2 where T = einsum(U,V,W)
        """
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        residual = T - 1.0  # shape (n, m, p)
        
        # Gradient w.r.t. U[i,j,c] = sum_{k} residual[i,j,k] * V[j,k,c] * W[k,i,c]
        grad_U_r = np.einsum('ijk,jkc,kic->ijc', residual, V_r, W_r)
        # Gradient w.r.t. V[j,k,c] = sum_{i} residual[i,j,k] * U[i,j,c] * W[k,i,c]
        grad_V_r = np.einsum('ijk,ijc,kic->jkc', residual, U_r, W_r)
        # Gradient w.r.t. W[k,i,c] = sum_{j} residual[i,j,k] * U[i,j,c] * V[j,k,c]
        grad_W_r = np.einsum('ijk,ijc,jkc->kic', residual, U_r, V_r)
        
        grad_U = grad_U_r.reshape(n*m, rank)
        grad_V = grad_V_r.reshape(m*p, rank)
        grad_W = grad_W_r.reshape(p*n, rank)
        
        return grad_U, grad_V, grad_W

    def gradient_descent(U_init, V_init, W_init, target_rank, max_iter=200, lr=0.01):
        """
        Perform gradient descent to minimize squared error.
        """
        U_curr = U_init.copy()
        V_curr = V_init.copy()
        W_curr = W_init.copy()
        
        best_err = float('inf')
        best_state = (U_curr.copy(), V_curr.copy(), W_curr.copy())
        
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            err = compute_error(U_curr, V_curr, W_curr, target_rank)
            if err < best_err:
                best_err = err
                best_state = (U_curr.copy(), V_curr.copy(), W_curr.copy())
            
            if err < 1e-9:
                return U_curr, V_curr, W_curr, True
            
            grad_U, grad_V, grad_W = compute_gradients(U_curr, V_curr, W_curr, target_rank)
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if grad_norm < 1e-12:
                break
            
            lr_adapt = lr / (1 + grad_norm)
            
            U_curr -= lr_adapt * grad_U
            V_curr -= lr_adapt * grad_V
            W_curr -= lr_adapt * grad_W
        
        # Return best state found
        return best_state[0], best_state[1], best_state[2], best_err < 1e-9

    def try_hungarian_support(target_rank):
        """
        Try to find a sparse support pattern using matching-inspired heuristic.
        For each rank-1 term, assign it to cover specific (i,j,k) indices.
        """
        if time.time() > deadline - 30:
            return None
        
        # Create a cost matrix for assignment
        # Rows: n*m*p tensor entries (240 for 4x5x6... wait, 4*5*6=120)
        num_entries = n * m * p
        # Cols: target_rank terms
        
        # Heuristic: try to assign each tensor entry to one of the rank terms
        # such that each term covers a consistent pattern
        
        # Simple approach: group entries by their index structure
        # Term c can have non-zero U[i*m+j,c], V[j*p+k,c], W[k*n+i,c]
        # For a term to contribute to T[i,j,k], it needs all three to be non-zero
        
        # Try structured patterns:
        # Pattern 1: Terms indexed by (i, j, k) groups
        # Pattern 2: Terms indexed by (i, k) or (j, k) pairs
        
        # For now, try a few structured initializations
        patterns = []
        
        # Pattern A: First target_rank schoolbook columns
        U_A = U_school[:, :target_rank].copy()
        V_A = V_school[:, :target_rank].copy()
        W_A = W_school[:, :target_rank].copy()
        patterns.append((U_A, V_A, W_A))
        
        # Pattern B: Random selection of schoolbook columns
        if target_rank <= rank_schoolbook:
            cols = np.random.choice(rank_schoolbook, target_rank, replace=False)
            U_B = U_school[:, cols].copy()
            V_B = V_school[:, cols].copy()
            W_B = W_school[:, cols].copy()
            patterns.append((U_B, V_B, W_B))
        
        # Pattern C: Structured - use all i,j for some k values
        # Assign terms to cover (i,j) pairs with shared k
        if target_rank >= n * m:
            U_C = np.zeros((n * m, target_rank))
            V_C = np.zeros((m * p, target_rank))
            W_C = np.zeros((p * n, target_rank))
            
            col = 0
            for i in range(n):
                for j in range(m):
                    if col >= target_rank:
                        break
                    # This term covers (i,j) for all k
                    U_C[i * m + j, col] = 1.0
                    for k in range(p):
                        V_C[j * p + k, col] = 1.0
                        W_C[k * n + i, col] = 1.0
                    col += 1
            
            if col == target_rank:
                patterns.append((U_C, V_C, W_C))
        
        # Pattern D: Structured - use all j,k for some i values
        if target_rank >= m * p:
            U_D = np.zeros((n * m, target_rank))
            V_D = np.zeros((m * p, target_rank))
            W_D = np.zeros((p * n, target_rank))
            
            col = 0
            for j in range(m):
                for k in range(p):
                    if col >= target_rank:
                        break
                    # This term covers (j,k) for all i
                    for i in range(n):
                        U_D[i * m + j, col] = 1.0
                        W_D[k * n + i, col] = 1.0
                    V_D[j * p + k, col] = 1.0
                    col += 1
            
            if col == target_rank:
                patterns.append((U_D, V_D, W_D))
        
        # Try each pattern with gradient descent
        best_result = None
        best_err = float('inf')
        
        for U_init, V_init, W_init in patterns:
            if time.time() > deadline - 30:
                break
            
            U_opt, V_opt, W_opt, success = gradient_descent(
                U_init, V_init, W_init, target_rank, max_iter=100, lr=0.05
            )
            
            if success:
                if verify_decomposition(U_opt, V_opt, W_opt, target_rank):
                    return (U_opt, V_opt, W_opt)
            
            err = compute_error(U_opt, V_opt, W_opt, target_rank)
            if err < best_err:
                best_err = err
                best_result = (U_opt, V_opt, W_opt)
        
        return best_result

    # Try to find lower rank decompositions
    # Start from just below schoolbook and work down
    
    for target_rank in range(119, 95, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        
        # Try Hungarian-inspired support patterns
        result = try_hungarian_support(target_rank)
        if result is not None:
            U_opt, V_opt, W_opt = result
            if verify_decomposition(U_opt, V_opt, W_opt, target_rank):
                best_rank = target_rank
                best_U = U_opt.copy()
                best_V = V_opt.copy()
                best_W = W_opt.copy()
                found = True
                break
            else:
                # Try rounding to half-integers
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                if verify_decomposition(U_rounded, V_rounded, W_rounded, target_rank):
                    best_rank = target_rank
                    best_U = U_rounded.copy()
                    best_V = V_rounded.copy()
                    best_W = W_rounded.copy()
                    found = True
                    break
        
        # Try multiple random restarts with gradient descent
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            scale = 0.5
            U_init = np.random.uniform(-scale, scale, (n * m, target_rank))
            V_init = np.random.uniform(-scale, scale, (m * p, target_rank))
            W_init = np.random.uniform(-scale, scale, (p * n, target_rank))
            
            U_opt, V_opt, W_opt, success = gradient_descent(
                U_init, V_init, W_init, target_rank, max_iter=150, lr=0.02
            )
            
            if success:
                if verify_decomposition(U_opt, V_opt, W_opt, target_rank):
                    best_rank = target_rank
                    best_U = U_opt.copy()
                    best_V = V_opt.copy()
                    best_W = W_opt.copy()
                    found = True
                    break
                else:
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    if verify_decomposition(U_rounded, V_rounded, W_rounded, target_rank):
                        best_rank = target_rank
                        best_U = U_rounded.copy()
                        best_V = V_rounded.copy()
                        best_W = W_rounded.copy()
                        found = True
                        break
        
        if found:
            break
    
    # Final verification of the best result
    if not verify_decomposition(best_U, best_V, best_W, best_rank):
        # Fall back to schoolbook
        best_rank = rank_schoolbook
        best_U = U_school.copy()
        best_V = V_school.copy()
        best_W = W_school.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
