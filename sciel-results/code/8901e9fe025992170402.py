import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as guaranteed fallback
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook.copy()
    best_V = V_schoolbook.copy()
    best_W = W_schoolbook.copy()
    
    def is_half_integer_matrix(M, tol=1e-8):
        """Check if all entries are integers or half-integers."""
        return np.allclose(M * 2, np.round(M * 2), atol=tol)

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W decompose the matrix multiplication tensor and are half-integers."""
        if not is_half_integer_matrix(U, tol) or not is_half_integer_matrix(V, tol) or not is_half_integer_matrix(W, tol):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1) > tol:
                        return False
        return True

    def schwartz_zippel_test(U, V, W, num_trials=10, tol=1e-6):
        """
        Polynomial identity testing via Schwartz-Zippel:
        Encode the matrix multiplication tensor constraints as a polynomial system.
        For each (i,j,k), the constraint is:
            sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        We test if the polynomial P(x) = prod over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)
        is identically zero by evaluating at random points.
        
        However, since U, V, W are fixed matrices (not variables), we directly verify
        the bilinear constraints. The Schwartz-Zippel framework here is used to
        efficiently check if a candidate decomposition satisfies all constraints
        by treating it as a polynomial identity in the rank parameter.
        
        For fixed U, V, W, we simply verify all n*m*p constraints exactly.
        This is O(n*m*p*rank) which is fast for our sizes.
        """
        if not is_half_integer_matrix(U, tol) or not is_half_integer_matrix(V, tol) or not is_half_integer_matrix(W, tol):
            return False
        
        rank = U.shape[1]
        
        # Vectorized check of all constraints
        # For each (i,j,k): dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:]) == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j, :]
                    v_row = V[j * p + k, :]
                    w_row = W[k * n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def polish_decomposition(U, V, W, max_iter=200, tol=1e-14):
        """
        High-precision polish pass: refine U, V, W to satisfy constraints exactly
        using alternating least-squares with tighter tolerance, then round to half-integers.
        """
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            rank = U.shape[1]
            
            # Solve for W given U, V
            num_eqs = n * p * m
            A_W = np.zeros((num_eqs, p * n * rank))
            b_W = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(rank):
                            A_W[eq_idx, (k * n + i) * rank + c] = U[i * m + j, c] * V[j * p + k, c]
                        eq_idx += 1
            
            W_sol, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W = W_sol.reshape(p * n, rank)
            
            # Solve for U given V, W
            A_U = np.zeros((num_eqs, n * m * rank))
            b_U = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(rank):
                            A_U[eq_idx, (i * m + j) * rank + c] = V[j * p + k, c] * W[k * n + i, c]
                        eq_idx += 1
            
            U_sol, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U = U_sol.reshape(n * m, rank)
            
            # Solve for V given U, W
            A_V = np.zeros((num_eqs, m * p * rank))
            b_V = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(rank):
                            A_V[eq_idx, (j * p + k) * rank + c] = U[i * m + j, c] * W[k * n + i, c]
                        eq_idx += 1
            
            V_sol, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V = V_sol.reshape(m * p, rank)
            
            # Check convergence with very tight tolerance
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < tol:
                break
        
        return U, V, W, max_err
    
    def alternating_minimize(U, V, W, max_iter=150, tol=1e-12):
        """Perform alternating least-squares minimization with tighter tolerance."""
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            rank = len(U[0])
            
            # Solve for W given U, V
            num_eqs = n * p * m  # 72
            A_W = np.zeros((num_eqs, p * n * rank))
            b_W = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(rank):
                            A_W[eq_idx, (k * n + i) * rank + c] = U[i * m + j, c] * V[j * p + k, c]
                        eq_idx += 1
            
            W_sol, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W = W_sol.reshape(p * n, rank)
            
            # Solve for U given V, W
            A_U = np.zeros((num_eqs, n * m * rank))
            b_U = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(rank):
                            A_U[eq_idx, (i * m + j) * rank + c] = V[j * p + k, c] * W[k * n + i, c]
                        eq_idx += 1
            
            U_sol, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U = U_sol.reshape(n * m, rank)
            
            # Solve for V given U, W
            A_V = np.zeros((num_eqs, m * p * rank))
            b_V = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(rank):
                            A_V[eq_idx, (j * p + k) * rank + c] = U[i * m + j, c] * W[k * n + i, c]
                        eq_idx += 1
            
            V_sol, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V = V_sol.reshape(m * p, rank)
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        error += (val - 1) ** 2
            error = np.sqrt(error)
            
            if error < tol:
                break
        
        return U, V, W, error
    
    # Try target ranks from high to low
    # Focus on ranks that might be achievable
    target_ranks = [64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 25:
            break
        
        found = False
        
        # Try multiple initializations
        for init_type in range(8):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(init_type * 1000 + target_rank * 13 + 7)
            
            # Different initialization strategies
            if init_type == 0:
                # Random initialization
                U = np.random.randn(n * m, target_rank) * 0.1
                V = np.random.randn(m * p, target_rank) * 0.1
                W = np.random.randn(p * n, target_rank) * 0.1
            elif init_type == 1:
                # Block-structured: initialize based on schoolbook pattern
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                # Spread schoolbook pattern across target_rank
                for c in range(min(target_rank, n * m * p)):
                    i = (c // (m * p)) % n
                    j = (c // p) % m
                    k = c % p
                    U[i * m + j, c % target_rank] = 1.0 / np.sqrt(target_rank)
                    V[j * p + k, c % target_rank] = 1.0 / np.sqrt(target_rank)
                    W[k * n + i, c % target_rank] = 1.0 / np.sqrt(target_rank)
            elif init_type == 2:
                # Kronecker-like structure
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for c in range(target_rank):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    if c < n * m * p:
                        U[i * m + j, c] = 1.0
                        V[j * p + k, c] = 1.0
                        W[k * n + i, c] = 1.0
            elif init_type == 3:
                # Sparse random
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for c in range(target_rank):
                    for _ in range(3):
                        U[np.random.randint(n * m), c] = np.random.choice([-1, 1])
                        V[np.random.randint(m * p), c] = np.random.choice([-1, 1])
                        W[np.random.randint(p * n), c] = np.random.choice([-1, 1])
            elif init_type == 4:
                # Mix of structured and random
                U = np.random.randn(n * m, target_rank) * 0.05
                V = np.random.randn(m * p, target_rank) * 0.05
                W = np.random.randn(p * n, target_rank) * 0.05
                # Add some structure
                for c in range(min(target_rank, 12)):
                    U[c % (n * m), c] += 1.0
                    V[c % (m * p), c] += 1.0
                    W[c % (p * n), c] += 1.0
            elif init_type == 5:
                # Integer-valued initialization
                U = np.random.randint(-1, 2, (n * m, target_rank)).astype(float)
                V = np.random.randint(-1, 2, (m * p, target_rank)).astype(float)
                W = np.random.randint(-1, 2, (p * n, target_rank)).astype(float)
            elif init_type == 6:
                # Half-integer valued initialization
                U = np.random.randint(-2, 3, (n * m, target_rank)).astype(float) / 2.0
                V = np.random.randint(-2, 3, (m * p, target_rank)).astype(float) / 2.0
                W = np.random.randint(-2, 3, (p * n, target_rank)).astype(float) / 2.0
            else:
                # Sparse structured: each rank component activates a small subset
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for c in range(target_rank):
                    # Pick a random (i,j) for U
                    idx_u = np.random.randint(n * m)
                    U[idx_u, c] = np.random.choice([-1.0, 1.0])
                    # Pick a random (j,k) for V consistent with U's j
                    j_idx = idx_u % m
                    idx_v = j_idx * p + np.random.randint(p)
                    V[idx_v, c] = np.random.choice([-1.0, 1.0])
                    # Pick a random (k,i) for W
                    k_idx = idx_v % p
                    idx_w = k_idx * n + idx_u // m
                    W[idx_w, c] = np.random.choice([-1.0, 1.0])
            
            # Run alternating minimization
            U_new, V_new, W_new, error = alternating_minimize(U, V, W, max_iter=100)
            
            if error < 1e-6:
                # Try rounding to integers or half-integers
                U_r = np.round(U_new * 2) / 2
                V_r = np.round(V_new * 2) / 2
                W_r = np.round(W_new * 2) / 2
                
                # Use Schwartz-Zippel-inspired fast verification
                if schwartz_zippel_test(U_r, V_r, W_r):
                    # Apply high-precision polish
                    U_pol, V_pol, W_pol, polish_err = polish_decomposition(U_r, V_r, W_r)
                    U_pol_r = np.round(U_pol * 2) / 2
                    V_pol_r = np.round(V_pol * 2) / 2
                    W_pol_r = np.round(W_pol * 2) / 2
                    
                    if schwartz_zippel_test(U_pol_r, V_pol_r, W_pol_r):
                        best_rank = target_rank
                        best_U = U_pol_r.copy()
                        best_V = V_pol_r.copy()
                        best_W = W_pol_r.copy()
                        found = True
                        break
                    elif check_decomposition(U_r, V_r, W_r):
                        best_rank = target_rank
                        best_U = U_r.copy()
                        best_V = V_r.copy()
                        best_W = W_r.copy()
                        found = True
                        break
        
        if found:
            break
    
    # Final verification
    if not check_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U = U_schoolbook
        best_V = V_schoolbook
        best_W = W_schoolbook
        best_rank = rank_schoolbook
    
    # Convert to lists, ensuring they are exactly half-integers
    U_list = [[float(x) for x in row] for row in np.round(best_U * 2) / 2]
    V_list = [[float(x) for x in row] for row in np.round(best_V * 2) / 2]
    W_list = [[float(x) for x in row] for row in np.round(best_W * 2) / 2]
    
    return {"U": U_list, "V": V_list, "W": W_list}
