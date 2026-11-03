"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140.0
    
    # Try rank 20 decomposition using structured ansatz
    # U: (n*m, rank) = (9, r)
    # V: (m*p, rank) = (12, r)  
    # W: (p*n, rank) = (12, r)
    
    # The constraint: for all i in [0,n), j in [0,m), k in [0,p):
    #   1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Use a Winograd-style structured ansatz
    # For 3x3x4, try rank 20
    
    best_result = None
    
    # Approach: Use the known structure. 
    # Let's try to build it from the 3x3x3 rank-23 and extend, 
    # or use a direct parameterization.
    
    # Direct approach: assume U, V have sparse structure
    # Try: U entries in {-1, 0, 1}, V entries in {-1, 0, 1}, W entries in {-1, 0, 1}
    # with specific sparsity patterns.
    
    # Actually, let me try a numerical solve with initialization.
    # The system is: for each (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # This is a tensor decomposition problem. Let me use an iterative approach.
    
    # Initialize with schoolbook and try to reduce rank
    # For rank r, we need to solve a nonlinear system.
    
    # Let me try rank 20 with a specific structured initialization.
    
    # Structure: split into groups
    # Group 1: 9 terms for "main" 3x3 part (columns 0,1,2)
    # Group 2: 11 terms for column 3 and corrections
    
    # Actually, let me try a different approach: 
    # Use the fact that B[j,3] can be written as a combination.
    
    # Let me try to construct explicitly.
    # For C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Winograd-style for 3x3x4:
    # Let's define intermediate products.
    
    # Try rank 20 with the following structure:
    # c = 0..8: handle (i,j) pairs with k=0,1,2 in a compressed way
    # c = 9..19: handle k=3 and cross terms
    
    # This is complex. Let me use a numerical solver.
    
    r = 20
    rng = np.random.RandomState(42)
    
    # Initialize U, V, W
    # U: 9 x r, V: 12 x r, W: 12 x r
    U = rng.randn(n*m, r) * 0.5
    V = rng.randn(m*p, r) * 0.5
    W = rng.randn(p*n, r) * 0.5
    
    # The loss: sum over all (i,j,k) of (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
    # This is a non-convex problem. Use gradient descent or similar.
    
    # Actually, for tensor decomposition, a better approach is alternating least squares
    # or using the fact that if we fix U and V, W is determined linearly.
    
    # For fixed U and V, the constraint for each (i,j,k) is:
    # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
    # This is linear in W! For each i,k, we have m equations (one for each j) in r unknowns W[k*n+i, :].
    
    # Wait, for fixed i,k, the equations are: for each j, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # That's m=3 linear equations in r=20 unknowns for W[k*n+i,:].
    # Underdetermined! So we can solve it.
    
    # Similarly, for fixed U and W, V is determined, and for fixed V and W, U is determined.
    
    # Let's use alternating minimization:
    # 1. Fix V, W -> solve for U (linear)
    # 2. Fix U, W -> solve for V (linear)
    # 3. Fix U, V -> solve for W (linear)
    
    # For U: for each (i,j), the constraint involves all k:
    # For each k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is p=4 linear equations in r=20 unknowns U[i*m+j,:].
    # Underdetermined, use least squares.
    
    # Let me implement this.
    
    def solve_for_U(V, W, n, m, p, r):
        """For each (i,j), solve linear system for U[i*m+j, :]"""
        U_sol = np.zeros((n*m, r))
        for i in range(n):
            for j in range(m):
                # For each k, equation: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Let a_c = V[j*p+k,c] * W[k*n+i,c]
                # Then sum_c U[i*m+j,c] * a_c = 1 for each k
                # Matrix A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                # Solve A @ U[i*m+j,:] = b (underdetermined, use lstsq)
                sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                U_sol[i*m+j, :] = sol
        return U_sol
    
    def solve_for_V(U, W, n, m, p, r):
        """For each (j,k), solve linear system for V[j*p+k, :]"""
        V_sol = np.zeros((m*p, r))
        for j in range(m):
            for k in range(p):
                # For each i, equation: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Let a_c = U[i*m+j,c] * W[k*n+i,c]
                # Then sum_c V[j*p+k,c] * a_c = 1 for each i
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                V_sol[j*p+k, :] = sol
        return V_sol
    
    def solve_for_W(U, V, n, m, p, r):
        """For each (k,i), solve linear system for W[k*n+i, :]"""
        W_sol = np.zeros((p*n, r))
        for k in range(p):
            for i in range(n):
                # For each j, equation: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Let a_c = U[i*m+j,c] * V[j*p+k,c]
                # Then sum_c W[k*n+i,c] * a_c = 1 for each j
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                W_sol[k*n+i, :] = sol
        return W_sol
    
    def compute_loss(U, V, W, n, m, p):
        """Compute the residual loss"""
        loss = 0.0
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    loss += (1.0 - val) ** 2
        return loss
    
    # Alternating minimization
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    best_loss = float('inf')
    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
    
    max_iters = 500
    for it in range(max_iters):
        if time.time() > deadline - 10.0:
            break
        
        U_cur = solve_for_U(V_cur, W_cur, n, m, p, r)
        V_cur = solve_for_V(U_cur, W_cur, n, m, p, r)
        W_cur = solve_for_W(U_cur, V_cur, n, m, p, r)
        
        loss = compute_loss(U_cur, V_cur, W_cur, n, m, p)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
        
        if loss < 1e-10:
            break
    
    # Now try to round to half-integers
    # Scale and round
    def round_to_half_int(mat):
        return np.round(mat * 2.0) / 2.0
    
    U_rounded = round_to_half_int(best_U)
    V_rounded = round_to_half_int(best_V)
    W_rounded = round_to_half_int(best_W)
    
    # Check if rounded version works
    def check_decomposition(U, V, W, n, m, p, tol=1e-6):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
        # Convert to lists
        return {
            "U": U_rounded.tolist(),
            "V": V_rounded.tolist(),
            "W": W_rounded.tolist()
        }
    
    # If rounding failed, try to find a better solution or use the continuous solution
    # But the problem requires integer/half-integer entries.
    # Let me try more iterations or different initialization.
    
    # Try multiple random initializations
    for seed in range(10):
        if time.time() > deadline - 5.0:
            break
        rng = np.random.RandomState(seed)
        U_init = rng.randn(n*m, r) * 0.5
        V_init = rng.randn(m*p, r) * 0.5
        W_init = rng.randn(p*n, r) * 0.5
        
        U_c, V_c, W_c = U_init.copy(), V_init.copy(), W_init.copy()
        best_loss_s = float('inf')
        best_U_s, best_V_s, best_W_s = U_c.copy(), V_c.copy(), W_c.copy()
        
        for it in range(200):
            if time.time() > deadline - 5.0:
                break
            U_c = solve_for_U(V_c, W_c, n, m, p, r)
            V_c = solve_for_V(U_c, W_c, n, m, p, r)
            W_c = solve_for_W(U_c, V_c, n, m, p, r)
            
            loss = compute_loss(U_c, V_c, W_c, n, m, p)
            if loss < best_loss_s:
                best_loss_s = loss
                best_U_s, best_V_s, best_W_s = U_c.copy(), V_c.copy(), W_c.copy()
            if loss < 1e-10:
                break
        
        if best_loss_s < best_loss:
            best_loss = best_loss_s
            best_U, best_V, best_W = best_U_s, best_V_s, best_W_s
        
        U_r = round_to_half_int(best_U)
        V_r = round_to_half_int(best_V)
        W_r = round_to_half_int(best_W)
        
        if check_decomposition(U_r, V_r, W_r, n, m, p):
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
    
    # If all else fails, fall back to schoolbook
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
