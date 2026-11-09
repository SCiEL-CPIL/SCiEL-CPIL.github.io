"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Attempt to find a lower rank decomposition (e.g., rank 29)
    # using gradient descent on the real numbers, then snapping to half-integers.
    
    def try_rank(rank):
        if time.time() > deadline - 10:
            return None
            
        # Initialize U, V, W with random values
        # U: (n*m) x rank
        # V: (m*p) x rank
        # W: (p*n) x rank
        
        np.random.seed(42)
        
        # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
        # Let's initialize with small random values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Define the loss function
        def loss_fn(params):
            # params is a flat vector
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank:n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n*m, rank)
            V = V_flat.reshape(m*p, rank)
            W = W_flat.reshape(p*n, rank)
            
            total_loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Sum over c
                        # U index: i*m + j
                        # V index: j*p + k
                        # W index: k*n + i
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        total_loss += (val - 1.0) ** 2
            return total_loss
        
        # Gradient descent
        # Number of parameters: (n*m + m*p + p*n) * rank
        num_params = (n*m + m*p + p*n) * rank
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        lr = 0.01
        for step in range(5000):
            if time.time() > deadline - 5:
                break
                
            # Compute gradient numerically or analytically?
            # Numerical gradient is slow. Let's try a simple coordinate descent or just a few steps of GD.
            # Given the complexity, let's try a different approach:
            # Fix U and V, solve for W?
            # The equations are linear in W for fixed U, V?
            # No, it's U*V*W. It's linear in W if U*V is fixed.
            # Let M_{ik}^{j} = sum_c U_{i,j,c} V_{j,k,c}.
            # Then sum_c M_{ik}^{j} W_{k,i,c} = 1.
            # This is linear in W!
            
            # So we can alternate:
            # 1. Fix U, V. Solve linear system for W.
            # 2. Fix V, W. Solve linear system for U.
            # 3. Fix W, U. Solve linear system for V.
            
            # Let's implement this alternating least squares (ALS) approach.
            
            # 1. Solve for W
            # Equations: for each i, k:
            # sum_j sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
            # Wait, the sum is over j? No, the equation is for each (i,j,k).
            # The condition is: for each i, j, k:
            # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
            
            # This is NOT linear in W alone because U and V depend on c.
            # But for a fixed c, it's a product.
            # Let's rewrite:
            # For each c, let u_c be the vector of U entries, v_c be V, w_c be W.
            # The equation is sum_c u_c[i,j] v_c[j,k] w_c[k,i] = 1.
            
            # This is a tensor decomposition problem.
            # ALS for tensor decomposition:
            # Fix U, V. The equation for W is:
            # For each i, k:
            # sum_j (sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]) = 1 ? No.
            # The sum is over c, not j.
            # The equation is for each triple (i,j,k).
            # So we have 30 equations.
            # Variables are W entries.
            # Let's group equations by (k, i).
            # For a fixed (k, i), we have equations for j=0..m-1.
            # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for each j.
            
            # Let A_{j, c} = U[i*m+j, c] * V[j*p+k, c].
            # Then sum_c A_{j, c} W_{k*n+i, c} = 1.
            # This is a linear system in W_{k*n+i, :}.
            # We have m equations (for j=0..m-1) and rank unknowns (W_{k*n+i, 0}...W_{k*n+i, rank-1}).
            # If rank > m, this is underdetermined. We can use least squares.
            
            # Let's try this ALS approach.
            
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            for iter in range(100):
                if time.time() > deadline - 5:
                    break
                    
                # Update W
                for k in range(p):
                    for i in range(n):
                        # Solve for W[k*n+i, :]
                        # Equations for j in 0..m-1:
                        # sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        # Solve A @ w = 1
                        # Least squares
                        w, residuals, rank_lsq, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                        
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Solve for U[i*m+j, :]
                        # Equations for k in 0..p-1:
                        # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        u, residuals, rank_lsq, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        # Solve for V[j*p+k, :]
                        # Equations for i in 0..n-1:
                        # sum_c (U[i*m+j, c] * W[k*n+i, c]) * V[j*p+k, c] = 1
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        v, residuals, rank_lsq, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                        
            # Check loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        loss += (val - 1.0) ** 2
                        
            if loss < 1e-6:
                # Snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify snapped solution
                loss_snapped = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_snapped[i*m+j, :] * V_snapped[j*p+k, :] * W_snapped[k*n+i, :])
                            loss_snapped += (val - 1.0) ** 2
                            
                if loss_snapped < 1e-6:
                    # Convert to list of lists
                    U_list = U_snapped.tolist()
                    V_list = V_snapped.tolist()
                    W_list = W_snapped.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            return None

    # Try rank 29
    result = try_rank(rank_schoolbook - 1)
    if result is not None:
        return result
        
    # Fallback to schoolbook
    rank = rank_schoolbook
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
