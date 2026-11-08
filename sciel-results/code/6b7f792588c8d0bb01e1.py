import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # Try to find lower rank
    # Start from rank_school - 1 down to some lower bound
    # Given time constraints, we might only check a few ranks
    
    # Helper to build the target tensor equations
    # Equation: sum_r W[r, i, k] * U[r, i, j1] * V[r, j2, k] = delta(j1, j2)
    # Note: U is (R, n, m), V is (R, m, p), W is (R, n, p)
    # In the prompt, U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # We will use 3D arrays for internal computation and reshape at the end.
    
    def _solve_rank(R):
        # Initialize with random or schoolbook
        # Schoolbook:
        # U[r, i, j] = 1 if r corresponds to (i,j)
        # V[r, j, k] = 1 if r corresponds to (j,k)
        # W[r, i, k] = 1 if r corresponds to (i,k)
        
        # Map index r to (i, j, k)
        # r = i * (m * p) + j * p + k ?
        # Schoolbook has n*m*p terms.
        # We only have R terms.
        # If R < n*m*p, we can't use pure schoolbook init.
        # We can use random init.
        
        np.random.seed(42)
        U = np.random.randn(R, n, m) * 0.1
        V = np.random.randn(R, m, p) * 0.1
        W = np.random.randn(R, n, p) * 0.1
        
        # Iterative solver
        max_iters = 200
        tol = 1e-6
        
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j1 in range(m):
                    # System for U[:, i, j1]
                    # A is (p, R)
                    A = np.zeros((p, R))
                    for k in range(p):
                        for j2 in range(m):
                            # Equation involves sum over j2?
                            # No, the equation is for specific j1, j2.
                            # sum_r W[r, i, k] * U[r, i, j1] * V[r, j2, k] = delta(j1, j2)
                            # This equation must hold for ALL j2.
                            # So we have a system for each j2?
                            # No, U[r, i, j1] is the unknown.
                            # The equation is:
                            # sum_r U[r, i, j1] * (W[r, i, k] * V[r, j2, k]) = delta(j1, j2)
                            # This must hold for all j2 in 0..m-1.
                            # So we have m equations for each k?
                            # Total p * m equations for the R variables U[:, i, j1].
                            
                            # Let's construct the system:
                            # Variables: x_r = U[r, i, j1] for r=0..R-1
                            # Equations: for each k in 0..p-1, for each j2 in 0..m-1:
                            # sum_r x_r * W[r, i, k] * V[r, j2, k] = delta(j1, j2)
                            
                            pass # We will build this properly below
            
            # Correct construction of linear system for U[:, i, j1]
            A = np.zeros((p * m, R))
            b = np.zeros(p * m)
            idx = 0
            for k in range(p):
                for j2 in range(m):
                    # A[idx, r] = W[r, i, k] * V[r, j2, k]
                    A[idx, :] = W[:, i, k] * V[:, j2, k]
                    b[idx] = 1.0 if j1 == j2 else 0.0
                    idx += 1
            
            # Solve A x = b
            # x = U[:, i, j1]
            try:
                U[:, i, j1], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                return None
            
            # Update V
            for j in range(m):
                for j2 in range(m): # j2 is the index in V? No, V is (R, m, p).
                    # Equation: sum_r W[r, i, k] * U[r, i, j1] * V[r, j2, k] = delta(j1, j2)
                    # Unknown: V[:, j, k]? No, V[r, j2, k] is the unknown for fixed j2, k?
                    # Let's fix the position in V we are solving for.
                    # V is (R, m, p).
                    # Let's solve for V[:, j, k] for fixed j, k.
                    # The equation involves V[r, j, k] (where j is the first index of V).
                    # sum_r W[r, i, k'] * U[r, i, j'] * V[r, j, k] = delta(j', j)
                    # This must hold for all i, j', k'.
                    # So for fixed j, k, we have variables x_r = V[r, j, k].
                    # Equations: for each i in 0..n-1, j' in 0..m-1, k' in 0..p-1:
                    # sum_r x_r * W[r, i, k'] * U[r, i, j'] = delta(j', j)
                    
                    A = np.zeros((n * m * p, R))
                    b = np.zeros(n * m * p)
                    idx = 0
                    for i in range(n):
                        for jp in range(m):
                            for kp in range(p):
                                A[idx, :] = W[:, i, kp] * U[:, i, jp]
                                b[idx] = 1.0 if jp == j else 0.0
                                idx += 1
                    try:
                        V[:, j, k], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        return None

            # Update W
            for i in range(n):
                for k in range(p):
                    # Unknown: W[:, i, k]
                    # Equation: sum_r W[r, i, k] * U[r, i, j1] * V[r, j2, k] = delta(j1, j2)
                    # Must hold for all j1, j2.
                    # Variables x_r = W[r, i, k].
                    # Equations: for each j1 in 0..m-1, j2 in 0..m-1:
                    # sum_r x_r * U[r, i, j1] * V[r, j2, k] = delta(j1, j2)
                    
                    A = np.zeros((m * m, R))
                    b = np.zeros(m * m)
                    idx = 0
                    for j1 in range(m):
                        for j2 in range(m):
                            A[idx, :] = U[:, i, j1] * V[:, j2, k]
                            b[idx] = 1.0 if j1 == j2 else 0.0
                            idx += 1
                    try:
                        W[:, i, k], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        return None
            
            # Check convergence
            # Calculate residual
            err = 0.0
            for i in range(n):
                for j1 in range(m):
                    for j2 in range(m):
                        for k in range(p):
                            val = np.sum(W[:, i, k] * U[:, i, j1] * V[:, j2, k])
                            target = 1.0 if j1 == j2 else 0.0
                            err += (val - target) ** 2
            if err < tol:
                break
                
        # Verify exactness with rational rounding
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if rounded solution is exact
        err = 0.0
        for i in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    for k in range(p):
                        val = np.sum(W_r[:, i, k] * U_r[:, i, j1] * V_r[:, j2, k])
                        target = 1.0 if j1 == j2 else 0.0
                        err += (val - target) ** 2
        
        if err < 1e-9:
            # Reshape to (n*m, R), (m*p, R), (p*n, R)
            # U is (R, n, m) -> (n*m, R)
            U_out = U_r.T.reshape(n*m, R)
            # V is (R, m, p) -> (m*p, R)
            V_out = V_r.T.reshape(m*p, R)
            # W is (R, n, p) -> (p*n, R)
            # Note: W index is k*n + i.
            # W_r is (R, n, p). W_r[r, i, k].
            # We need W_out[k*n + i, r] = W_r[r, i, k].
            W_out = np.zeros((p*n, R))
            for r in range(R):
                for i in range(n):
                    for k in range(p):
                        W_out[k*n + i, r] = W_r[r, i, k]
                        
            return U_out, V_out, W_out
        
        return None

    start_time = time.time()
    deadline = start_time + 120.0
    
    # Try ranks from schoolbook down
    for R in range(rank_school, 15, -1):
        if time.time() > deadline - 10:
            break
            
        res = _solve_rank(R)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    R = rank_school
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
