import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Indices:
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # Target tensor T[a, b, d, i, k] = delta(a, i) * delta(d, k)
    # Equation: sum_r W[3*k+i, r] * U[3*a+b, r] * V[5*b+d, r] = T[a, b, d, i, k]
    
    # We will try to find a decomposition for various ranks R.
    # We use Alternating Least Squares (ALS).
    
    # Precompute indices
    # a in 0..n-1, b in 0..m-1, d in 0..p-1, i in 0..n-1, k in 0..p-1
    # U_idx = 3*a + b
    # V_idx = 5*b + d
    # W_idx = 3*k + i
    
    # We want to minimize sum (sum_r W[3*k+i, r] U[3*a+b, r] V[5*b+d, r] - delta(a,i)delta(d,k))^2
    
    # For a fixed a, b, the unknowns are U[3*a+b, r] for r=0..R-1.
    # The equations are for all i, k, d.
    # Let x_r = U[3*a+b, r].
    # Eq: sum_r x_r * (W[3*k+i, r] * V[5*b+d, r]) = delta(a,i)delta(d,k)
    # This is a linear system A x = y, where A is (num_eqs x R), y is (num_eqs).
    # A[i, k, d, r] = W[3*k+i, r] * V[5*b+d, r]
    # y[i, k, d] = delta(a,i)delta(d,k)
    
    # We can solve this for each a, b.
    
    def solve_U(U, V, W, R):
        new_U = np.zeros((n*m, R))
        for a in range(n):
            for b in range(m):
                # Build A and y
                # Equations for all i, k, d
                A = []
                y = []
                for i in range(n):
                    for k in range(p):
                        for d in range(p):
                            # Coefficients for x_r
                            row = []
                            for r in range(R):
                                row.append(W[3*k+i, r] * V[5*b+d, r])
                            A.append(row)
                            # Target
                            target = 1.0 if (a == i and d == k) else 0.0
                            y.append(target)
                A = np.array(A)
                y = np.array(y)
                # Solve least squares
                x, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
                new_U[3*a+b, :] = x
        return new_U

    def solve_V(U, V, W, R):
        new_V = np.zeros((m*p, R))
        for b in range(m):
            for d in range(p):
                A = []
                y = []
                for a in range(n):
                    for i in range(n):
                        for k in range(p):
                            row = []
                            for r in range(R):
                                row.append(U[3*a+b, r] * W[3*k+i, r])
                            A.append(row)
                            target = 1.0 if (a == i and d == k) else 0.0
                            y.append(target)
                A = np.array(A)
                y = np.array(y)
                x, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
                new_V[5*b+d, :] = x
        return new_V

    def solve_W(U, V, W, R):
        new_W = np.zeros((p*n, R))
        for k in range(p):
            for i in range(n):
                A = []
                y = []
                for a in range(n):
                    for b in range(m):
                        for d in range(p):
                            row = []
                            for r in range(R):
                                row.append(U[3*a+b, r] * V[5*b+d, r])
                            A.append(row)
                            target = 1.0 if (a == i and d == k) else 0.0
                            y.append(target)
                A = np.array(A)
                y = np.array(y)
                x, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
                new_W[3*k+i, :] = x
        return new_W

    def check_solution(U, V, W, R):
        # Check if the solution is exact
        # sum_r W[3*k+i, r] U[3*a+b, r] V[5*b+d, r] == delta(a,i)delta(d,k)
        for a in range(n):
            for b in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += W[3*k+i, r] * U[3*a+b, r] * V[5*b+d, r]
                            target = 1.0 if (a == i and d == k) else 0.0
                            if abs(s - target) > 1e-9:
                                return False
        return True

    # Try ranks from low to high
    start_time = time.time()
    max_time = 100  # Leave some time for safety
    
    for R in range(10, 46):
        if time.time() - start_time > max_time:
            break
            
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # ALS iterations
        for it in range(100):
            U = solve_U(U, V, W, R)
            V = solve_V(U, V, W, R)
            W = solve_W(U, V, W, R)
            
            if check_solution(U, V, W, R):
                # Found a solution
                # Convert to list of lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # If no solution found, return schoolbook
    rank = n * m * p
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
