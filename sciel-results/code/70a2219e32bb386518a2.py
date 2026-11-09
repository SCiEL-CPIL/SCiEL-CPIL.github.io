"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x4 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes
        (n*m, rank), (m*p, rank), (p*n, rank)
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    def check_solution(U, V, W, tol=1e-8):
        """Check if the decomposition is exact within tolerance."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err < tol
    
    def round_to_half_ints(U, V, W):
        """Try to round to half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if check_solution(U_r, V_r, W_r, tol=1e-6):
            return U_r, V_r, W_r
        return None
    
    # Try to find a solution with rank < 36
    for r in [32, 33, 34, 35]:
        if time.time() > deadline:
            break
        
        # Try several random initializations with different strategies
        for seed in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            
            # Use different initialization strategies based on seed
            if seed % 3 == 0:
                # Standard random initialization
                U = np.random.randn(9, r)
                V = np.random.randn(12, r)
                W = np.random.randn(12, r)
            elif seed % 3 == 1:
                # Sparse initialization with some structure
                U = np.zeros((9, r))
                V = np.zeros((12, r))
                W = np.zeros((12, r))
                for col in range(r):
                    # Assign random non-zero entries
                    u_idx = np.random.choice(9, size=min(3, 9), replace=False)
                    v_idx = np.random.choice(12, size=min(4, 12), replace=False)
                    w_idx = np.random.choice(12, size=min(4, 12), replace=False)
                    U[u_idx, col] = np.random.choice([-1, 1], size=len(u_idx))
                    V[v_idx, col] = np.random.choice([-1, 1], size=len(v_idx))
                    W[w_idx, col] = np.random.choice([-1, 1], size=len(w_idx))
            else:
                # Mixed initialization
                U = np.random.randn(9, r) * 0.5
                V = np.random.randn(12, r) * 0.5
                W = np.random.randn(12, r) * 0.5
            
            # Alternating least squares with convergence tracking
            prev_err = float('inf')
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for W: fix U, V
                A = np.zeros((36, 12 * r))
                b = np.ones(36)
                eq = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            out = k * n + i
                            A[eq, out*r:(out+1)*r] = U[i*m+j] * V[j*p+k]
                            eq += 1
                W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(12, r)
                
                # Solve for V: fix U, W
                A = np.zeros((36, 12 * r))
                b = np.ones(36)
                eq = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            out = j * p + k
                            A[eq, out*r:(out+1)*r] = U[i*m+j] * W[k*n+i]
                            eq += 1
                V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V = V_flat.reshape(12, r)
                
                # Solve for U: fix V, W
                A = np.zeros((36, 9 * r))
                b = np.ones(36)
                eq = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            out = i * m + j
                            A[eq, out*r:(out+1)*r] = V[j*p+k] * W[k*n+i]
                            eq += 1
                U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U = U_flat.reshape(9, r)
                
                # Check error with smooth surrogate (squared error sum)
                err_sum = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            err_sum += (val - 1.0) ** 2
                
                max_err = np.sqrt(err_sum / 36.0)
                
                # Early stopping if converged
                if max_err < 1e-12:
                    rounded = round_to_half_ints(U, V, W)
                    if rounded is not None:
                        return {
                            "U": rounded[0].tolist(),
                            "V": rounded[1].tolist(),
                            "W": rounded[2].tolist()
                        }
                    break
                
                # Stagnation detection
                if it > 10 and abs(prev_err - max_err) < 1e-14:
                    break
                prev_err = max_err
            
            # After ALS, try rounding if error is small
            if check_solution(U, V, W, tol=1e-8):
                rounded = round_to_half_ints(U, V, W)
                if rounded is not None:
                    return {
                        "U": rounded[0].tolist(),
                        "V": rounded[1].tolist(),
                        "W": rounded[2].tolist()
                    }
        
        # If no solution found for this r, continue to next
    
    # Fallback: schoolbook rank 36
    r = 36
    U = [[0] * r for _ in range(9)]
    V = [[0] * r for _ in range(12)]
    W = [[0] * r for _ in range(12)]
    for c in range(36):
        i = c // 12
        j = (c % 12) // 4
        k = c % 4
        U[i * 3 + j][c] = 1
        V[j * 4 + k][c] = 1
        W[k * 3 + i][c] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
