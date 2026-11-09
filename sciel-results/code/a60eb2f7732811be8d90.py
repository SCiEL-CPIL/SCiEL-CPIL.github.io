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
    
    # Try to find a solution with rank < 36
    for r in [32, 33, 34, 35]:
        if time.time() > deadline:
            break
        
        # Try several random initializations
        for seed in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            U = np.random.randn(9, r)
            V = np.random.randn(12, r)
            W = np.random.randn(12, r)
            
            # Alternating least squares
            for it in range(150):
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
                
                # Check error
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-10:
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    max_err_r = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                                max_err_r = max(max_err_r, abs(val - 1.0))
                    
                    if max_err_r < 1e-6:
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
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
