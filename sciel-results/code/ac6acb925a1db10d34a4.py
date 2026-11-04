import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    # Known rank of 3x3 matrix multiplication is 21 (or 22).
    # We will try to find a solution for R=21.
    # If it fails or takes too long, we fall back to schoolbook or higher ranks.
    
    deadline = time.time() + 110  # 10s margin
    
    # Construct the target tensor T
    # T[I, J, K] = 1 if I=(i,j), J=(j,k), K=(k,i) matches
    # I = i*m + j
    # J = j*p + k
    # K = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def compute_residuals(x, R):
        # x is flattened U, V, W
        # U: n*m x R
        # V: m*p x R
        # W: p*n x R
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        
        U = x[:size_u].reshape(n*m, R)
        V = x[size_u:size_u+size_v].reshape(m*p, R)
        W = x[size_u+size_v:size_u+size_v+size_w].reshape(p*n, R)
        
        # Compute P[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
        # This is a batched matrix multiplication or einsum
        # P = np.einsum('ir,jr,kr->ijk', U, V, W)
        # This might be slow for large R, but R=21 is small.
        P = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Residual is P - T
        return (P - T).ravel()

    def try_rank(R):
        if time.time() > deadline:
            return None
        
        # Initialize
        # Random initialization
        np.random.seed(42)
        x0 = np.random.randn(n*m*R + m*p*R + p*n*R)
        
        # Scale initialization to help convergence
        # The product of 3 random numbers ~ N(0,1) is small.
        # We want sum to be 1. So scale by R^(1/3)?
        x0 = x0 * (R ** (1/3))
        
        try:
            # Use least_squares
            # max_nfev to limit time
            res = least_squares(compute_residuals, x0, args=(R,), method='trf', max_nfev=2000)
            
            if time.time() > deadline:
                return None
                
            # Check residual
            # res.fun is the residual
            # If max abs residual is small, we have a solution
            # Then round to half-integers
            x_sol = res.x
            size_u = n * m * R
            size_v = m * p * R
            size_w = p * n * R
            
            U_sol = x_sol[:size_u].reshape(n*m, R)
            V_sol = x_sol[size_u:size_u+size_v].reshape(m*p, R)
            W_sol = x_sol[size_u+size_v:size_u+size_v+size_w].reshape(p*n, R)
            
            # Round to nearest 0.5
            U_round = np.round(U_sol * 2) / 2
            V_round = np.round(V_sol * 2) / 2
            W_round = np.round(W_sol * 2) / 2
            
            # Verify exactness
            P_check = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
            if np.allclose(P_check, T, atol=1e-9):
                # Convert to lists
                return {
                    "U": U_round.tolist(),
                    "V": V_round.tolist(),
                    "W": W_round.tolist()
                }
            else:
                # If not exact, maybe try to refine or return None
                # For now, return None to try next rank
                return None
                
        except Exception as e:
            return None

    # Try ranks from 21 upwards
    for R in range(21, 28):
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result is not None:
            return result

    # Fallback to schoolbook
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
