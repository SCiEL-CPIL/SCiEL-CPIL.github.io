"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
_start_time = time.time()

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _construct_rank55():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard
    deadline = _start_time + 110
    
    # Start with rank-55 construction
    result = _construct_rank55()
    best_rank = 55
    best_result = result
    
    # Try to reduce rank by searching
    # For each candidate rank from 54 down to 40, try to find a valid decomposition
    for target_rank in range(54, 39, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Iterative refinement: fix two, solve for third
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for W given U, V
                # For each (i,k): W[k*n+i, r] should satisfy
                # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is: sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
                # Let M_ik[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                # Then M_ik @ W_col = 1 for each (i,k)
                
                # Build the system for W
                # For each (i,k), we need: sum_r M_ik[r] * W[k*n+i, r] = 1
                # This is a set of n*p equations in n*p*target_rank unknowns
                # But W[k*n+i, r] only appears in equation (i,k)
                # So for each (i,k): M_ik @ W[k*n+i, :] = 1 (scalar)
                # This is underdetermined if target_rank > 1
                
                # Use least squares
                for i in range(n):
                    for k in range(p):
                        M_ik = np.zeros(target_rank)
                        for j in range(m):
                            M_ik += U[i*m+j, :] * V[j*p+k, :]
                        # M_ik @ w = 1, where w = W[k*n+i, :]
                        if np.linalg.norm(M_ik) > 1e-10:
                            # Least squares solution
                            w, _, _, _ = np.linalg.lstsq(M_ik.reshape(-1,1), np.array([1.0]), rcond=None)
                            W[k*n+i, :] = w.flatten()
                
                # Similarly solve for U and V
                for i in range(n):
                    for j in range(m):
                        # sum_k sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] * W[k*n+i, r]) = 1
                        M_ij = np.zeros(target_rank)
                        for k in range(p):
                            M_ij += V[j*p+k, :] * W[k*n+i, :]
                        if np.linalg.norm(M_ij) > 1e-10:
                            u, _, _, _ = np.linalg.lstsq(M_ij.reshape(-1,1), np.array([1.0]), rcond=None)
                            U[i*m+j, :] = u.flatten()
                
                for j in range(m):
                    for k in range(p):
                        # sum_i sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # sum_r V[j*p+k, r] * (sum_i U[i*m+j, r] * W[k*n+i, r]) = 1
                        M_jk = np.zeros(target_rank)
                        for i in range(n):
                            M_jk += U[i*m+j, :] * W[k*n+i, :]
                        if np.linalg.norm(M_jk) > 1e-10:
                            v, _, _, _ = np.linalg.lstsq(M_jk.reshape(-1,1), np.array([1.0]), rcond=None)
                            V[j*p+k, :] = v.flatten()
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < 1e-10:
                    # Found a valid decomposition!
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    # Round to nearest half-integer
                    U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                    V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                    W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                    # Verify
                    if _verify(U_rounded, V_rounded, W_rounded, n, m, p, target_rank):
                        return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    break
                
                if max_err < 1e-6:
                    break
    
    # Return best result found
    return best_result
# EVOLVE-BLOCK-END
