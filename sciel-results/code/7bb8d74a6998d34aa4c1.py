"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                computed = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(computed - expected) > 1e-10:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Best result so far: schoolbook rank 112
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions via optimization
    # The constraint: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Strategy: try ranks from low to high, use gradient-based optimization
    # For each rank r, initialize randomly and minimize sum of squared errors
    
    from scipy.optimize import minimize
    
    # Precompute the index mapping
    # For each (i,j,k) -> (u_idx, v_idx, w_idx)
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i*m+j, j*p+k, k*n+i))
    
    num_triples = len(triples)
    
    # Try various ranks
    for rank in range(16, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            np.random.seed(42 + rank * 100 + attempt)
            U0 = np.random.randn(n*m, rank) * 0.1
            V0 = np.random.randn(m*p, rank) * 0.1
            W0 = np.random.randn(p*n, rank) * 0.1
            
            # Flatten
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                U = x[:n*m*rank].reshape(n*m, rank)
                V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                errors = []
                for u_idx, v_idx, w_idx in triples:
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    errors.append(val - 1)
                
                return np.sum(np.array(errors) ** 2)
            
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < 1e-8:
                # Check if it's actually a valid decomposition
                U = result.x[:n*m*rank].reshape(n*m, rank)
                V = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Verify
                valid = True
                for u_idx, v_idx, w_idx in triples:
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                
                if valid:
                    # Round to nearest half-integer if close
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Check if rounded version still works
                    valid_rounded = True
                    for u_idx, v_idx, w_idx in triples:
                        val = np.sum(U_rounded[u_idx, :] * V_rounded[v_idx, :] * W_rounded[w_idx, :])
                        if abs(val - 1) > 1e-9:
                            valid_rounded = False
                            break
                    
                    if valid_rounded:
                        U_final = U_rounded
                        V_final = V_rounded
                        W_final = W_rounded
                    else:
                        U_final = U
                        V_final = V
                        W_final = W
                    
                    best = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                    best_rank = rank
                    break
        
        if best_rank == rank:
            break
    
    return best
# EVOLVE-BLOCK-END
