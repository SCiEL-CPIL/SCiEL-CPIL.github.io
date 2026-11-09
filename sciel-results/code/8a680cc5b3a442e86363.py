"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W form an exact decomposition.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_r_sparse_construction(n, m, p, r, max_attempts=1000, seed=42):
    """
    Try to find a rank-r decomposition by constructing U, V with sparse integer entries
    and deriving W directly to satisfy the equations.
    """
    np.random.seed(seed)
    
    for attempt in range(max_attempts):
        # Generate sparse U and V
        # We want M[i,k,c] = sum_j U[i,j,c] * V[j,k,c] to be small integers (1, -1, 2, -2)
        # so that W[k,i,c] = 1/M[i,k,c] is integer or half-integer.
        
        # Strategy: Random sparse matrices with low density
        density = 0.2 + 0.5 * (attempt / max_attempts) # Start sparse, get denser
        U = np.zeros((n * m, r), dtype=int)
        V = np.zeros((m * p, r), dtype=int)
        
        # Initialize U and V with random -1, 0, 1
        # To keep M small, we should ensure that for each c, U[:,c] and V[:,c] are sparse
        for c in range(r):
            # Randomly pick a few non-zeros for U[:, c]
            num_nonzero_u = max(1, int(np.random.exponential(2.0)))
            if num_nonzero_u > n * m: num_nonzero_u = n * m
            u_indices = np.random.choice(n * m, num_nonzero_u, replace=False)
            U[u_indices, c] = np.random.choice([-1, 1], num_nonzero_u)
            
            num_nonzero_v = max(1, int(np.random.exponential(2.0)))
            if num_nonzero_v > m * p: num_nonzero_v = m * p
            v_indices = np.random.choice(m * p, num_nonzero_v, replace=False)
            V[v_indices, c] = np.random.choice([-1, 1], num_nonzero_v)
            
        # Now compute M[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        # And try to construct W
        W = np.zeros((p * n, r), dtype=float)
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Vector m_vec of length r
                m_vec = np.zeros(r)
                for j in range(m):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    m_vec += u_val * v_val
                
                # We need w_vec such that dot(m_vec, w_vec) = 1
                # We want w_vec to have entries in {0, 1, -1, 0.5, -0.5}
                # Check if there is a c where m_vec[c] is in {1, -1, 2, -2}
                found_c = -1
                for c in range(r):
                    val = m_vec[c]
                    if val == 1:
                        found_c = c
                        w_val = 1.0
                        break
                    elif val == -1:
                        found_c = c
                        w_val = -1.0
                        break
                    elif val == 2:
                        found_c = c
                        w_val = 0.5
                        break
                    elif val == -2:
                        found_c = c
                        w_val = -0.5
                        break
                
                if found_c == -1:
                    valid = False
                    break
                
                W[k * n + i, found_c] = w_val
        
        if not valid:
            continue
            
        # Final verification
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin (evaluator kills at 150s)
    
    # Trivial solution fallback
    rank_trivial = n * m * p
    U_triv = np.zeros((n * m, rank_trivial))
    V_triv = np.zeros((m * p, rank_trivial))
    W_triv = np.zeros((p * n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, col] = 1.0
                V_triv[j * p + k, col] = 1.0
                W_triv[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_trivial
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Lower bound for rank is max(n*m, m*p, p*n)
    lb = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound upwards
    for r in range(lb, rank_trivial):
        if time.time() > deadline:
            break
            
        # Try multiple seeds for this rank
        # The sparse construction is fast, so we can try more
        attempts = 500 if r < lb + 20 else 100
        
        found = False
        for seed in range(attempts):
            if time.time() > deadline:
                break
                
            res = _try_rank_r_sparse_construction(n, m, p, r, max_attempts=1, seed=seed + r * 1000)
            if res is not None:
                U, V, W = res
                best_rank = r
                best_U, best_V, best_W = U, V, W
                found = True
                break
                
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
