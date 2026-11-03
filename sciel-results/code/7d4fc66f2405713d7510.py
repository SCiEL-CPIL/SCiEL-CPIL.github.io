"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 54
    
    # Initialize with schoolbook decomposition
    best_result = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact using vectorized numpy."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        # Build the target tensor
        M = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    M[a, b, c] = 1.0
        # Reconstruct
        M_rec = np.zeros_like(M)
        for t in range(rank):
            M_rec += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
        return np.allclose(M, M_rec, atol=1e-9)
    
    # Strategy: Exploit block structure C[:,0:3] = A*B1, C[:,3:6] = A*B2
    # Share U factor between blocks
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For block 1 (k=0,1,2): V1 = V[0:9, :], W1 = W[0:9, :]
    # For block 2 (k=3,4,5): V2 = V[9:18, :], W2 = W[9:18, :]
    # U is shared: U[i*m+j, t] appears in both blocks
    
    # Try to find a decomposition with shared U structure
    # Use values from {-1, 0, 1} and {-0.5, 0, 0.5}
    
    for rank in range(50, 30, -1):
        if time.time() > deadline - 20:
            break
        
        max_attempts = 20 if rank <= 40 else 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.RandomState(attempt * 7919 + rank * 31 + 1000)
            
            # Generate U with sparsity
            density_u = 0.5
            U = rng.choice([-1, 0, 1], size=(n * m, rank))
            mask_u = rng.random((n * m, rank)) < density_u
            U = U * mask_u
            
            # Generate V and W with sparsity
            density_v = 0.4
            density_w = 0.4
            V = rng.choice([-1, 0, 1], size=(m * p, rank))
            W = rng.choice([-1, 0, 1], size=(p * n, rank))
            mask_v = rng.random((m * p, rank)) < density_v
            mask_w = rng.random((p * n, rank)) < density_w
            V = V * mask_v
            W = W * mask_w
            
            if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best_rank == rank:
            continue
    
    # Try half-integer values
    if time.time() < deadline - 30:
        for rank in [48, 46, 44, 42, 40, 38]:
            if time.time() > deadline - 15:
                break
                
            for attempt in range(15):
                if time.time() > deadline - 5:
                    break
                    
                rng = np.random.RandomState(attempt * 104729 + rank * 13 + 2000)
                
                vals = [-1, -0.5, 0, 0, 0, 0.5, 1]
                U = rng.choice(vals, size=(n * m, rank))
                V = rng.choice(vals, size=(m * p, rank))
                W = rng.choice(vals, size=(p * n, rank))
                
                if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                    break
    
    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
# EVOLVE-BLOCK-END
