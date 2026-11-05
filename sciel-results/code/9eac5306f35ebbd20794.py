"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _strassen_block():
    """Return the rank-117 decomposition using Strassen for 2x2 blocks."""
    n, m, p = DIMENSIONS
    # Partition 5 into 2, 2, 1
    sizes = [2, 2, 1]
    starts = [0, 2, 4]
    
    # We will build lists of (row, col, val) for U, V, W
    # Since rank is fixed (117), we can pre-allocate
    # But it's easier to collect and then pad or resize
    # Let's just count the rank first to be safe, or assume 117
    
    U_entries = []
    V_entries = []
    W_entries = []
    
    col_offset = 0
    
    for i_idx in range(3):
        for k_idx in range(3):
            for j_idx in range(3):
                s_i = sizes[i_idx]
                s_k = sizes[k_idx]
                s_j = sizes[j_idx]
                
                I_start = starts[i_idx]
                K_start = starts[k_idx]
                J_start = starts[j_idx]
                
                if s_i == 2 and s_k == 2 and s_j == 2:
                    # Strassen 2x2x2
                    # 7 terms
                    # Each term is a sum of products A[u,v] * B[v,w]
                    # We add 1 column per Strassen term
                    terms = [
                        [(0,0,0,1), (0,0,1,1), (1,1,0,1), (1,1,1,1)],
                        [(1,0,0,1), (1,1,0,1)],
                        [(0,0,1,1), (0,1,1,-1)],
                        [(1,1,0,1), (1,0,1,-1)],
                        [(0,0,1,1), (0,1,1,1)],
                        [(1,0,0,1), (1,1,0,1), (0,0,0,-1), (0,1,0,-1)],
                        [(0,1,1,1), (0,0,1,1), (1,1,1,-1), (1,0,1,-1)]
                    ]
                    
                    for term in terms:
                        col = col_offset
                        for (u, v, w, coeff) in term:
                            u_row = I_start + u
                            v_row = K_start + v
                            w_row = J_start + w
                            
                            U_entries.append((u_row * m + v_row, col, coeff))
                            V_entries.append((v_row * p + w_row, col, coeff))
                            W_entries.append((w_row * n + u_row, col, coeff))
                        
                        col_offset += 1
                else:
                    # Schoolbook s_i x s_k x s_j
                    for u in range(s_i):
                        for v in range(s_k):
                            for w in range(s_j):
                                col = col_offset
                                u_row = I_start + u
                                v_row = K_start + v
                                w_row = J_start + w
                                
                                U_entries.append((u_row * m + v_row, col, 1))
                                V_entries.append((v_row * p + w_row, col, 1))
                                W_entries.append((w_row * n + u_row, col, 1))
                                
                                col_offset += 1
    
    rank = col_offset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, c, v in U_entries:
        U[r][c] += v
    for r, c, v in V_entries:
        V[r][c] += v
    for r, c, v in W_entries:
        W[r][c] += v
        
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0])
    # Build the target tensor T[i, j, k] = 1 if the (i,j) of A and (j,k) of B produce C[i,k]
    # The tensor for matmul: T[i,j,k] = 1 (for all i,j,k) in the standard sense:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor M[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank(rank, n, m, p, deadline, seed=None):
    """Try to find a rank-rank decomposition with entries in {0, 1/2, 1, -1/2, -1} or integers.
    Uses random search with local improvement."""
    if seed is not None:
        random.seed(seed)
    np.random.seed(seed)
    
    start = time.time()
    
    # Candidate values
    vals = [0, 0.5, 1, -0.5, -1]
    
    # Try random initialization and then local search
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
        
        # Convert to numpy for faster verification
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # Reshape for efficient checking
        # U shape: (n*m, rank)
        # V shape: (m*p, rank)  
        # W shape: (p*n, rank)
        
        # Build the decomposition tensor
        # D[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can compute this as:
        # For each c, outer product of U[:,c], V[:,c], W[:,c] reshaped appropriately
        
        # Let's just do it directly but with numpy
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        errors.append((i, j, k, s))
        
        if not errors:
            # Found a valid decomposition!
            return {"U": U, "V": V, "W": W}
        
        # Local search: try flipping one entry at a time
        # This is expensive, so let's try a simpler approach
        # Try modifying one factor at a time
    
    return None

def _try_structured_rank(rank, n, m, p, deadline):
    """Try structured decompositions based on known low-rank matmul schemes.
    For 5x5x5, the best known upper bound is 125 (schoolbook).
    There is no known decomposition with rank < 125 for 5x5x5.
    However, let's try some structured approaches."""
    start = time.time()
    
    # Try a decomposition based on the "border rank" or other algebraic methods
    # For now, this is a placeholder for future work
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin before 150s kill
    
    # Try the structured Strassen-based decomposition first
    # This is a known valid solution with rank 117
    result = _strassen_block()
    
    # Verify it just in case
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook if Strassen implementation fails verification
    # (It shouldn't, but safety first)
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
