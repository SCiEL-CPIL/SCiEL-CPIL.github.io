import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n_tests=5):
    """
    Verify the decomposition by checking if it computes matrix multiplication
    for a set of random matrices.
    """
    n = m = p = 3
    rank = len(U[0]) if U else 0
    
    if rank == 0:
        return False
    
    # Convert to numpy arrays for efficient multiplication
    U_np = np.array(U, dtype=float).reshape(n, m, rank)
    V_np = np.array(V, dtype=float).reshape(m, p, rank)
    W_np = np.array(W, dtype=float).reshape(p, n, rank)
    
    # Test with random matrices
    for _ in range(n_tests):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_sum = np.dot(U_np[i, :, r], A[i, :])
                    v_sum = np.dot(V_np[:, k, r], B[:, k])
                    val += W_np[k, i, r] * u_sum * v_sum
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
            
    return True

def _strassen_rank26():
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    # 2x2 Strassen core (7 terms)
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # Border terms (19 terms)
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
            
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, current_rank, max_iterations=1000):
    """
    Try to reduce the rank by finding linear dependencies among the term vectors.
    Each term r corresponds to a vector v_r = (U[:,r], V[:,r], W[:,r]) in R^81.
    If we find a linear dependency, we can eliminate one term.
    """
    n = m = p = 3
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Stack all term vectors
    # Each term is a 81-dimensional vector
    term_vectors = []
    for r in range(current_rank):
        vec = np.concatenate([U_np[:, r], V_np[:, r], W_np[:, r]])
        term_vectors.append(vec)
    
    term_vectors = np.array(term_vectors)  # shape: (current_rank, 81)
    
    # Find the rank of this matrix
    # If rank < current_rank, there's a linear dependency
    try:
        rank = np.linalg.matrix_rank(term_vectors, tol=1e-10)
    except np.linalg.LinAlgError:
        return None
    
    if rank >= current_rank:
        return None
    
    # Find null space vectors
    try:
        _, s, vh = np.linalg.svd(term_vectors, full_matrices=True)
    except np.linalg.LinAlgError:
        return None
    
    # Null space vectors are rows of vh where singular values are near zero
    null_space = vh[rank:, :]  # shape: (current_rank - rank, current_rank)
    
    # Try to use each null space vector to eliminate a term
    for ns_vec in null_space:
        # Find the term with largest absolute coefficient in this dependency
        abs_coeffs = np.abs(ns_vec)
        max_idx = np.argmax(abs_coeffs)
        
        if abs_coeffs[max_idx] < 1e-10:
            continue
        
        # Normalize so that the max coefficient is 1
        scale = ns_vec[max_idx]
        if abs(scale) < 1e-15:
            continue
        ns_vec = ns_vec / scale
        
        # Create new U, V, W with the max_idx term eliminated
        new_rank = current_rank - 1
        new_U = np.zeros((n * m, new_rank))
        new_V = np.zeros((m * p, new_rank))
        new_W = np.zeros((p * n, new_rank))
        
        # Copy all terms except max_idx
        for old_r in range(current_rank):
            if old_r == max_idx:
                continue
            new_r = old_r if old_r < max_idx else old_r - 1
            new_U[:, new_r] = U_np[:, old_r]
            new_V[:, new_r] = V_np[:, old_r]
            new_W[:, new_r] = W_np[:, old_r]
        
        # Add the linear combination to each remaining term
        for new_r in range(new_rank):
            old_r = new_r if new_r < max_idx else new_r + 1
            coeff = ns_vec[old_r]
            if abs(coeff) > 1e-15:
                new_U[:, new_r] += coeff * U_np[:, max_idx]
                new_V[:, new_r] += coeff * V_np[:, max_idx]
                new_W[:, new_r] += coeff * W_np[:, max_idx]
        
        # Convert back to list format
        new_U_list = new_U.tolist()
        new_V_list = new_V.tolist()
        new_W_list = new_W.tolist()
        
        # Verify the new decomposition
        if _verify_decomposition(new_U_list, new_V_list, new_W_list, n_tests=3):
            return {"U": new_U_list, "V": new_V_list, "W": new_W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with rank-26 decomposition
    result = _strassen_rank26()
    
    # Verify it works
    if not _verify_decomposition(result["U"], result["V"], result["W"]):
        # Fallback to schoolbook
        n = m = p = 3
        rank = 27
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce the rank
    current_rank = len(result["U"][0])
    best_result = result
    
    while time.time() < deadline - 5.0:
        reduced = _try_reduce_rank(best_result["U"], best_result["V"], best_result["W"], current_rank)
        if reduced is None:
            break
        
        best_result = reduced
        current_rank = len(best_result["U"][0])
    
    return best_result

# EVOLVE-BLOCK-END
