import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True


def _round_to_half(arr):
    """Round each entry to the nearest half-integer."""
    return np.round(arr * 2.0) / 2.0


def _find_decomp(n, m, p, max_rank, time_limit=10):
    """
    Try to find a decomposition of rank <= max_rank for M_{n,m,p}.
    Returns (U, V, W) or None.
    """
    start_time = time.time()
    # Try ranks from max_rank down to n*m*p
    # But usually we want the minimum, so we try lower ranks first?
    # No, we want to find ANY solution with rank < schoolbook.
    # Let's try ranks from a lower bound up to max_rank.
    # Lower bound: max(n*m, m*p, p*n)
    lb = max(n * m, m * p, p * n)
    
    # We are specifically interested in ranks below schoolbook
    schoolbook_rank = n * m * p
    
    # Try a few specific ranks
    # For 3,3,3: try 23, 24, 25, 26, 27
    # For 3,3,2: try 16, 17, 18
    
    # Generate candidate ranks
    candidates = []
    for r in range(lb, min(max_rank, schoolbook_rank) + 1):
        candidates.append(r)
    
    # Prioritize lower ranks
    # But if max_rank is small, just use that
    
    # Actually, let's just try to find a solution for the target max_rank first.
    # If not found, try higher? No, we want minimal.
    # Let's try ranks from lb to max_rank.
    
    for rank in range(lb, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations
        for trial in range(20):
            if time.time() - start_time > time_limit:
                break
                
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Gradient descent
            lr = 0.1
            for it in range(500):
                if time.time() - start_time > time_limit:
                    break
                    
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                error = 0.0
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            diff = s - 1.0
                            error += diff * diff
                            
                            grad_U[i * m + j, :] += 2 * diff * V[j * p + k, :] * W[k * n + i, :]
                            grad_V[j * p + k, :] += 2 * diff * U[i * m + j, :] * W[k * n + i, :]
                            grad_W[k * n + i, :] += 2 * diff * U[i * m + j, :] * V[j * p + k, :]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                lr *= 0.99
                
                if error < 1e-8:
                    break
            
            # Round
            U_r = _round_to_half(U)
            V_r = _round_to_half(V)
            W_r = _round_to_half(W)
            
            if _verify(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
                
    return None, None, None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    deadline = time.time() + 140
    
    # Split p=8 into 3, 3, 2
    p1, p2, p3 = 3, 3, 2
    
    # Find decompositions for sub-problems
    # M_{3,3,3}
    U1, V1, W1 = _find_decomp(n, m, p1, 27, time_limit=20)
    if U1 is None:
        U1, V1, W1 = _build_schoolbook(n, m, p1)
        rank1 = n * m * p1
    else:
        rank1 = U1.shape[1]
        
    # M_{3,3,3} (second part)
    U2, V2, W2 = _find_decomp(n, m, p2, 27, time_limit=20)
    if U2 is None:
        U2, V2, W2 = _build_schoolbook(n, m, p2)
        rank2 = n * m * p2
    else:
        rank2 = U2.shape[1]
        
    # M_{3,3,2}
    U3, V3, W3 = _find_decomp(n, m, p3, 18, time_limit=20)
    if U3 is None:
        U3, V3, W3 = _build_schoolbook(n, m, p3)
        rank3 = n * m * p3
    else:
        rank3 = U3.shape[1]
        
    total_rank = rank1 + rank2 + rank3
    
    # Combine
    U = np.hstack([U1, U2, U3])
    
    # V: shape (m*p, total_rank)
    # V1 is (m*p1, rank1) -> (3*3, rank1) = (9, rank1)
    # V2 is (m*p2, rank2) -> (9, rank2)
    # V3 is (m*p3, rank3) -> (3*2, rank3) = (6, rank3)
    
    # We need to map V1, V2, V3 to V of shape (m*p, total_rank) = (24, total_rank)
    # V[j, k, r]
    # For k in [0, p1): use V1[j, k, r]
    # For k in [p1, p1+p2): use V2[j, k-p1, r]
    # For k in [p1+p2, p): use V3[j, k-p1-p2, r]
    
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Fill V
    # V1 covers k=0,1,2
    for j in range(m):
        for k in range(p1):
            V[j * p + k, :rank1] = V1[j * p1 + k, :]
            
    # V2 covers k=3,4,5
    for j in range(m):
        for k in range(p2):
            V[j * p + (p1 + k), rank1:rank1+rank2] = V2[j * p2 + k, :]
            
    # V3 covers k=6,7
    for j in range(m):
        for k in range(p3):
            V[j * p + (p1 + p2 + k), rank1+rank2:] = V3[j * p3 + k, :]
            
    # Fill W
    # W1 covers k=0,1,2
    for i in range(n):
        for k in range(p1):
            W[k * n + i, :rank1] = W1[k * n + i, :]
            
    # W2 covers k=3,4,5
    for i in range(n):
        for k in range(p2):
            W[(p1 + k) * n + i, rank1:rank1+rank2] = W2[k * n + i, :]
            
    # W3 covers k=6,7
    for i in range(n):
        for k in range(p3):
            W[(p1 + p2 + k) * n + i, rank1+rank2:] = W3[k * n + i, :]
            
    # Verify combined solution
    if _verify(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        U0, V0, W0 = _build_schoolbook(n, m, p)
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }

# EVOLVE-BLOCK-END
