"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _make_tensor(n, m, p):
    """Create the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1"""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that U, V, W exactly decompose tensor T."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reconstruct tensor
    for c in range(rank):
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    # T_reconstructed[i,j,k] += U[i,c]*V[j,c]*W[k,c]
                    pass
    # More efficient check using einsum
    T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
    if not np.allclose(T_rec, T, atol=tol):
        return False
    return True

def _try_strassen_block(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition by partitioning into blocks
    and using Strassen-like 7-product base case where applicable.
    """
    # For 4x5x5, we can try splitting 4 = 2+2 and 5 = 2+3 (or other splits)
    # The 2x2x2 case has rank 7 (Strassen)
    
    # Let's try a simple block approach:
    # Split A (4x5) into top (2x5) and bottom (2x5)
    # Split B (5x5) into left (5x2) and right (5x3)
    # Then C (4x5) has blocks C11 (2x2), C12 (2x3), C21 (2x2), C22 (2x3)
    
    # Products needed:
    # C11 = A11*B11 + A12*B21  (2x2 * 2x2 + 2x3 * 3x2) -- not standard
    # This gets complicated with non-square blocks.
    
    # Alternative: try rank minimization via numerical optimization
    # Start from schoolbook and try to merge terms
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    T = _make_tensor(n, m, p)
    
    # Try a few random initializations with lower rank
    for target_rank in range(70, rank_schoolbook):  # Start from reasonable lower bound
        if time.time() > deadline - 30:
            break
        for trial in range(20):
            if time.time() > deadline - 30:
                break
            np.random.seed(trial * 1000 + target_rank)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Simple gradient-free approach: check if this random init is close
            T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_rec - T)
            if err < 1e-6:
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return None

def construct():
    deadline = time.time() + 120 - 10  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Start with schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {'U': U, 'V': V, 'W': W}
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decomposition
    result = _try_strassen_block(n, m, p, deadline)
    if result is not None:
        U_try = np.array(result['U'])
        rank_try = U_try.shape[1]
        if rank_try < best_rank:
            # Verify
            T = _make_tensor(n, m, p)
            if _verify_decomposition(U_try, np.array(result['V']), np.array(result['W']), T):
                best_result = result
                best_rank = rank_try
    
    return best_result
# EVOLVE-BLOCK-END
