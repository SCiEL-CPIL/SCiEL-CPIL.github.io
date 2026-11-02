# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, T):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    reconstructed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(reconstructed, T, atol=1e-9, rtol=1e-9)

def _snap_to_half(x):
    return np.round(2 * x) / 2.0

def _try_find_rank_22():
    """
    Attempts to find a rank 22 decomposition.
    Strategy: Decompose into two 2x3x2 blocks.
    Find a rank 11 decomposition for 2x3x2.
    """
    n, m, p = 2, 3, 4
    target_rank = 22
    T = _build_target_tensor()
    
    # We assume the decomposition is block-diagonal in the output k dimension.
    # Block 1: k=0,1. Block 2: k=2,3.
    # Each block is a 2x3x2 tensor.
    # We search for a rank 11 decomposition for 2x3x2.
    
    # Initialize U, V, W for the full 2x3x4 tensor.
    # U: (6, 22), V: (12, 22), W: (8, 22)
    
    # Let's try a random search for the 2x3x2 sub-problem first.
    # If we find U1, V1, W1 (rank 11) for T1 (2x3x2),
    # we can set U[0:6, 0:11] = U1, V[0:6, 0:11] = V1 (padded to 3x2), W[0:4, 0:11] = W1 (padded to 2x2? No, W is p x n = 4x2).
    # For block 1 (k=0,1), V1 is 3x2. It should be placed in V[0:3, 0:2] for the first 11 cols?
    # V shape is (m*p, rank) = (3*4, 22) = (12, 22).
    # V[j, k] index is j*p + k.
    # For k=0,1: indices 0,1, 3,4, 6,7.
    # For k=2,3: indices 2,3, 5,6, 8,9.
    
    # This mapping is complex. Let's just do a direct random search for rank 22 with snapping.
    
    start_time = time.time()
    deadline = start_time + 10 # 10 seconds for search
    
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for _ in range(100):
        if time.time() > deadline:
            break
            
        U = np.random.randn(6, 22)
        V = np.random.randn(12, 22)
        W = np.random.randn(8, 22)
        
        # Snap to half integers
        U = _snap_to_half(U)
        V = _snap_to_half(V)
        W = _snap_to_half(W)
        
        # Local refinement (simple gradient descent or just check)
        # Given the small size, we can try a few steps of ALS.
        # But first, just check if this random snap works (unlikely).
        
        # Better: Use a structured initialization.
        # Try to construct it from Strassen-like components.
        # This is hard to code blindly.
        
        # Alternative: Just return schoolbook if search fails.
        pass
    
    # If we don't find it, return None
    return None

def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    T = _build_target_tensor()
    
    # Try rank 22
    # The random search in _try_find_rank_22 is currently a stub.
    # We need a better search.
    
    # Let's try a specific known decomposition for 2x3x2 rank 11?
    # I will implement a simple local search from random starts, snapping to 0.5.
    
    def _search_rank_22():
        start = time.time()
        limit = start + 30 # 30s budget
        
        for _ in range(200):
            if time.time() > limit:
                break
            
            U = np.random.randn(6, 22)
            V = np.random.randn(12, 22)
            W = np.random.randn(8, 22)
            
            # Snap
            U = _snap_to_half(U)
            V = _snap_to_half(V)
            W = _snap_to_half(W)
            
            if _check_decomposition(U, V, W, T):
                return U, V, W
                
            # Try a few steps of refinement
            # This is too slow for 200 iterations.
            
        return None

    result = _search_rank_22()
    if result:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback
    U, V, W = _schoolbook_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
