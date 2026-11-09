"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
# n=2, m=4, p=4
# U: (n*m, rank) = (8, rank)
# V: (m*p, rank) = (16, rank)
# W: (p*n, rank) = (8, rank)
# Tensor T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
# T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
# Otherwise T[a,b,c] = 0

n, m, p = DIMENSIONS
NM = n * m  # 8
MP = m * p  # 16
PN = p * n  # 8

def build_tensor():
    """Build the full tensor T of shape (NM, MP, PN)."""
    T = np.zeros((NM, MP, PN), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def schoolbook_rank():
    """Return the schoolbook decomposition as (U, V, W) with rank = n*m*p."""
    rank = n * m * p  # 32
    U = np.zeros((NM, rank), dtype=np.int64)
    V = np.zeros((MP, rank), dtype=np.int64)
    W = np.zeros((PN, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def verify(U, V, W):
    """Verify that U, V, W exactly decompose the tensor."""
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    T = build_tensor()
    for a in range(NM):
        for b in range(MP):
            s = 0
            for r in range(rank):
                s += U[a, r] * V[b, r] * W[0, r]  # placeholder, need proper index
            # Actually W index is c, not 0. Let me redo.
    # Better: for each (a,b,c), compute sum_r U[a,r]*V[b,r]*W[c,r]
    for a in range(NM):
        for b in range(MP):
            for c in range(PN):
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if s != T[a, b, c]:
                    return False
    return True

def verify_fast(U, V, W):
    """Verify using einsum for speed."""
    rank = U.shape[1]
    # T_reconstructed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    T = build_tensor()
    return np.array_equal(T_rec, T)

def try_rank_small(rank, timeout=30):
    """Try to find a decomposition of the given rank using random search.
    Returns (U, V, W) if found, else None."""
    start = time.time()
    T = build_tensor()
    
    # We need U (8 x rank), V (16 x rank), W (8 x rank) with integer/half-integer entries
    # such that T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    # Strategy: try various small ranks with random initialization
    # and use a simple iterative refinement
    
    best = None
    
    for attempt in range(200):
        if time.time() - start > timeout:
            break
        
        # Random initialization with small integer values
        scale = 2
        U = np.random.randint(-scale, scale + 1, (NM, rank))
        V = np.random.randint(-scale, scale + 1, (MP, rank))
        W = np.random.randint(-scale, scale + 1, (PN, rank))
        
        # Quick check
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        if np.array_equal(T_rec, T):
            return U, V, W
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 2x4 by a 4x4 matrix.
    
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    deadline = time.time() + 140  # 140s to be safe within 150s limit
    
    # Start with schoolbook (rank 32) as baseline
    U_sb, V_sb, W_sb = schoolbook_rank()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p  # 32
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 2x4 * 4x4, the tensor rank is at least max(n*m, m*p, p*n) = 16
    # But actually the lower bound is more subtle. Let's try ranks from 16 upward.
    
    # First, check if rank 16 works (this would be very good)
    for rank in range(16, best_rank):
        if time.time() > deadline:
            break
        result = try_rank_small(rank, timeout=10)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                break  # Found minimum in this range, stop
    
    # If we didn't find anything better than schoolbook, return schoolbook
    # Trim zero columns from schoolbook if any
    U, V, W = best_U, best_V, best_W
    rank = U.shape[1]
    
    # Check for zero columns and trim
    col_sums = np.abs(U) + np.abs(V) + np.abs(W)
    non_zero_cols = np.where(np.any(col_sums > 0, axis=0))[0]
    
    if len(non_zero_cols) < rank:
        U = U[:, non_zero_cols]
        V = V[:, non_zero_cols]
        W = W[:, non_zero_cols]
    
    # Convert to list of lists for the expected output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
