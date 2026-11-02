"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

N, M, P = 2, 2, 8
NM, MP, PN = N * M, M * P, P * N  # 4, 16, 16
DEADLINE = time.time() + 130  # 20s safety margin before 150s kill

def verify(U, V, W, n=N, m=M, p=P):
    """Verify exact decomposition. Returns True/False."""
    nm, mp, pn = n * m, m * p, p * n
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for mat in (U, V, W):
        for row in mat:
            if len(row) != rank:
                return False
    # Convert to numpy for speed
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    # Build valid set
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = sum(Ua[a, r] * Va[b, r] * Wa[c, r] for r in range(rank))
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True

def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return U, V, W

def strassen_decomposition_228():
    """
    Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}.
    """
    # Correct Strassen terms for M_{2,2,2}
    # u, v, w are vectors of length 4 (flattened 2x2 matrices: [00, 01, 10, 11])
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    # 4 blocks of 2x2 in the 2x8 matrix
    for block_idx in range(4):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            # Fill U
            for i in range(4):
                U[i][col] = u[i]
                
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * P + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]
                    
            # Fill W
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * N + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]
                    
    return U, V, W

def _try_alternative_constructions():
    """
    Try alternative structural constructions that might yield rank < 28.
    Returns (U, V, W) if a valid lower-rank solution is found, else None.
    """
    if time.time() > DEADLINE:
        return None
    
    # Try a construction based on a different block decomposition.
    # Instead of 4 blocks of 2x2, try 2 blocks of 2x4.
    # For 2x2x4, we can try to find a rank-13 or rank-14 decomposition.
    # If we find rank-14 for each, total is 28 (same).
    # If we find rank-13 for each, total is 26 (improvement!).
    
    # Known: minimal rank for 2x2x4 is at most 14.
    # Let's try to construct a rank-14 decomposition for 2x2x4.
    
    # For 2x2x4: A is 2x2, B is 2x4, C is 2x4.
    # Tensor entries: T[a,b,c] = 1 if a=i*2+j, b=j*4+k, c=k*2+i
    # for i,j in 0..1, k in 0..3.
    
    # Try a Strassen-like construction for 2x2x4.
    # Split B into two 2x2 blocks: B = [B1, B2]
    # Then C = A*B1 + A*B2 (but with the specific index structure).
    
    # Actually, the structure is different. Let me think...
    # C[k*2+i] = sum_j A[i,j] * B[j, k]
    # So C is 2x4, with C[i, k] = sum_j A[i,j] * B[j, k]
    
    # This is just standard 2x2 times 2x4 matrix multiplication.
    # The minimal rank for this is known to be at most 14 (from literature).
    
    # For now, return None. Constructing an explicit rank-14 solution
    # for 2x2x4 and combining them is complex and may not yield improvement
    # over the existing rank-28 solution.
    
    return None

def construct():
    """Entry point: return the best decomposition found."""
    try:
        # Try rank 28 first (known good solution)
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            # Check if we have time to search for better
            if time.time() < DEADLINE - 10:
                # Try alternative constructions
                result = _try_alternative_constructions()
                if result is not None:
                    U2, V2, W2 = result
                    if verify(U2, V2, W2):
                        return {"U": U2, "V": V2, "W": W2}
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook if verification fails (should not happen)
            U, V, W = schoolbook()
            return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
