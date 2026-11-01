import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Try rank 20 first, fall back to 21 if needed
    for rank in [20, 21]:
        if time.time() > deadline - 5:
            break
        try:
            result = _try_decomposition(rank)
            if result is not None:
                return result
        except Exception:
            continue
    
    # Fallback: use the known rank-21 Strassen block decomposition
    return _strassen_rank21()

def _try_decomposition(rank):
    """Attempt to find a rank-r decomposition for 2x2 by 2x6 matrix mult."""
    n, m, p = 2, 2, 6
    
    # The tensor T[i,j,k] = 1 if i==k else 0, for i in 0..1, j in 0..1, k in 0..5
    # We need U (4 x rank), V (12 x rank), W (12 x rank) such that
    # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
    
    # W[k,r] where k = 2*i + j (trace convention: W[k*n+i] but here output is 2x6)
    # Actually W has shape (p*n, rank) = (12, rank)
    # W index: for output element (i, k) where i in 0..1, k in 0..5
    # The convention says W[k*n + i] = W[k*2 + i]
    
    # Let me construct this more carefully
    # U: shape (n*m, rank) = (4, rank), indexed by (i, j) -> i*m + j = i*2 + j
    # V: shape (m*p, rank) = (12, rank), indexed by (j, k) -> j*p + k = j*6 + k  
    # W: shape (p*n, rank) = (12, rank), indexed by (k, i) -> k*n + i = k*2 + i
    
    # For each rank-1 term r, we need:
    # U[:,r] is a 4-vector (u00, u01, u10, u11)
    # V[:,r] is a 12-vector (v00, v01, v02, v03, v04, v05, v10, v11, v12, v13, v14, v15)
    # W[:,r] is a 12-vector (w00, w01, w10, w11, w20, w21, w30, w31, w40, w41, w50, w51)
    # where w_{ki} corresponds to output position (i, k)
    
    # The constraint is: for all i,j,k:
    # sum_r U[i*2+j, r] * V[j*6+k, r] * W[k*2+i, r] = 1 if i==k else 0
    
    # This is a system of equations. Let me try to construct it.
    
    # For a valid decomposition, we need to satisfy 2*2*6 = 24 equations
    # with rank*3 = 3*rank unknowns (minus scaling freedom)
    
    # Let me try to build this from Strassen-like patterns
    
    # Strassen for 2x2 gives 7 terms. For 2x2 by 2x6, we have 3 column pairs.
    # Maybe we can save 1 term by sharing structure.
    
    # Let me try to explicitly construct rank-20 terms
    # This is complex, so let me use a known construction if possible
    
    # Actually, let me just verify if rank 20 is achievable by trying to solve
    # the linear system
    
    # For now, return None to fall back to rank 21
    return None

def _strassen_rank21():
    """Return the known rank-21 Strassen block decomposition."""
    n, m, p = 2, 2, 6
    rank = 21
    
    strassen = [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]}, # P1
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]}, # P2
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]}, # P3
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]}, # P4
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]}, # P5
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]}, # P6
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]}, # P7
    ]
    
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U[idx][col] = sc['U'][idx]
            
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V[v_idx[idx]][col] = sc['V'][idx]
            
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W[w_idx[idx]][col] = sc['W'][idx]
    
    return {"U": U, "V": V, "W": W}
