import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, try to find a rank-27 decomposition via numerical search
    # If not found within time budget, fall back to rank 28
    
    deadline = start_time + 100  # Leave 50s safety margin
    
    # Build the tensor T[i,j,k] = delta_{i,j,k} for matrix mult 2x4x4
    # T has shape (n*m, m*p, p*n) = (8, 16, 8)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    
    # Actually, the tensor for matrix multiplication C = A @ B is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] where the multiplication tensor has
    # T[a, b, c] where a indexes A, b indexes B, c indexes C
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find rank-27 decomposition
    # We need U (8,27), V (16,27), W (8,27) such that
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is a hard non-convex problem. Let's try a greedy/iterative approach.
    # But given time constraints, let's first check if we can find one via
    # a different algebraic construction.
    
    # Alternative: try to modify the Strassen decomposition to save 1 term
    # by exploiting linear dependencies.
    
    # For now, use the known rank-28 solution as fallback
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def map_u_a(side, local_idx):
        if side == 'L':
            return [0, 1, 4, 5][local_idx]
        else:
            return [2, 3, 6, 7][local_idx]
            
    def map_v_b(block, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if block == 'TL':
            return r * 4 + c
        elif block == 'TR':
            return r * 4 + (2 + c)
        elif block == 'BL':
            return (2 + r) * 4 + c
        elif block == 'BR':
            return (2 + r) * 4 + (2 + c)
            
    def map_w_c(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if side == 'L':
            return c * 2 + r
        else:
            return (2 + c) * 2 + r
            
    col = 0
    for u_l, v_l, w_l in strassen_terms:
        for idx, coeff in u_l:
            U[map_u_a('L', idx), col] += coeff
        for idx, coeff in v_l:
            V[map_v_b('TL', idx), col] += coeff
        for idx, coeff in w_l:
            W[map_w_c('L', idx), col] += coeff
        col += 1
        
    for u_l, v_l, w_l in strassen_terms:
        for idx, coeff in u_l:
            U[map_u_a('R', idx), col] += coeff
        for idx, coeff in v_l:
            V[map_v_b('BL', idx), col] += coeff
        for idx, coeff in w_l:
            W[map_w_c('L', idx), col] += coeff
        col += 1
            
    for u_l, v_l, w_l in strassen_terms:
        for idx, coeff in u_l:
            U[map_u_a('L', idx), col] += coeff
        for idx, coeff in v_l:
            V[map_v_b('TR', idx), col] += coeff
        for idx, coeff in w_l:
            W[map_w_c('R', idx), col] += coeff
        col += 1
            
    for u_l, v_l, w_l in strassen_terms:
        for idx, coeff in u_l:
            U[map_u_a('R', idx), col] += coeff
        for idx, coeff in v_l:
            V[map_v_b('BR', idx), col] += coeff
        for idx, coeff in w_l:
            W[map_w_c('R', idx), col] += coeff
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
