import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    # A is n x m (2x4), B is m x p (4x7), C is n x p (2x7)
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Decomposition: C[i, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (28, rank)
    # W shape: (p*n, rank) = (14, rank)
    
    # We'll use a block decomposition:
    # Split j into blocks of size 2: j in {0,1}, {2,3}
    # For each j-block, split k into blocks: {0,1}, {2,3}, {4,5}, {6}
    # For 2x2 * 2x2 blocks: use Strassen (rank 7)
    # For 2x2 * 2x1 blocks: use schoolbook (rank 4)
    
    # Total: 2 j-blocks * (3 Strassen + 1 schoolbook) = 2*(21+4) = 50
    # But let's see if we can do better.
    
    # Alternative: Try to use a more efficient decomposition.
    # For 2x4 * 4x7, the lower bound is max(2*4, 4*7, 2*7) = 28.
    # Known results suggest the rank is around 48-50 for this size.
    
    # Let's implement the standard block decomposition with Strassen.
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        # Input indices in U: A[i, j] -> U[i*m + j, col]
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        # Input indices in V: B[j, k] -> V[j*p + k, col]
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        # Output indices in W: C[i, k] -> W[k*n + i, col]
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> contributes to C00 and C11
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (c+d)e -> contributes to C10 and C11
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = a(f-h) -> contributes to C01 and C11
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4 = d(g-e) -> contributes to C00 and C10
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (a+b)h -> contributes to C00 and C01
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> contributes to C11
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> contributes to C00
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        # P1 = a*e -> C00
        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1
        
        # P2 = b*g -> C00
        U[b][col] += 1
        V[g][col] += 1
        W[c00][col] += 1
        col += 1
        
        # P3 = c*e -> C10
        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1
        
        # P4 = d*g -> C10
        U[d][col] += 1
        V[g][col] += 1
        W[c10][col] += 1
        col += 1
        
        return True

    # Process all blocks
    for j_start in [0, 2]:
        if time.time() > deadline:
            break
        # k blocks: [0,1], [2,3], [4,5] use Strassen, [6] uses schoolbook
        for k_start in [0, 2, 4]:
            if time.time() > deadline:
                break
            if not add_strassen_222(0, j_start, k_start):
                break
        if time.time() > deadline:
            break
        if not add_schoolbook_221(0, j_start, 6):
            break

    return {"U": U, "V": V, "W": W}
