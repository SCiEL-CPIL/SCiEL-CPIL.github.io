"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140  # Safety margin
    
    def validate(U, V, W, rank):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def strassen_2x2x2(U, V, W, col, i0, i1, j0, j1, k0, k1):
        """Add Strassen decomposition for 2x2x2 block. Returns new col."""
        # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+i1, col] += 1; W[k1*n+i1, col] -= 1
        col += 1
        # P3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        U[i0*m+j0, col] += 1
        V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
        W[k1*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        U[i1*m+j1, col] += 1
        V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+i0, col] += 1; W[k0*n+i1, col] += 1
        col += 1
        # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+i0, col] -= 1; W[k1*n+i0, col] += 1
        col += 1
        # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        U[i1*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
        W[k1*n+i1, col] += 1
        col += 1
        # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U[i0*m+j1, col] += 1; U[i1*m+j1, col] -= 1
        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1
        col += 1
        return col
    
    def schoolbook_block(U, V, W, col, i_range, j_range, k_range):
        """Add schoolbook terms for a block. Returns new col."""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+i, col] += 1
                    col += 1
        return col
    
    def gen_splits(total, max_size):
        """Generate all ways to split 'total' into parts of size 1 to max_size."""
        if total == 0:
            yield []
            return
        for size in range(1, min(max_size, total) + 1):
            for rest in gen_splits(total - size, max_size):
                yield [size] + rest
    
    def compute_rank(j_split, k_split):
        """Compute the rank for a given partition."""
        rank = 0
        for js in j_split:
            for ks in k_split:
                if js == 2 and ks == 2:
                    rank += 7  # Strassen
                else:
                    rank += n * js * ks  # schoolbook
        return rank
    
    def build_decomposition(j_split, k_split, rank):
        """Build the U, V, W matrices for a given partition."""
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        j_offset = 0
        for js in j_split:
            j_indices = list(range(j_offset, j_offset + js))
            k_offset = 0
            for ks in k_split:
                k_indices = list(range(k_offset, k_offset + ks))
                if js == 2 and ks == 2:
                    col = strassen_2x2x2(U, V, W, col, 0, 1, j_indices[0], j_indices[1], k_indices[0], k_indices[1])
                else:
                    col = schoolbook_block(U, V, W, col, [0, 1], j_indices, k_indices)
                k_offset += ks
            j_offset += js
        return U, V, W, rank
    
    # Generate all candidate partitions
    j_splits = list(gen_splits(m, max_size=2))
    k_splits = list(gen_splits(p, max_size=2))
    
    # Sort by estimated rank to try best first
    candidates = []
    for js in j_splits:
        for ks in k_splits:
            r = compute_rank(js, ks)
            candidates.append((r, js, ks))
    candidates.sort()
    
    best_U, best_V, best_W, best_rank = None, None, None, float('inf')
    
    for r, js, ks in candidates:
        if r >= best_rank:
            break
        if time.time() > deadline - 10:
            break
        
        U, V, W, rank = build_decomposition(js, ks, r)
        if validate(U, V, W, rank):
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break  # Found the best possible
    
    # Fallback to schoolbook if no valid decomposition found
    if best_U is None:
        best_rank = n * m * p
        best_U = np.zeros((n*m, best_rank))
        best_V = np.zeros((m*p, best_rank))
        best_W = np.zeros((p*n, best_rank))
        col = schoolbook_block(best_U, best_V, best_W, 0, range(n), range(m), range(p))
    
    # Convert to lists for the return interface
    U_list = [[int(best_U[i, c]) if abs(best_U[i, c] - round(best_U[i, c])) < 1e-9 else float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(best_V[i, c]) if abs(best_V[i, c] - round(best_V[i, c])) < 1e-9 else float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(best_W[i, c]) if abs(best_W[i, c] - round(best_W[i, c])) < 1e-9 else float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
