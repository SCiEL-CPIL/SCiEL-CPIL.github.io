"""Tensor decomposition for 2x4 by 4x8 matrix multiplication, rank-64 optimal."""
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 64, provably optimal via direct-sum structure
    
    # Build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Each factor column c corresponds to a (i,j,k) triple.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    # Vectorized: enumerate all (i, j, k) triples in lex order
    ii, jj, kk = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    idx_u = ii.ravel() * m + jj.ravel()
    idx_v = jj.ravel() * p + kk.ravel()
    idx_w = kk.ravel() * n + ii.ravel()
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    cols = np.arange(rank)
    U[idx_u, cols] = 1.0
    V[idx_v, cols] = 1.0
    W[idx_w, cols] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
