"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (24, rank) 
    # W shape: (p*n, rank) = (18, rank)
    # W[k*n+i, c] corresponds to output C[i,k]
    
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Try to find a low-rank decomposition
    # Start with schoolbook rank and try to reduce
    
    # For 3x4 * 4x6, let's try rank = 30
    # Construction: use the fact that we can share terms
    
    # Method: express the tensor as sum of rank-1 tensors
    # T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where (U[:,c])_{i*m+j} = u_{i,j}, (V[:,c])_{j*p+k} = v_{j,k}, (W[:,c])_{k*n+i} = w_{i,k}
    
    # For each (i,j,k), we need: sum_c u_{i,j} * v_{j,k} * w_{i,k} = 1 if this is the (i,j,k) term, 0 otherwise
    
    # This is a hard problem. Let me use a known construction.
    
    # For m×n × n×p, one construction uses rank = m*p + (n-1)*(min(m,p)-1) + ... 
    # Let me just try to construct a valid decomposition with a reasonable rank.
    
    # Use the "border rank" approach or direct construction
    
    # Actually, let me use a simple but effective construction:
    # Group by the middle index j
    
    # For each j, we have A[:,j] (3-vector) and B[j,:] (6-vector)
    # C += A[:,j] ⊗ B[j,:]
    # This is rank 3*6 = 18 per j, total 4*18 = 72 (schoolbook)
    
    # But we can do better by combining.
    
    # Let me try rank = 30 with a structured approach.
    
    # Use the following: express the product using shared bilinear forms.
    
    # I'll use a greedy/iterative approach to find a low-rank decomposition.
    
    # For now, return a valid decomposition with rank = 30
    # using a structured construction.
    
    rank = 30
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Construction: for each (i,k), we need sum_j A[i,j]*B[j,k]
    # Use the fact that we can write this as a sum of rank-1 terms.
    
    # Let me use a different approach: use the known result that
    # the bilinear complexity of 3x4 * 4x6 is at most 30.
    
    # I'll construct it explicitly.
    
    # Terms 0-17: handle the "main" products
    # For i in range(3), k in range(6):
    #   We need to capture sum_j A[i,j]*B[j,k]
    
    # Use a structured decomposition based on the following idea:
    # For each pair (i,k), introduce terms that capture the contributions.
    
    # Let me use a simpler approach: just use rank = 24 (the max of m*p, n*p, m*n)
    # and see if it works. Actually, the rank is at least max(m*p, m*n, n*p) = 24.
    
    # Try rank = 24
    rank = 24
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # For each (i,k), assign one rank-1 term
    # But we need to capture all j contributions.
    
    # This won't work with rank 24 alone.
    
    # Let me use rank = 30 and construct it properly.
    
    # I'll use the following construction:
    # The matrix multiplication tensor can be decomposed using the
    # following structured approach.
    
    # For now, let me return the schoolbook decomposition as a fallback
    # and try to improve within the time budget.
    
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        # Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
        # and 0 otherwise (but the tensor only has 1s at (i,j,k) positions)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Also check that there are no spurious terms
        # The tensor M_{n,m,p} has T[i,j,k] = 1 if the indices match the multiplication pattern
        # Actually, the tensor is defined as T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where
        # the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
        # We need to check that for all (a,b,c) where a=i*m+j, b=j'*p+k, c=k'*n+i',
        # sum_d U[a,d]*V[b,d]*W[c,d] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # with consistent indices, else 0.
        
        # More precisely: the tensor has non-zero entries only at positions
        # (i*m+j, j*p+k, k*n+i) for i in [0,n), j in [0,m), k in [0,p)
        # and the value is 1.
        
        # So we need: for all (a,b,c), sum_d U[a,d] V[b,d] W[c,d] = 1 if there exist
        # i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i, else 0.
        
        for a in range(n*m):
            i = a // m
            j = a % m
            for b in range(m*p):
                j2 = b // p
                k = b % p
                for c in range(p*n):
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    s = 0.0
                    for d in range(U.shape[1]):
                        s += U[a,d] * V[b,d] * W[c,d]
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # This should not happen for schoolbook
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
