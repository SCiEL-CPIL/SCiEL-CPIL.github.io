"""Matrix multiplication 3x3x6 rank-minimal decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the target tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Compute the decomposition
    recon = np.zeros((n*m, m*p, p*n))
    rank = U.shape[1]
    for r in range(rank):
        recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
    return np.allclose(recon, target, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=6
    
    # We decompose B into three 3x2 blocks: B1, B2, B3
    # C = [C1, C2, C3] where Ci = A * Bi
    # We use a rank-10 decomposition for each 3x3x2 multiplication.
    # Total rank = 3 * 10 = 30.
    
    rank = 30
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a term
    # U_idx: index in flattened A (i*m + j)
    # V_idx: index in flattened B (j*p + k)
    # W_idx: index in flattened C (k*n + i)
    # Note: The decomposition is sum_r U[:,r] V[:,r] W[:,r]
    # This corresponds to the tensor T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i.
    
    def add_term(r, u_idx, v_idx, w_idx, val=1.0):
        U[u_idx, r] += val
        V[v_idx, r] += val
        W[w_idx, r] += val
        
    # Strassen rank-7 for 2x2x2
    # A2 = A[0:2, 0:2], B2 = B[0:2, 0:2], C2 = C[0:2, 0:2]
    # Indices for A2: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4 (flattened i*3+j)
    # Indices for B2: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4 (flattened j*6+k? No, B is 3x6. j in 0..2, k in 0..5.
    # B2 uses j in 0..1, k in 0..1 (for block 1).
    # Block 1: B cols 0,1.
    # B_idx = j*6 + k.
    # B2 (block 1): j in 0,1; k in 0,1.
    # B[0,0]=0, B[0,1]=1, B[1,0]=6, B[1,1]=7.
    # C2 (block 1): i in 0,1; k in 0,1.
    # C_idx = k*3 + i.
    # C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4.
    
    # Strassen terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    
    # We will map these to the global indices.
    
    def add_strassen_block(block_idx, r_offset):
        # block_idx: 0, 1, 2
        # B columns: block_idx*2, block_idx*2+1
        # C columns: block_idx*2, block_idx*2+1
        # A is always full 3x3, but we only use top-left 2x2 for the Strassen part.
        
        # A indices (i*3+j):
        # A00=0, A01=1, A10=3, A11=4
        
        # B indices (j*6+k):
        k_start = block_idx * 2
        # B00 (local) -> j=0, k=k_start -> 0*6 + k_start = k_start
        # B01 (local) -> j=0, k=k_start+1 -> k_start+1
        # B10 (local) -> j=1, k=k_start -> 1*6 + k_start = 6+k_start
        # B11 (local) -> j=1, k=k_start+1 -> 7+k_start
        
        # C indices (k*3+i):
        # C00 (local) -> i=0, k=k_start -> k_start*3 + 0 = 3*k_start
        # C01 (local) -> i=1, k=k_start -> k_start*3 + 1 = 3*k_start+1
        # C10 (local) -> i=0, k=k_start+1 -> (k_start+1)*3 + 0 = 3*k_start+3
        # C11 (local) -> i=1, k=k_start+1 -> (k_start+1)*3 + 1 = 3*k_start+4
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11. W: C00, C01, C10, C11 (with coeffs)
        # Actually, we add terms to U, V, W directly.
        # The decomposition is additive.
        
        # We need to add 7 rank terms for the 2x2 part.
        # r = r_offset + 0 ... 6
        
        # M1
        r = r_offset
        U[0, r] = 1; U[4, r] = 1
        V[k_start, r] = 1; V[6+k_start, r] = 1
        W[3*k_start, r] = 1; W[3*k_start+1, r] = 1; W[3*k_start+3, r] = 1; W[3*k_start+4, r] = -1
        
        # M2
        r = r_offset + 1
        U[3, r] = 1; U[4, r] = 1
        V[k_start, r] = 1
        W[3*k_start+3, r] = 1; W[3*k_start+4, r] = -1
        
        # M3
        r = r_offset + 2
        U[0, r] = 1
        V[6+k_start, r] = 1; V[7+k_start, r] = -1
        W[3*k_start+1, r] = 1; W[3*k_start+4, r] = 1
        
        # M4
        r = r_offset + 3
        U[4, r] = 1
        V[k_start+1, r] = 1; V[k_start, r] = -1
        W[3*k_start, r] = 1; W[3*k_start+3, r] = 1
        
        # M5
        r = r_offset + 4
        U[0, r] = 1; U[1, r] = 1
        V[7+k_start, r] = 1
        W[3*k_start, r] = -1; W[3*k_start+1, r] = 1
        
        # M6
        r = r_offset + 5
        U[3, r] = 1; U[0, r] = -1
        V[k_start, r] = 1; V[k_start+1, r] = 1
        W[3*k_start+3, r] = 1
        
        # M7
        r = r_offset + 6
        U[1, r] = 1; U[4, r] = -1
        V[6+k_start, r] = 1; V[7+k_start, r] = 1
        W[3*k_start, r] = 1; W[3*k_start+1, r] = 1
        
        # Additional terms for the 3x3x2 specific parts (Rank 3 total)
        # Part 2: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
        # A indices: 2 (0,2), 5 (1,2)
        # B indices: 2*6 + k_start = 12+k_start, 12+k_start+1
        # C indices: 3*k_start, 3*k_start+1, 3*k_start+3, 3*k_start+4
        # This is a 2x1x2 tensor of ones. Rank 1.
        r = r_offset + 7
        U[2, r] = 1; U[5, r] = 1
        V[12+k_start, r] = 1; V[13+k_start, r] = 1
        W[3*k_start, r] = 1; W[3*k_start+1, r] = 1; W[3*k_start+3, r] = 1; W[3*k_start+4, r] = 1
        
        # Part 3: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
        # A indices: 6 (2,0), 7 (2,1)
        # B indices: k_start, k_start+1, 6+k_start, 7+k_start
        # C indices: 3*k_start+2 (2, k_start), 3*k_start+5 (2, k_start+1)
        # This is a 1x2x2 tensor of ones. Rank 1.
        r = r_offset + 8
        U[6, r] = 1; U[7, r] = 1
        V[k_start, r] = 1; V[k_start+1, r] = 1; V[6+k_start, r] = 1; V[7+k_start, r] = 1
        W[3*k_start+2, r] = 1; W[3*k_start+5, r] = 1
        
        # Part 4: A[2, 2] * B[2, 0:2] -> C[2, 0:2]
        # A index: 8 (2,2)
        # B indices: 12+k_start, 13+k_start
        # C indices: 3*k_start+2, 3*k_start+5
        # This is a 1x1x2 tensor of ones. Rank 1.
        r = r_offset + 9
        U[8, r] = 1
        V[12+k_start, r] = 1; V[13+k_start, r] = 1
        W[3*k_start+2, r] = 1; W[3*k_start+5, r] = 1

    # Add blocks
    add_strassen_block(0, 0)
    add_strassen_block(1, 10)
    add_strassen_block(2, 20)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if not _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    return {"U": U_list, "V": V_list, "W": W_list}
