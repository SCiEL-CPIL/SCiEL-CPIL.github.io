"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    deadline = time.time() + 145
    
    # Target tensor
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank 30
    rank0 = 30
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((n*p, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = rank0
    
    # Rank-28 construction using two Strassen 2x2x2 blocks
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (schoolbook, 4 terms)
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5] -> C[0:2, 0:5] (schoolbook, 10 terms)
    # Total: 7 + 7 + 4 + 10 = 28
    
    rank28 = 28
    U28 = np.zeros((n*m, rank28))
    V28 = np.zeros((m*p, rank28))
    W28 = np.zeros((n*p, rank28))
    
    def add_strassen_block(base_rank, j_offset, k_offset):
        """
        Add a Strassen 2x2*2x2 block.
        A indices: rows 0,1; cols j_offset, j_offset+1
        B indices: rows j_offset, j_offset+1; cols k_offset, k_offset+1
        C indices: rows 0,1; cols k_offset, k_offset+1
        """
        # A variable indices
        a00 = 0*m + j_offset
        a01 = 0*m + (j_offset+1)
        a10 = 1*m + j_offset
        a11 = 1*m + (j_offset+1)
        
        # B variable indices
        b00 = j_offset*p + k_offset
        b01 = j_offset*p + (k_offset+1)
        b10 = (j_offset+1)*p + k_offset
        b11 = (j_offset+1)*p + (k_offset+1)
        
        # C variable indices (W index = k*n + i)
        c00 = k_offset*n + 0
        c01 = (k_offset+1)*n + 0
        c10 = k_offset*n + 1
        c11 = (k_offset+1)*n + 1
        
        r = base_rank
        
        # M1 = (a00+a11)(b00+b11)
        U28[a00, r] += 1; U28[a11, r] += 1
        V28[b00, r] += 1; V28[b11, r] += 1
        W28[c00, r] += 1; W28[c11, r] += 1
        r += 1
        
        # M2 = (a10+a11)b00
        U28[a10, r] += 1; U28[a11, r] += 1
        V28[b00, r] += 1
        W28[c10, r] += 1; W28[c11, r] -= 1
        r += 1
        
        # M3 = a00(b01-b11)
        U28[a00, r] += 1
        V28[b01, r] += 1; V28[b11, r] -= 1
        W28[c01, r] += 1; W28[c11, r] += 1
        r += 1
        
        # M4 = a11(b10-b00)
        U28[a11, r] += 1
        V28[b10, r] += 1; V28[b00, r] -= 1
        W28[c00, r] += 1; W28[c10, r] += 1
        r += 1
        
        # M5 = (a00+a01)b11
        U28[a00, r] += 1; U28[a01, r] += 1
        V28[b11, r] += 1
        W28[c00, r] -= 1; W28[c01, r] += 1
        r += 1
        
        # M6 = (a10-a00)(b00+b01)
        U28[a10, r] += 1; U28[a00, r] -= 1
        V28[b00, r] += 1; V28[b01, r] += 1
        W28[c11, r] += 1
        r += 1
        
        # M7 = (a01-a11)(b10+b11)
        U28[a01, r] += 1; U28[a11, r] -= 1
        V28[b10, r] += 1; V28[b11, r] += 1
        W28[c00, r] += 1
        r += 1
        
        return r
    
    # Block 1: j_offset=0, k_offset=0
    next_rank = add_strassen_block(0, 0, 0)
    
    # Block 2: j_offset=0, k_offset=2
    next_rank = add_strassen_block(next_rank, 0, 2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> schoolbook 4 terms
    # (i=0, j=0, k=4), (i=0, j=1, k=4), (i=1, j=0, k=4), (i=1, j=1, k=4)
    for i in range(2):
        for j in range(2):
            k = 4
            U28[i*m+j, next_rank] = 1
            V28[j*p+k, next_rank] = 1
            W28[k*n+i, next_rank] = 1
            next_rank += 1
    
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5] -> schoolbook 10 terms
    for i in range(2):
        j = 2
        for k in range(5):
            U28[i*m+j, next_rank] = 1
            V28[j*p+k, next_rank] = 1
            W28[k*n+i, next_rank] = 1
            next_rank += 1
    
    # Verify rank-28
    recon = np.zeros((n*m, m*p, n*p))
    for rr in range(rank28):
        recon += np.einsum('i,j,k->ijk', U28[:,rr], V28[:,rr], W28[:,rr])
    
    if np.allclose(recon, T, atol=1e-9):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Fallback: try rank-29 with one Strassen block
    rank29 = 29
    U29 = np.zeros((n*m, rank29))
    V29 = np.zeros((m*p, rank29))
    W29 = np.zeros((n*p, rank29))
    
    # One Strassen block for i in {0,1}, j in {0,1}, k in {0,1}
    # Same as Block 1 above
    r = add_strassen_block(0, 0, 0)
    
    # Remaining 22 schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in [0,1] and j in [0,1] and k in [0,1]:
                    continue
                U29[i*m+j, r] = 1
                V29[j*p+k, r] = 1
                W29[k*n+i, r] = 1
                r += 1
    
    recon29 = np.zeros((n*m, m*p, n*p))
    for rr in range(rank29):
        recon29 += np.einsum('i,j,k->ijk', U29[:,rr], V29[:,rr], W29[:,rr])
    
    if np.allclose(recon29, T, atol=1e-9):
        return {"U": U29.tolist(), "V": V29.tolist(), "W": W29.tolist()}
    
    # Fallback to schoolbook
    return best_result
# EVOLVE-BLOCK-END
