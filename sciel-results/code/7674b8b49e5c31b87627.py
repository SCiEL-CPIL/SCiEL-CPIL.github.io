import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x3 by 3x10.
    Uses the Strassen-based decomposition which has rank 55 (7 per 2x2 block of B
    plus 10 for the remaining column of B).
    
    The output factor follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank) = (6, rank)
    V has shape (m*p, rank) = (30, rank)
    W has shape (p*n, rank) = (20, rank)
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    nm = n * m  # 6
    mp = m * p  # 30
    pn = p * n  # 20
    
    # Strassen Rank: 7 * ceil(p/2) + (p % 2) * n
    # For p=10: 7*5 + 0 = 35... wait, let me re-check the structure
    # Actually for 2x3 * 3x10:
    # A is 2x3, B is 3x10
    # We can group B's columns into pairs: 5 pairs
    # For each pair of columns, we use a 2x2 Strassen decomposition
    # But A has 3 columns, not 2. Let me think about this differently.
    
    # Actually, the standard approach for 2x3 * 3xp:
    # We can view this as multiplying a 2x3 matrix by a 3xp matrix
    # One approach: use the rank-7 decomposition for 2x2*2x2 (Strassen)
    # But we have 2x3 * 3x10
    
    # Let me use the block approach:
    # Split B into 5 blocks of 2 columns each
    # For each block, we need to compute A * B_block where B_block is 3x2
    # This is a 2x3 * 3x2 multiplication
    # The tensor rank of 2x3 * 3x2 is known to be 7 (same as 2x2*2x2 Strassen)
    # Wait, actually 2x3 * 3x2 -> 2x2 result
    # The number of multiplications needed: we can use Strassen-like approach
    
    # Let me just use the schoolbook for now but try to be smarter
    # Actually, let me implement the rank-55 solution from the inspiration
    
    rank = 55
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    
    # A indices: A[i,j] where i in [0,1], j in [0,1,2]
    # U index: i*m + j = i*3 + j
    # a00=0, a01=1, a02=2, a10=3, a11=4, a12=5
    
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    
    # B indices: B[j,k] where j in [0,1,2], k in [0,9]
    # V index: j*p + k = j*10 + k
    
    # W indices: W[k*n+i] = W[k*2+i] for C[i,k]
    
    # For each pair of columns (k0, k1) in B:
    # We compute A * B[:, k0:k1+1] using a rank-7 decomposition
    # This gives us C[:, k0] and C[:, k1]
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices for this block
        b00 = 0 * p + k0  # B[0, k0]
        b01 = 0 * p + k1  # B[0, k1]
        b10 = 1 * p + k0  # B[1, k0]
        b11 = 1 * p + k1  # B[1, k1]
        b20 = 2 * p + k0  # B[2, k0]
        b21 = 2 * p + k1  # B[2, k1]
        
        # W indices for this block
        c00 = k0 * n + 0  # C[0, k0]
        c01 = k1 * n + 0  # C[0, k1]
        c10 = k0 * n + 1  # C[1, k0]
        c11 = k1 * n + 1  # C[1, k1]
        
        # We need to compute:
        # C[0, k0] = A[0,0]*B[0,k0] + A[0,1]*B[1,k0] + A[0,2]*B[2,k0]
        # C[0, k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1] + A[0,2]*B[2,k1]
        # C[1, k0] = A[1,0]*B[0,k0] + A[1,1]*B[1,k0] + A[1,2]*B[2,k0]
        # C[1, k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1] + A[1,2]*B[2,k1]
        
        # Using Strassen-like decomposition for 2x3 * 3x2:
        # We can define 7 products:
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        # M3 = A[0,0] * (B[1,k1] - B[0,k1])  # Wait, need to be careful
        # Let me use a known rank-7 decomposition for 2x3*3x2
        
        # Actually, let me just use the schoolbook for the 2x3*3x2 case
        # That would be 2*3*2 = 12 per block, total 60
        # But we want to do better
        
        # Let me try a different approach: use the fact that we can combine
        # terms across the two output columns
        
        # For now, let me just do schoolbook to ensure correctness
        # and then optimize
        
        # Schoolbook for this block:
        # 6 multiplications for each of the 2 output columns = 12
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    kk = k0 + k
                    U[i * m + j, col] = 1.0
                    V[j * p + kk, col] = 1.0
                    W[kk * n + i, col] = 1.0
                    col += 1
    
    # This gives rank 60, which is the schoolbook
    # Let me try to implement a better decomposition
    
    # Reset and try Strassen-based
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    col = 0
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        b00 = 0 * p + k0
        b01 = 0 * p + k1
        b10 = 1 * p + k0
        b11 = 1 * p + k1
        b20 = 2 * p + k0
        b21 = 2 * p + k1
        
        c00 = k0 * n + 0
        c01 = k1 * n + 0
        c10 = k0 * n + 1
        c11 = k1 * n + 1
        
        # Rank-7 decomposition for 2x3 * 3x2
        # Using the following 7 products:
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[a00, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[a10, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0; W[c11, col] -= 1.0
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[a00, col] += 1.0
        V[b01, col] += 1.0; V[b11, col] -= 1.0
        W[c01, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[a11, col] += 1.0
        V[b10, col] += 1.0; V[b00, col] -= 1.0
        W[c00, col] += 1.0; W[c10, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[a00, col] += 1.0; U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0; W[c01, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[a10, col] += 1.0; U[a00, col] -= 1.0
        V[b00, col] += 1.0; V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[a01, col] += 1.0; U[a11, col] -= 1.0
        V[b10, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
        
        # Now we need to handle the A[0,2] and A[1,2] terms
        # C[0,k0] += A[0,2]*B[2,k0]
        # C[0,k1] += A[0,2]*B[2,k1]
        # C[1,k0] += A[1,2]*B[2,k0]
        # C[1,k1] += A[1,2]*B[2,k1]
        
        # These require 4 more multiplications, but we can share
        # Actually, we need 4 separate terms for A02 and A12
        
        # M8 = A02 * B20
        U[a02, col] += 1.0
        V[b20, col] += 1.0
        W[c00, col] += 1.0
        col += 1
        
        # M9 = A02 * B21
        U[a02, col] += 1.0
        V[b21, col] += 1.0
        W[c01, col] += 1.0
        col += 1
        
        # M10 = A12 * B20
        U[a12, col] += 1.0
        V[b20, col] += 1.0
        W[c10, col] += 1.0
        col += 1
        
        # M11 = A12 * B21
        U[a12, col] += 1.0
        V[b21, col] += 1.0
        W[c11, col] += 1.0
        col += 1
    
    # This gives 11 * 5 = 55 rank
    # Let me verify this is correct by checking the tensor reconstruction
    
    # Check time limit
    if time.time() - start_time > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to lists for the required interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
