import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (14, rank)
    
    # Indexing conventions:
    # U[i*m + j, t] -> A[i,j]
    # V[j*p + k, t] -> B[j,k]
    # W[k*n + i, t] -> C[i,k] (Output factor follows W[k*n+i,r])
    
    # We use a block decomposition.
    # Split A (2x4) into left (2x2) and right (2x2) based on j: 0,1 and 2,3
    # Split B (4x7) into top (2x7) and bottom (2x7) based on j: 0,1 and 2,3
    # The multiplication is C = A_L * B_top + A_R * B_bottom
    
    # For each term (A_block * B_block), we have 2x2 * 2x7.
    # We can decompose the 2x7 multiplication.
    # Let's use a combination of Strassen-like for 2x2 and schoolbook for the rest,
    # or a specific low-rank decomposition for 2x2*2x7.
    
    # Rank of 2x2*2x7:
    # Naive: 2*2*7 = 28.
    # Strassen-like for 2x2 part: 7 mults for each 2x2 column block.
    # There are 3 full 2x2 column blocks and one 2x1 column block.
    # 3 * 7 + 4 (schoolbook for 2x1) = 21 + 4 = 25 per A_block.
    # Total rank = 2 * 25 = 50.
    
    # Can we do better?
    # Maybe mix the A blocks?
    # A is 2x4. B is 4x7.
    # Total rank 50 is a strong baseline.
    
    # Let's try to optimize the 2x2 * 2x7 block.
    # Actually, the 2x1 block can be handled with 4 mults (schoolbook).
    # Is there a rank 3 decomposition for 2x1? No, min rank is 2?
    # 2x1 matrix M = u v^T. Rank 1.
    # Wait, the tensor is multiplication of a 2x1 matrix by a 1x1 matrix?
    # No, the block is A[2x2] * B[2x1] -> C[2x1].
    # Tensor size 2*2*1 = 4.
    # Rank of 2x2*2x1 multiplication tensor?
    # C0 = A00 B00 + A01 B10
    # C1 = A10 B00 + A11 B10
    # This is just matrix-vector multiply.
    # Rank is 4? 
    # u1 v1 + u2 v2 + u3 v3 + u4 v4.
    # Can we do it in 3?
    # It's known that rank of 2x2*2x1 is 4.
    
    # So 25 per block seems tight for that splitting.
    
    # Let's try to verify the rank 50 solution first.
    
    rank = 50
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_2x2x2(A_start, B_start, C_start):
        """
        Adds rank 7 terms for A[2x2] * B[2x2] -> C[2x2]
        A indices: rows i, cols j. A_start = (i0, j0)
        B indices: rows j, cols k. B_start = (j0, k0) -- Note: j0 must match A's j0
        C indices: rows i, cols k. C_start = (i0, k0)
        """
        nonlocal col
        
        i0, j0 = A_start
        k0 = C_start[1]
        # B_start is determined by j0 and k0
        # B indices are (j, k). So B row index starts at j0, col index at k0.
        
        # U indices (A):
        # A[0,0] -> U[0*m + 0] relative to block? No, absolute index.
        # U index = i * m + j
        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)
        
        # V indices (B):
        # V index = j * p + k
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        
        # W indices (C):
        # W index = k * n + i
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # Strassen:
        # m1 = (a00 + a11)(b00 + b11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # m2 = a11(b00 - b10)
        U[a11, col] += 1
        V[b00, col] += 1; V[b10, col] -= 1
        W[c10, col] += 1; W[c11, col] += 1
        col += 1
        
        # m3 = (a00 - a10)b10
        U[a00, col] += 1; U[a10, col] -= 1
        V[b10, col] += 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # m4 = (a10 + a01)b00
        U[a10, col] += 1; U[a01, col] += 1
        V[b00, col] += 1
        W[c00, col] += 1; W[c01, col] += 1
        col += 1
        
        # m5 = (a01 - a11)(b10 + b11)
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c01, col] += 1
        col += 1
        
        # m6 = (a00 + a01)(b11 - b01)
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1; V[b01, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # m7 = (a10 + a11)(b01 - b00)
        U[a10, col] += 1; U[a11, col] += 1
        V[b01, col] += 1; V[b00, col] -= 1
        W[c10, col] += 1; W[c11, col] += 1
        col += 1
        
    def add_schoolbook_2x2x1(A_start, B_start, C_start):
        """
        Adds rank 4 terms for A[2x2] * B[2x1] -> C[2x1]
        """
        nonlocal col
        
        i0, j0 = A_start
        k0 = C_start[1]
        
        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)
        
        b00 = j0 * p + k0
        b10 = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        # C0 = a00 b00 + a01 b10
        # C1 = a10 b00 + a11 b10
        
        U[a00, col] += 1; V[b00, col] += 1; W[c00, col] += 1; col += 1
        U[a01, col] += 1; V[b10, col] += 1; W[c00, col] += 1; col += 1
        U[a10, col] += 1; V[b00, col] += 1; W[c10, col] += 1; col += 1
        U[a11, col] += 1; V[b10, col] += 1; W[c10, col] += 1; col += 1
        
    # Block 1: A[0:2, 0:2] * B[0:2, :]
    # A starts at (0,0). B starts at (0,0) for cols, but we split B cols.
    # B is 4x7. Rows 0,1 are the top block.
    # So for j0=0, we process k=0..6.
    # k=0,1 -> Strassen
    # k=2,3 -> Strassen
    # k=4,5 -> Strassen
    # k=6   -> Schoolbook
    
    for k0 in [0, 2, 4]:
        add_strassen_2x2x2((0, 0), (0, k0), (0, k0))
        
    add_schoolbook_2x2x1((0, 0), (0, 6), (0, 6))
    
    # Block 2: A[0:2, 2:4] * B[2:4, :]
    # A starts at (0,2). B rows start at 2.
    # j0=2.
    for k0 in [0, 2, 4]:
        add_strassen_2x2x2((0, 2), (2, k0), (0, k0))
        
    add_schoolbook_2x2x1((0, 2), (2, 6), (0, 6))
    
    # Check col
    if col != rank:
        # Adjust if needed, but 50 is expected
        pass
        
    return {"U": U, "V": V, "W": W}
