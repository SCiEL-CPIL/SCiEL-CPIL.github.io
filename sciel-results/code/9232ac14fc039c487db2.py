import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = 2, 5, 5
    
    # Use block decomposition: split 2x5x5 into 2x2x2 blocks
    # A is 2x5, B is 5x5, C = A*B is 2x5
    # Split A into [A1, A2] where A1 is 2x2, A2 is 2x3
    # Split B into [[B1, B2], [B3, B4]] where B1 is 2x2, B2 is 2x3, B3 is 3x2, B4 is 3x3
    # C = [A1*B1 + A2*B3, A1*B2 + A2*B4]
    
    # For the 2x2x2 part (A1*B1), use Strassen's algorithm (rank 7)
    # For the remaining parts, use schoolbook
    
    # Total rank: 7 (Strassen for 2x2x2) + 2*3*2 (A2*B3) + 2*3*3 (A2*B4)
    # Wait, let me recalculate. 
    # Actually, let's think of it differently.
    
    # A is 2x5, B is 5x5
    # Let's split columns of A and rows of B:
    # A = [A_left, A_right] where A_left is 2x2, A_right is 2x3
    # B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 3x5
    
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: 2x2 * 2x5 = 2x5, rank 2*2*5 = 20 (schoolbook)
    # A_right * B_bottom: 2x3 * 3x5 = 2x5, rank 2*3*5 = 30 (schoolbook)
    # Total: 50
    
    # Better: Use Strassen on the 2x2x2 sub-problem
    # Split A_left (2x2) and B_top (2x5) further
    # B_top = [B_tl, B_tr] where B_tl is 2x2, B_tr is 2x3
    
    # A_left * B_tl: 2x2 * 2x2 = 2x2, use Strassen (rank 7)
    # A_left * B_tr: 2x2 * 2x3 = 2x3, rank 2*2*3 = 12
    # A_right * B_bottom: 2x3 * 3x5 = 2x5, rank 2*3*5 = 30
    # Total: 7 + 12 + 30 = 49
    
    # Even better: apply Strassen more cleverly
    # Let's use a different decomposition
    
    # Actually, the measured improvement shows rank 46 is achievable.
    # Let me implement a known rank-46 decomposition for 2x5x5.
    
    # Based on literature, we can achieve rank 46 by:
    # 1. Using Strassen for the 2x2x2 core (saves 1 rank: 8->7)
    # 2. Optimizing the remaining terms
    
    # Let me construct the rank-46 decomposition:
    
    rank = 46
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Indices:
    # U: (i, j) where i in 0..1 (rows of A), j in 0..4 (cols of A)
    # V: (j, k) where j in 0..4 (rows of B), k in 0..4 (cols of B)  
    # W: (k, i) where k in 0..4 (rows of C), i in 0..1 (cols of C)
    # Output index: k*n + i = k*2 + i
    
    col = 0
    
    # Strassen's 7 terms for the 2x2x2 sub-problem (indices 0,1 for both)
    # Let A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # Let B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    
    # M1 = (A11+A22)(B11+B22)
    U[0*5+0, col] = 1; U[1*5+1, col] = 1
    V[0*5+0, col] = 1; V[1*5+1, col] = 1
    W[0*2+0, col] = 1; W[1*2+1, col] = 1; W[0*2+1, col] = -1; W[1*2+0, col] = -1
    col += 1
    
    # M2 = (A21+A22)B11
    U[1*5+0, col] = 1; U[1*5+1, col] = 1
    V[0*5+0, col] = 1
    W[0*2+1, col] = 1; W[1*2+1, col] = 1
    col += 1
    
    # M3 = A11(B12-B22)
    U[0*5+0, col] = 1
    V[0*5+1, col] = 1; V[1*5+1, col] = -1
    W[0*2+0, col] = 1; W[1*2+0, col] = 1
    col += 1
    
    # M4 = A22(B21-B11)
    U[1*5+1, col] = 1
    V[1*5+0, col] = 1; V[0*5+0, col] = -1
    W[0*2+1, col] = 1; W[1*2+1, col] = 1
    col += 1
    
    # M5 = (A11+A12)B22
    U[0*5+0, col] = 1; U[0*5+1, col] = 1
    V[1*5+1, col] = 1
    W[0*2+0, col] = 1; W[1*2+0, col] = 1
    col += 1
    
    # M6 = (A21-A11)(B11+B12)
    U[1*5+0, col] = 1; U[0*5+0, col] = -1
    V[0*5+0, col] = 1; V[0*5+1, col] = 1
    W[0*2+1, col] = 1; W[1*2+0, col] = 1
    col += 1
    
    # M7 = (A12-A22)(B21+B22)
    U[0*5+1, col] = 1; U[1*5+1, col] = -1
    V[1*5+0, col] = 1; V[1*5+1, col] = 1
    W[0*2+0, col] = 1; W[1*2+1, col] = 1
    col += 1
    
    # Now handle the remaining terms:
    # A[0,2:5] * B[0:2, :]  (2x3 * 2x5 = 2x5) - 15 terms
    # A[1,2:5] * B[0:2, :]  (2x3 * 2x5 = 2x5) - 15 terms  
    # A[0,2:5] * B[2:5, :]  (2x3 * 3x5 = 2x5) - 15 terms
    # A[1,2:5] * B[2:5, :]  (2x3 * 3x5 = 2x5) - 15 terms
    # Total: 60 terms, but we only have 46-7=39 left
    
    # This doesn't work. Let me use a different approach.
    
    # Reset and use a known rank-46 construction
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    
    # I'll use the fact that we can share some computations
    # For now, let me just return the schoolbook rank 50 as fallback
    # and note that rank 46 is achievable with proper optimization
    
    # Actually, let me implement a proper rank-46 decomposition
    # by carefully combining terms
    
    # Clear and restart
    U[:] = 0
    V[:] = 0
    W[:] = 0
    col = 0
    
    # Use a hybrid approach:
    # Split into 2x2x2 (Strassen, rank 7) and 2x3x5 terms
    # But optimize the 2x3x5 part
    
    # For A[:, 2:5] (2x3) * B[0:2, :] (2x5):
    # This is 2*2*5 = 20 terms schoolbook
    # But we can do better with block structure
    
    # Implement proper rank-46 decomposition
    # Strassen handles 2x2x2 (7 terms), remaining 39 terms for the rest
    
    # The remaining computation is:
    # C[:, 2:5] involves A[:, 2:5] * B[:, 2:5] plus cross terms
    # Let me use a systematic approach
    
    # Actually, based on the measured result, rank 46 is achievable.
    # Let me construct it properly by:
    # 1. Strassen for 2x2x2 core: 7 terms
    # 2. Remaining 39 terms for A[:,2:5]*B[0:2,:] and A[:,2:5]*B[2:5,:]
    
    # For A[:,2:5] (2x3) * B[0:2,:] (2x5): use 2*2*5=20 but optimize
    # For A[:,2:5] (2x3) * B[2:5,:] (3x5): use 2*3*5=30 but optimize
    
    # Total would be 7+20+30=57, still too high
    
    # Better approach: share terms between the two products
    # A[:,2:5] * [B[0:2,:]; B[2:5,:]] = A[:,2:5] * B[:,2:5] (2x3 * 3x5 = 30 terms)
    # Plus A[:,0:2] * B[2:5,:] (2x2 * 3x5 = 30 terms)
    # Total: 7 + 30 + 30 = 67, worse
    
    # I need to use a known rank-46 algorithm. Let me look up the construction.
    
    # For now, return a valid rank-50 decomposition as fallback
    rank = 50
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
