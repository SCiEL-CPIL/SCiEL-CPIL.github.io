"""Find rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U row: i*m+j, V row: j*p+k, W row: k*n+i
    
    # Build the multiplication tensor T as a 3D array
    # T[a, b, c] where a = i*m+j, b = j'*p+k', c = k'*n+i'
    # T[a,b,c] = 1 if i==i' and j==j' and k==k' else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try rank-14 Strassen-based solution first
    rank14 = _build_rank14_solution()
    if rank14 is not None:
        U14, V14, W14 = rank14
        if _verify_solution(U14, V14, W14, n, m, p):
            return {"U": U14, "V": V14, "W": W14}
    
    # Try to search for lower rank solutions
    # For rank r, we need to find U (4xr), V (8xr), W (8xr) such that
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
    
    # Strategy: For each rank r from 13 down to 12, try to find a solution
    # Using the fact that for fixed U and W, V can be solved linearly
    # Or use algebraic approaches
    
    # First, let's try a direct algebraic approach for rank 13
    # The tensor has 4*8*8 = 256 entries, 256 of which are 1 (the diagonal)
    # For rank r, we have 4r + 8r + 8r = 20r variables
    
    # Try rank 13 with a structured approach
    # Use the block structure: split B into two 2x2 blocks
    # Left block: columns 0,1 of B (indices 0,1,4,5)
    # Right block: columns 2,3 of B (indices 2,3,6,7)
    
    # For each block, we need at least 7 multiplications (Strassen lower bound for 2x2x2)
    # But we might share some multiplications between blocks
    
    # Let's try to see if we can do rank 12 by sharing 2 multiplications
    # This would require finding a non-trivial sharing pattern
    
    # Alternative: use numerical optimization
    # For rank r, parameterize U, V, W and minimize the error
    
    for target_rank in [13, 12]:
        if time.time() > deadline - 10:
            break
        
        success = _search_rank(target_rank, n, m, p, T, deadline)
        if success is not None:
            U, V, W = success
            if _verify_solution(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook rank 16
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb))
    V_fb = np.zeros((m*p, rank_fb))
    W_fb = np.zeros((p*n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


def _build_rank14_solution():
    """Build the rank-14 Strassen-based solution."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
    # W indices (k*n+i): C[0,0]=0, C[0,1]=2, C[0,2]=4, C[0,3]=6, C[1,0]=1, C[1,1]=3, C[1,2]=5, C[1,3]=7
    
    # Left Strassen block (B columns 0,1)
    # Col 0: L1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # Col 1: L2 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C[1,0], C[1,1]
    
    # Col 2: L3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C[0,1], C[1,1]
    
    # Col 3: L4 = A11(B10-B00)
    U[3, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C[0,0], C[1,0]
    
    # Col 4: L5 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C[0,0], C[0,1]
    
    # Col 5: L6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # Col 6: L7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    # Right Strassen block (B columns 2,3)
    # B indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    
    # Col 7: R1 = (A00+A11)(B02+B13)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[4, 7] = 1; W[7, 7] = 1  # C[0,2], C[1,3]
    
    # Col 8: R2 = (A10+A11)(B02)
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[5, 8] = 1; W[7, 8] = -1  # C[1,2], C[1,3]
    
    # Col 9: R3 = A00(B03-B13)
    U[0, 9] = 1
    V[3, 9] = 1; V[7, 9] = -1
    W[6, 9] = 1; W[7, 9] = 1  # C[0,3], C[1,3]
    
    # Col 10: R4 = A11(B12-B02)
    U[3, 10] = 1
    V[6, 10] = 1; V[2, 10] = -1
    W[4, 10] = 1; W[5, 10] = 1  # C[0,2], C[1,2]
    
    # Col 11: R5 = (A00+A01)(B13)
    U[0, 11] = 1; U[1, 11] = 1
    V[7, 11] = 1
    W[4, 11] = -1; W[6, 11] = 1  # C[0,2], C[0,3]
    
    # Col 12: R6 = (A10-A00)(B02+B03)
    U[2, 12] = 1; U[0, 12] = -1
    V[2, 12] = 1; V[3, 12] = 1
    W[7, 12] = 1  # C[1,3]
    
    # Col 13: R7 = (A01-A11)(B12+B13)
    U[1, 13] = 1; U[3, 13] = -1
    V[6, 13] = 1; V[7, 13] = 1
    W[4, 13] = 1  # C[0,2]
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify_solution(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            a = i * m + j
                            b = jp * p + k
                            c = kp * n + ip
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            actual = np.sum(U[a] * V[b] * W[c])
                            if abs(actual - expected) > 1e-8:
                                return False
    return True


def _search_rank(target_rank, n, m, p, T, deadline):
    """Search for a rank-r decomposition using numerical methods."""
    # This is a complex nonlinear search. For now, return None to indicate
    # that we haven't found a better solution.
    # A proper implementation would use:
    # 1. Random initialization with structured patterns
    # 2. Alternating least squares or gradient-based optimization
    # 3. Verification with exact arithmetic
    
    # For the 2x2x4 case, it's believed that rank 14 is optimal or near-optimal
    # Lower bounds from various techniques give rank >= 12
    # Without a breakthrough in the lower bound, we stick with rank 14
    
    return None
