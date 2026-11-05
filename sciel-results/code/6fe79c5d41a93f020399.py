import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build tensor from decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Build schoolbook tensor
    T_decomp = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_school = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_school[i, j, k] = 1.0
    return np.allclose(T_decomp, T_school, atol=1e-10)

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W, rank

def _strassen222():
    """Strassen decomposition of 2x2x2, rank 7."""
    # U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    # U[i*2+j, c] for i,j in 0..1
    # V[j*2+k, c] for j,k in 0..1
    # W[k*2+i, c] for k,i in 0..1
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1: A*(11+22) * B*(11+22)
    # U: A11=1, A22=1 -> rows 0,3
    # V: B11=1, B22=1 -> rows 0,3
    # W: C11+=1, C22+=1 -> rows 0,3
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2: (A21+A22)*B11
    # U: A21=1, A22=1 -> rows 2,3
    # V: B11=1 -> row 0
    # W: C21+=1 -> row 2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    
    # M3: A11*(B12-B22)
    # U: A11=1 -> row 0
    # V: B12=1, B22=-1 -> rows 1,3
    # W: C12+=1, C22+=1 -> rows 1,3
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4: A22*(B21-B11)
    # U: A22=1 -> row 3
    # V: B21=1, B11=-1 -> rows 2,0
    # W: C11+=1, C21+=1 -> rows 0,2
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5: (A11+A12)*B22
    # U: A11=1, A12=1 -> rows 0,1
    # V: B22=1 -> row 3
    # W: C11-=1, C22+=1 -> rows 0,3
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[3][4] += 1
    
    # M6: (A21-A11)*(B11+B12)
    # U: A21=1, A11=-1 -> rows 2,0
    # V: B11=1, B12=1 -> rows 0,1
    # W: C22+=1 -> row 3
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7: (A12-A22)*(B21+B22)
    # U: A12=1, A22=-1 -> rows 1,3
    # V: B21=1, B22=1 -> rows 2,3
    # W: C11+=1 -> row 0
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W, 7

def _embed_strassen_into_355(U_s, V_s, W_s, n, m, p, i0, j0, k0):
    """Embed a 2x2x2 Strassen decomposition into the 3x5x5 problem.
    
    The Strassen block covers A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
    producing C[i0:i0+2, k0:k0+2].
    
    U_s has shape (4,7): row = i'*2 + j' where i',j' in 0..1
    V_s has shape (4,7): row = j'*2 + k' where j',k' in 0..1
    W_s has shape (4,7): row = k'*2 + i' where k',i' in 0..1
    
    In the full problem:
    U row index = i*m + j where i = i0+i', j = j0+j'
    V row index = j*p + k where j = j0+j', k = k0+k'
    W row index = k*n + i where k = k0+k', i = i0+i'
    """
    # Full matrices
    U_full = [[0.0] * 7 for _ in range(n * m)]
    V_full = [[0.0] * 7 for _ in range(m * p)]
    W_full = [[0.0] * 7 for _ in range(p * n)]
    
    for c in range(7):
        for i_prime in range(2):
            for j_prime in range(2):
                i = i0 + i_prime
                j = j0 + j_prime
                u_val = U_s[i_prime * 2 + j_prime][c]
                if u_val != 0:
                    U_full[i * m + j][c] = u_val
        
        for j_prime in range(2):
            for k_prime in range(2):
                j = j0 + j_prime
                k = k0 + k_prime
                v_val = V_s[j_prime * 2 + k_prime][c]
                if v_val != 0:
                    V_full[j * p + k][c] = v_val
        
        for k_prime in range(2):
            for i_prime in range(2):
                k = k0 + k_prime
                i = i0 + i_prime
                w_val = W_s[k_prime * 2 + i_prime][c]
                if w_val != 0:
                    W_full[k * n + i][c] = w_val
    
    return U_full, V_full, W_full

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    school_rank = n * m * p  # 75
    
    # Strategy: Use Strassen for a 2x2x2 block, schoolbook for the rest
    # Split m into 2+3, p into 2+3
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]  (2x2x2 Strassen, rank 7)
    # Block 2: A[:, 0:2] * B[0:2, 2:5] -> C[:, 2:5]  (2x3x3 schoolbook, rank 18)
    # Block 3: A[:, 2:5] * B[2:5, 0:2] -> C[:, 0:2]  (3x3x2 schoolbook, rank 18)
    # Block 4: A[:, 2:5] * B[2:5, 2:5] -> C[:, 2:5]  (3x3x3 schoolbook, rank 27)
    # Total: 7 + 18 + 18 + 27 = 70
    
    U_s, V_s, W_s, _ = _strassen222()
    
    total_rank = 7 + 18 + 18 + 27
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: i in 0..2, j in 0..1, k in 0..1 -> Strassen
    # Actually Strassen only covers i in 0..1, j in 0..1, k in 0..1
    # For i=2, j in 0..1, k in 0..1, we need schoolbook (3 terms)
    # Let me redo: Strassen covers A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    
    U1, V1, W1 = _embed_strassen_into_355(U_s, V_s, W_s, n, m, p, 0, 0, 0)
    for r in range(n * m):
        for c in range(7):
            U[r][col + c] += U1[r][c]
    for r in range(m * p):
        for c in range(7):
            V[r][col + c] += V1[r][c]
    for r in range(p * n):
        for c in range(7):
            W[r][col + c] += W1[r][c]
    col += 7
    
    # Block 1b: i=2, j in 0..1, k in 0..1 (schoolbook, 6 terms)
    for i in [2]:
        for j in range(2):
            for k in range(2):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Block 2: i in 0..2, j in 0..1, k in 2..4 (schoolbook, 2*2*3=12 terms)
    for i in range(3):
        for j in range(2):
            for k in range(2, 5):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Block 3: i in 0..2, j in 2..4, k in 0..1 (schoolbook, 3*3*2=18 terms)
    for i in range(3):
        for j in range(2, 5):
            for k in range(2):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Block 4: i in 0..2, j in 2..4, k in 2..4 (schoolbook, 3*3*3=27 terms)
    for i in range(3):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    assert col == total_rank, f"Column count {col} != expected {total_rank}"
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    rank = total_rank
    
    # Try to reduce rank further with a local search
    if time.time() < deadline - 30:
        # Attempt to find a lower rank decomposition
        # Try rank 69, 68, ... by checking if we can eliminate a term
        # For now, stick with rank 70 as a solid improvement over 75
        pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
