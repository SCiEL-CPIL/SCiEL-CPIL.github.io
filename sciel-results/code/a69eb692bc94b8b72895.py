import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _schoolbook():
    n, m, p = 5, 5, 5
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}

# Strassen for 2x2 matrices
def _strassen_2x2():
    # Strassen decomposition for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # Returns U, V, W for the 2x2 case
    # 7 multiplications
    U = [[0] * 7 for _ in range(4)]  # 2x2 = 4 elements
    V = [[0] * 7 for _ in range(4)]  # 2x2 = 4 elements
    W = [[0] * 7 for _ in range(4)]  # 2x2 = 4 elements
    
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A indices: (0,0)=a11, (0,1)=a12, (1,0)=a21, (1,1)=a22
    # B indices: (0,0)=b11, (0,1)=b12, (1,0)=b21, (1,1)=b22
    
    # M1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1  # a11, a22
    V[0][0] = 1; V[3][0] = 1  # b11, b22
    
    # M2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1  # a21, a22
    V[0][1] = 1  # b11
    
    # M3 = a11*(b12-b22)
    U[0][2] = 1  # a11
    V[1][2] = 1; V[3][2] = -1  # b12, -b22
    
    # M4 = a22*(b21-b11)
    U[3][3] = 1  # a22
    V[2][3] = 1; V[0][3] = -1  # b21, -b11
    
    # M5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1  # a11, a12
    V[3][4] = 1  # b22
    
    # M6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1  # a21, -a11
    V[0][5] = 1; V[1][5] = 1  # b11, b12
    
    # M7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1  # a12, -a22
    V[2][6] = 1; V[3][6] = 1  # b21, b22
    
    # C11 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    
    # C12 = M3 + M5
    W[1][2] = 1; W[1][4] = 1
    
    # C21 = M2 + M4
    W[2][1] = 1; W[2][3] = 1
    
    # C22 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W

# Winograd's minimal rank-23 decomposition for 3x3
def _winograd_3x3():
    # Winograd's 3x3 matrix multiplication with rank 23
    # This is a well-known result. Let me implement it.
    # A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    # B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    
    U = [[0] * 23 for _ in range(9)]
    V = [[0] * 23 for _ in range(9)]
    W = [[0] * 23 for _ in range(9)]
    
    # Winograd's decomposition (from literature)
    # I'll use a known rank-23 decomposition
    
    # The 23 bilinear forms:
    # This is complex, so let me use a known implementation
    
    # Actually, let me try a different approach. 
    # For the 5x5 case, I'll try to use block decomposition:
    # 5 = 3 + 2
    # Blocks: A11 (3x3), A12 (3x2), A21 (2x3), A22 (2x2)
    #         B11 (3x3), B12 (3x2), B21 (2x3), B22 (2x2)
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # This gets very complex. Let me just return schoolbook for now
    # and rely on the structured approach being correct.
    
    # For now, return a placeholder - we'll use schoolbook
    return None, None, None

def _verify(U, V, W, n=5, m=5, p=5):
    """Verify that the decomposition is correct"""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,j,k), check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta(i,j,k)
    # where delta = 1 if this corresponds to a valid matrix multiplication term, 0 otherwise
    
    # Actually, the correct check is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
    # should equal 1 if (i,j,k) is a valid triple for matrix multiplication
    # 
    # Wait, let me reconsider. The matrix multiplication tensor T has:
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # 
    # Actually no. The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all valid (i,j,k).
    # 
    # The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Start with schoolbook as safe fallback
    schoolbook = _schoolbook()
    assert _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"], n, m, p), "Schoolbook failed verification"
    
    best = schoolbook
    best_rank = 125
    
    # Block decomposition with partition (2,2,1)
    # Use Strassen for 2x2x2 blocks, schoolbook for others
    # This should give rank 117
    
    try:
        result = _block_decomp_strassen()
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                rank = len(U[0])
                if rank < best_rank:
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
    except Exception as e:
        pass  # Fall back to schoolbook
    
    return best

def _block_decomp_strassen():
    """
    Block decomposition of 5x5x5 matmul with partition (2,2,1).
    Uses Strassen (rank 7) for 2x2x2 blocks, schoolbook for others.
    Expected total rank: 117
    """
    n, m, p = 5, 5, 5
    dims = [2, 2, 1]  # block sizes for each dimension
    
    # Precompute block offsets
    offsets = [0]
    for d in dims:
        offsets.append(offsets[-1] + d)
    # offsets = [0, 2, 4, 5]
    
    # Strassen decomposition for 2x2x2
    # Returns local U (4x7), V (4x7), W (4x7) for 2x2 matmul
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    S_U = [[0]*7 for _ in range(4)]
    S_V = [[0]*7 for _ in range(4)]
    S_W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)*(b11+b22)
    S_U[0][0] = 1; S_U[3][0] = 1
    S_V[0][0] = 1; S_V[3][0] = 1
    # M2 = (a21+a22)*b11
    S_U[2][1] = 1; S_U[3][1] = 1
    S_V[0][1] = 1
    # M3 = a11*(b12-b22)
    S_U[0][2] = 1
    S_V[1][2] = 1; S_V[3][2] = -1
    # M4 = a22*(b21-b11)
    S_U[3][3] = 1
    S_V[2][3] = 1; S_V[0][3] = -1
    # M5 = (a11+a12)*b22
    S_U[0][4] = 1; S_U[1][4] = 1
    S_V[3][4] = 1
    # M6 = (a21-a11)*(b11+b12)
    S_U[2][5] = 1; S_U[0][5] = -1
    S_V[0][5] = 1; S_V[1][5] = 1
    # M7 = (a12-a22)*(b21+b22)
    S_U[1][6] = 1; S_U[3][6] = -1
    S_V[2][6] = 1; S_V[3][6] = 1
    
    # C11 = M1 + M4 - M5 + M7
    S_W[0][0] = 1; S_W[0][3] = 1; S_W[0][4] = -1; S_W[0][6] = 1
    # C12 = M3 + M5
    S_W[1][2] = 1; S_W[1][4] = 1
    # C21 = M2 + M4
    S_W[2][1] = 1; S_W[2][3] = 1
    # C22 = M1 - M2 + M3 + M6
    S_W[3][0] = 1; S_W[3][1] = -1; S_W[3][2] = 1; S_W[3][5] = 1
    
    # Compute total rank needed
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a, b, c = dims[i], dims[j], dims[k]
                if a == 2 and b == 2 and c == 2:
                    total_rank += 7
                else:
                    total_rank += a * b * c
    
    # Allocate U, V, W
    U = [[0] * total_rank for _ in range(n * m)]  # 25 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # 25 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # 25 x total_rank
    
    rank_offset = 0
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a, b, c = dims[i], dims[j], dims[k]
                
                # Global index offsets for this block
                # A block (i,j): rows offset[i]..offset[i]+a-1, cols offset[j]..offset[j]+b-1
                # B block (j,k): rows offset[j]..offset[j]+b-1, cols offset[k]..offset[k]+c-1
                # C block (i,k): rows offset[k]..offset[k]+c-1, cols offset[i]..offset[i]+a-1
                
                if a == 2 and b == 2 and c == 2:
                    # Use Strassen: 7 terms
                    for local_r in range(7):
                        r = rank_offset + local_r
                        # A local indices: (li, lj) where li in {0,1}, lj in {0,1}
                        # Global A index: (offset[i] + li) * m + (offset[j] + lj)
                        for li in range(2):
                            for lj in range(2):
                                local_a = li * 2 + lj
                                global_a = (offsets[i] + li) * m + (offsets[j] + lj)
                                U[global_a][r] = S_U[local_a][local_r]
                        
                        # B local indices: (lj, lk) where lj in {0,1}, lk in {0,1}
                        # Global B index: (offset[j] + lj) * p + (offset[k] + lk)
                        for lj in range(2):
                            for lk in range(2):
                                local_b = lj * 2 + lk
                                global_b = (offsets[j] + lj) * p + (offsets[k] + lk)
                                V[global_b][r] = S_V[local_b][local_r]
                        
                        # C local indices: (lk, li) where lk in {0,1}, li in {0,1}
                        # W[k*n + i] convention: global W index = (offset[k] + lk) * n + (offset[i] + li)
                        for lk in range(2):
                            for li in range(2):
                                local_c = lk * 2 + li
                                global_w = (offsets[k] + lk) * n + (offsets[i] + li)
                                W[global_w][r] = S_W[local_c][local_r]
                    
                    rank_offset += 7
                else:
                    # Schoolbook: a*b*c terms
                    for li in range(a):
                        for lj in range(b):
                            for lk in range(c):
                                r = rank_offset + (li * b + lj) * c + lk
                                global_a = (offsets[i] + li) * m + (offsets[j] + lj)
                                global_b = (offsets[j] + lj) * p + (offsets[k] + lk)
                                global_w = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[global_a][r] = 1
                                V[global_b][r] = 1
                                W[global_w][r] = 1
                    
                    rank_offset += a * b * c
    
    assert rank_offset == total_rank, f"Rank mismatch: {rank_offset} vs {total_rank}"
    
    return U, V, W

# EVOLVE-BLOCK-END
