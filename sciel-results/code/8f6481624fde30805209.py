"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    n, m, p = 2, 3, 3
    
    # Build the multiplication tensor T[i,j,k] = delta for (A_{i,j} * B_{j,k}) -> C_{i,k}
    # T is a 3-tensor of shape (n*m, m*p, p*n) = (6, 9, 6)
    # T[a, b, c] = 1 if there exist i,j,k such that:
    #   a = i*m + j, b = j*p + k, c = k*n + i
    # For 2x3 * 3x3:
    #   i in {0,1}, j in {0,1,2}, k in {0,1,2}
    #   a = i*3 + j, b = j*3 + k, c = k*2 + i
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Method 1: Strassen on 2x2 sub-blocks + direct for cross terms
    # Split A (2x3) into [A00 A01 | A02; A10 A11 | A12] where Aij is 1x1
    # Split B (3x3) into [[B00 B01 B02]; [B10 B11 B12]; [B20 B21 B22]]
    # 
    # C = A * B (2x3)
    # C[0,0] = A00*B00 + A01*B10 + A02*B20
    # C[0,1] = A00*B01 + A01*B11 + A02*B21
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,0] = A10*B00 + A11*B10 + A12*B20
    # C[1,1] = A10*B01 + A11*B11 + A12*B21
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    #
    # Use Strassen on the 2x2 part (columns 0,1 of A, rows 0,1 of B):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B11-B00)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*(B10+B11)
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C[0,0] = M1 - M4 - M7 + M2
    # C[0,1] = M1 + M3 - M2 - M6
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 - M5 + M6
    #
    # Cross terms (column 2 of A, row 2 of B):
    # C[0,0] += A02*B20
    # C[0,1] += A02*B21
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,0] += A12*B20
    # C[1,1] += A12*B21
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    #
    # Direct scalar multiplications for cross terms:
    # M8 = A02*B20
    # M9 = A02*B21
    # M10 = A00*B02
    # M11 = A01*B12
    # M12 = A02*B22
    # M13 = A12*B20
    # M14 = A12*B21
    # M15 = A10*B02
    # M16 = A11*B12
    # M17 = A12*B22
    #
    # Total: 7 + 10 = 17 multiplications
    
    # Let's build this decomposition
    # U shape: (6, rank), V shape: (9, rank), W shape: (6, rank)
    
    # Index mapping:
    # A indices: a = i*3 + j, i in {0,1}, j in {0,1,2}
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
    # B indices: b = j*3 + k, j in {0,1,2}, k in {0,1,2}
    #   B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=3, B[1,1]=4, B[1,2]=5, B[2,0]=6, B[2,1]=7, B[2,2]=8
    # C indices: c = k*2 + i, k in {0,1,2}, i in {0,1}
    #   C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3, C[0,2]=4, C[1,2]=5
    
    rank = 17
    U = np.zeros((6, rank), dtype=np.int64)
    V = np.zeros((9, rank), dtype=np.int64)
    W = np.zeros((6, rank), dtype=np.int64)
    
    # Strassen multiplications (7)
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1  # A00, A11
    V[0, 0] = 1; V[4, 0] = 1  # B00, B11
    # C[0,0] += M1, C[0,1] += M1, C[1,0] += M1, C[1,1] += M1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[3, 1] = 1; U[4, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    # C[0,0] -= M2, C[0,1] -= M2, C[1,0] += M2
    W[0, 1] = -1; W[2, 1] = -1; W[1, 1] = 1
    
    # M3 = A00*(B11-B00)
    U[0, 2] = 1  # A00
    V[4, 2] = 1; V[0, 2] = -1  # B11, B00
    # C[0,1] += M3, C[1,1] -= M3
    W[2, 2] = 1; W[3, 2] = -1
    
    # M4 = A11*(B01-B10)
    U[4, 3] = 1  # A11
    V[1, 3] = 1; V[3, 3] = -1  # B01, B10
    # C[0,0] -= M4, C[1,0] += M4
    W[0, 3] = -1; W[1, 3] = 1
    
    # M5 = (A00+A01)*(B10+B11)
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1; V[4, 4] = 1  # B10, B11
    # C[1,1] -= M5
    W[3, 4] = -1
    
    # M6 = (A10-A00)*(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1  # A10, A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    # C[0,1] -= M6, C[1,1] += M6
    W[2, 5] = -1; W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1  # A01, A11
    V[3, 6] = 1; V[4, 6] = 1  # B10, B11
    # C[0,0] -= M7
    W[0, 6] = -1
    
    # Direct cross-term multiplications (10)
    # M8 = A02*B20
    U[2, 7] = 1  # A02
    V[6, 7] = 1  # B20
    W[0, 7] = 1  # C[0,0]
    
    # M9 = A02*B21
    U[2, 8] = 1  # A02
    V[7, 8] = 1  # B21
    W[2, 8] = 1  # C[0,1]
    
    # M10 = A00*B02
    U[0, 9] = 1  # A00
    V[2, 9] = 1  # B02
    W[4, 9] = 1  # C[0,2]
    
    # M11 = A01*B12
    U[1, 10] = 1  # A01
    V[5, 10] = 1  # B12
    W[4, 10] = 1  # C[0,2]
    
    # M12 = A02*B22
    U[2, 11] = 1  # A02
    V[8, 11] = 1  # B22
    W[4, 11] = 1  # C[0,2]
    
    # M13 = A12*B20
    U[5, 12] = 1  # A12
    V[6, 12] = 1  # B20
    W[1, 12] = 1  # C[1,0]
    
    # M14 = A12*B21
    U[5, 13] = 1  # A12
    V[7, 13] = 1  # B21
    W[3, 13] = 1  # C[1,1]
    
    # M15 = A10*B02
    U[3, 14] = 1  # A10
    V[2, 14] = 1  # B02
    W[5, 14] = 1  # C[1,2]
    
    # M16 = A11*B12
    U[4, 15] = 1  # A11
    V[5, 15] = 1  # B12
    W[5, 15] = 1  # C[1,2]
    
    # M17 = A12*B22
    U[5, 16] = 1  # A12
    V[8, 16] = 1  # B22
    W[5, 16] = 1  # C[1,2]
    
    # Verify the decomposition
    def verify(U, V, W, T):
        # Check: sum_c U[a,c]*V[b,c]*W[c_idx,c] == T[a,b,c_idx] for all a,b,c_idx
        # Actually, the tensor decomposition is: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    lhs = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if lhs != T[a, b, c]:
                        return False
        return True
    
    if not verify(U, V, W, T):
        # Fallback to schoolbook
        rank = 18
        U = np.zeros((6, rank), dtype=np.int64)
        V = np.zeros((9, rank), dtype=np.int64)
        W = np.zeros((6, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    else:
        # Try to reduce rank by checking for linear dependencies
        # Build the full tensor and try to find a lower rank decomposition
        if time.time() < deadline - 30:
            # Attempt rank reduction by trying to merge terms
            # Check if any column of U,V,W can be removed or combined
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Try removing each column and see if we can compensate
            for r_remove in range(rank):
                if time.time() > deadline - 10:
                    break
                # Try to see if column r_remove is linearly dependent
                U_reduced = np.delete(U, r_remove, axis=1)
                V_reduced = np.delete(V, r_remove, axis=1)
                W_reduced = np.delete(W, r_remove, axis=1)
                
                # Check if the reduced decomposition still works
                valid = True
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            lhs = sum(U_reduced[a, r] * V_reduced[b, r] * W_reduced[c, r] 
                                     for r in range(rank - 1))
                            if lhs != T[a, b, c]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_rank = rank - 1
                    best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
                    rank = best_rank
                    U, V, W = best_U, best_V, best_W
                    # Continue trying to remove more
                    r_remove = 0  # Reset to try again from beginning
                    continue
            
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook if verification failed
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
