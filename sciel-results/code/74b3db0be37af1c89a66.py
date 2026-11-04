"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Strategy: Split middle dimension m=3 into 2+1
    # A is 2x3, B is 3x10
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B = [B1; B2] where B1 is 2x10, B2 is 1x10
    # C = A*B = A1*B1 + A2*B2
    # 
    # For A1*B1: 2x2 * 2x10 -> can use Strassen (rank 7) for each 2x2 block of B1
    # Actually, we need to think about this differently.
    #
    # Better approach: Use the fact that we can decompose the tensor.
    # The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Split j into j=0,1 (first two) and j=2 (last one)
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # The first two terms form a 2x2 * 2x10 multiplication (A[0:2,0:2] * B[0:2,:])
    # The last term is 2x1 * 1x10 (A[0:2,2] * B[2,:])
    #
    # For 2x2 * 2x10: We can view this as 5 independent 2x2 * 2x2 multiplications
    # (since 10 = 5*2), each with rank 7 via Strassen. Total: 5*7 = 35
    # But wait, that's not quite right either. Let me think more carefully.
    #
    # Actually, the tensor for 2x2 * 2x2 has rank 7 (Strassen).
    # For 2x2 * 2x10, we can think of it as: for each pair of output columns (k, k+1),
    # we do a 2x2 * 2x2 multiplication. There are 5 such pairs (if we group columns).
    # But this doesn't work because the input B columns are shared.
    #
    # Let me reconsider. The standard approach:
    # M_{2,2,2} has rank 7 (Strassen)
    # M_{2,2,10} can be done by treating it as 5 copies of M_{2,2,2} if we can
    # decompose B into 5 blocks of 2 columns each. But B is 2x10, so we have
    # B = [B_0 | B_1 | B_2 | B_3 | B_4] where each B_i is 2x2.
    # Then C = A * B = A*B_0 + A*B_1 + ... + A*B_4
    # Each A*B_i is a 2x2 * 2x2 = 2x2 multiplication, rank 7.
    # Total rank: 5 * 7 = 35
    #
    # For the 2x1 * 1x10 part: rank = 2*1*10 = 20
    # Total: 35 + 20 = 55
    #
    # This matches the inspiration! Let's implement this.
    
    # Build Strassen decomposition for 2x2 * 2x2
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    # For our problem: A is 2x2 (call it A_sub), B is 2x10 (call it B_sub)
    # We split B_sub into 5 blocks of 2x2: B_0, B_1, B_2, B_3, B_4
    # C_sub = A_sub * B_sub = A_sub*B_0 + A_sub*B_1 + ... + A_sub*B_4
    #
    # For each block B_i, we use Strassen with 7 products.
    # The U factors come from A_sub, V factors from B_i, W factors from C_sub.
    
    n, m, p = 2, 3, 10
    U_rows = n * m  # 6
    V_rows = m * p  # 30
    W_rows = p * n  # 20
    
    # Rank breakdown:
    # Strassen part: 5 blocks * 7 = 35
    # Schoolbook part (A[:,2] * B[2,:]): 2 * 10 = 20
    # Total: 55
    
    total_rank = 55
    U = [[0.0] * total_rank for _ in range(U_rows)]
    V = [[0.0] * total_rank for _ in range(V_rows)]
    W = [[0.0] * total_rank for _ in range(W_rows)]
    
    # Strassen coefficients for 2x2 * 2x2
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    strassen_U = {
        1: [(0, 1), (1, 1)],  # M1: a00 + a11
        2: [(1, 1), (1, 1)],  # M2: a10 + a11
        3: [(0, 1)],           # M3: a00
        4: [(1, 1)],           # M4: a11
        5: [(0, 1), (0, 1)],  # M5: a00 + a01
        6: [(1, 1), (0, -1)], # M6: a10 - a00
        7: [(0, 1), (1, -1)], # M7: a01 - a11
    }
    
    strassen_V = {
        1: [(0, 1), (1, 1)],  # M1: b00 + b11
        2: [(0, 1)],           # M2: b00
        3: [(1, 1), (1, -1)], # M3: b01 - b11
        4: [(1, 1), (0, -1)], # M4: b10 - b00
        5: [(1, 1)],           # M5: b11
        6: [(0, 1), (0, 1)],  # M6: b00 + b01
        7: [(1, 1), (1, 1)],  # M7: b10 + b11
    }
    
    strassen_W = {
        1: [(0, 1), (1, -1)],  # M1: C00 + C11
        2: [(1, 1)],           # M2: C10
        3: [(1, 1), (0, -1)], # M3: C11 - C01
        4: [(0, -1), (1, 1)], # M4: -C00 + C10
        5: [(0, 1), (1, 1)],  # M5: C00 + C01
        6: [(1, 1), (0, -1)], # M6: C11 - C10
        7: [(0, 1), (1, 1)],  # M7: C00 + C11
    }
    
    # Let me verify Strassen more carefully.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # Let me redefine with clear indices:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # c00 = M1 - M4 + M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M6 + M7
    
    # U factors (from A):
    # M1: a00 + a11 -> U[0,0]=1, U[1,0]=1 (using row indices i*m+j)
    #    For A[i,j], i in {0,1}, j in {0,1}
    #    a00 is A[0,0], a11 is A[1,1]
    #    U[0*m+0] = U[0] += 1, U[1*m+1] = U[4] += 1
    # M2: a10 + a11 -> U[1*m+0]=U[3] += 1, U[1*m+1]=U[4] += 1
    # M3: a00 -> U[0] += 1
    # M4: a11 -> U[4] += 1
    # M5: a00 + a01 -> U[0] += 1, U[1] += 1
    # M6: a10 - a00 -> U[3] += 1, U[0] -= 1
    # M7: a01 - a11 -> U[1] += 1, U[4] -= 1
    
    # V factors (from B): B is 2x10, split into 5 blocks of 2x2
    # For block t (t=0..4), columns are 2*t and 2*t+1
    # B_sub[j, k] where j in {0,1}, k in {0,1} maps to B[j, 2*t+k]
    # M1: b00 + b11 -> V[0*p + 2*t] += 1, V[1*p + 2*t+1] += 1
    # M2: b00 -> V[0*p + 2*t] += 1
    # M3: b01 - b11 -> V[0*p + 2*t+1] += 1, V[1*p + 2*t+1] -= 1
    # M4: b10 - b00 -> V[1*p + 2*t] += 1, V[0*p + 2*t] -= 1
    # M5: b11 -> V[1*p + 2*t+1] += 1
    # M6: b00 + b01 -> V[0*p + 2*t] += 1, V[0*p + 2*t+1] += 1
    # M7: b10 + b11 -> V[1*p + 2*t] += 1, V[1*p + 2*t+1] += 1
    
    # W factors (from C): C is 2x10
    # c00 is C[0, 2*t], c01 is C[0, 2*t+1], c10 is C[1, 2*t], c11 is C[1, 2*t+1]
    # W index: k*n + i where k is column, i is row
    # c00: W[2*t*n + 0] = W[2*t] 
    # c01: W[(2*t+1)*n + 0] = W[4*t+2]
    # c10: W[2*t*n + 1] = W[2*t+1]
    # c11: W[(2*t+1)*n + 1] = W[4*t+3]
    #
    # M1: c00 + c11 -> W[2*t] += 1, W[4*t+3] += 1
    # M2: c10 -> W[2*t+1] += 1
    # M3: c01 -> W[4*t+2] += 1
    # M4: -c00 + c10 -> W[2*t] -= 1, W[2*t+1] += 1
    # M5: c00 + c01 -> W[2*t] += 1, W[4*t+2] += 1
    # M6: c11 - c10 -> W[4*t+3] += 1, W[2*t+1] -= 1
    # M7: c00 + c11 -> W[2*t] += 1, W[4*t+3] += 1
    
    # Wait, I need to double-check the W indices.
    # W has shape (p*n, rank) = (20, rank)
    # Index is k*n + i where k is column index (0..9), i is row index (0..1)
    # So for column 2*t, row 0: index = (2*t)*2 + 0 = 4*t
    # For column 2*t, row 1: index = (2*t)*2 + 1 = 4*t+1
    # For column 2*t+1, row 0: index = (2*t+1)*2 + 0 = 4*t+2
    # For column 2*t+1, row 1: index = (2*t+1)*2 + 1 = 4*t+3
    
    # Let me redo:
    # c00 = C[0, 2*t] -> W[4*t]
    # c01 = C[0, 2*t+1] -> W[4*t+2]
    # c10 = C[1, 2*t] -> W[4*t+1]
    # c11 = C[1, 2*t+1] -> W[4*t+3]
    
    # M1: c00 + c11 -> W[4*t] += 1, W[4*t+3] += 1
    # M2: c10 -> W[4*t+1] += 1
    # M3: c01 -> W[4*t+2] += 1
    # M4: -c00 + c10 -> W[4*t] -= 1, W[4*t+1] += 1
    # M5: c00 + c01 -> W[4*t] += 1, W[4*t+2] += 1
    # M6: c11 - c10 -> W[4*t+3] += 1, W[4*t+1] -= 1
    # M7: c00 + c11 -> W[4*t] += 1, W[4*t+3] += 1
    
    # Now let's implement:
    # Strassen part uses ranks 0..34 (5 blocks * 7 = 35)
    # Schoolbook part uses ranks 35..54 (2*10 = 20)
    
    for t in range(5):  # 5 blocks of 2 columns each
        for s in range(1, 8):  # 7 Strassen products
            r = t * 7 + (s - 1)  # rank index
            
            # U factors (A is 2x2, indices i in {0,1}, j in {0,1})
            # U index: i*m + j where m=3, but we're only using j in {0,1} for the 2x2 part
            # Wait, A is 2x3. For the 2x2 sub-block, we use A[0:2, 0:2]
            # U index for A[i,j] is i*m + j = i*3 + j
            
            if s == 1:  # M1: a00 + a11
                U[0*3 + 0][r] += 1  # a00
                U[1*3 + 1][r] += 1  # a11
            elif s == 2:  # M2: a10 + a11
                U[1*3 + 0][r] += 1  # a10
                U[1*3 + 1][r] += 1  # a11
            elif s == 3:  # M3: a00
                U[0*3 + 0][r] += 1  # a00
            elif s == 4:  # M4: a11
                U[1*3 + 1][r] += 1  # a11
            elif s == 5:  # M5: a00 + a01
                U[0*3 + 0][r] += 1  # a00
                U[0*3 + 1][r] += 1  # a01
            elif s == 6:  # M6: a10 - a00
                U[1*3 + 0][r] += 1  # a10
                U[0*3 + 0][r] -= 1  # -a00
            elif s == 7:  # M7: a01 - a11
                U[0*3 + 1][r] += 1  # a01
                U[1*3 + 1][r] -= 1  # -a11
            
            # V factors (B is 2x10, for block t, columns 2*t and 2*t+1)
            # V index: j*p + k where j in {0,1}, k in {2*t, 2*t+1}
            if s == 1:  # M1: b00 + b11
                V[0*10 + 2*t][r] += 1  # b00
                V[1*10 + 2*t+1][r] += 1  # b11
            elif s == 2:  # M2: b00
                V[0*10 + 2*t][r] += 1  # b00
            elif s == 3:  # M3: b01 - b11
                V[0*10 + 2*t+1][r] += 1  # b01
                V[1*10 + 2*t+1][r] -= 1  # -b11
            elif s == 4:  # M4: b10 - b00
                V[1*10 + 2*t][r] += 1  # b10
                V[0*10 + 2*t][r] -= 1  # -b00
            elif s == 5:  # M5: b11
                V[1*10 + 2*t+1][r] += 1  # b11
            elif s == 6:  # M6: b00 + b01
                V[0*10 + 2*t][r] += 1  # b00
                V[0*10 + 2*t+1][r] += 1  # b01
            elif s == 7:  # M7: b10 + b11
                V[1*10 + 2*t][r] += 1  # b10
                V[1*10 + 2*t+1][r] += 1  # b11
            
            # W factors (C is 2x10)
            # W index: k*n + i where k in {2*t, 2*t+1}, i in {0,1}
            if s == 1:  # M1: c00 + c11
                W[(2*t)*2 + 0][r] += 1  # c00
                W[(2*t+1)*2 + 1][r] += 1  # c11
            elif s == 2:  # M2: c10
                W[(2*t)*2 + 1][r] += 1  # c10
            elif s == 3:  # M3: c01
                W[(2*t+1)*2 + 0][r] += 1  # c01
            elif s == 4:  # M4: -c00 + c10
                W[(2*t)*2 + 0][r] -= 1  # -c00
                W[(2*t)*2 + 1][r] += 1  # c10
            elif s == 5:  # M5: c00 + c01
                W[(2*t)*2 + 0][r] += 1  # c00
                W[(2*t+1)*2 + 0][r] += 1  # c01
            elif s == 6:  # M6: c11 - c10
                W[(2*t+1)*2 + 1][r] += 1  # c11
                W[(2*t)*2 + 1][r] -= 1  # -c10
            elif s == 7:  # M7: c00 + c11
                W[(2*t)*2 + 0][r] += 1  # c00
                W[(2*t+1)*2 + 1][r] += 1  # c11
    
    # Schoolbook part for A[:,2] * B[2,:]
    # This is 2x1 * 1x10, rank = 2*10 = 20
    # A[i,2] * B[2,k] for i in {0,1}, k in {0..9}
    # U index for A[i,2]: i*3 + 2
    # V index for B[2,k]: 2*10 + k = 20 + k
    # W index for C[i,k]: k*2 + i
    
    for i in range(2):
        for k in range(10):
            r = 35 + i * 10 + k  # rank index 35..54
            
            U[i*3 + 2][r] = 1  # A[i,2]
            V[2*10 + k][r] = 1  # B[2,k]
            W[k*2 + i][r] = 1  # C[i,k]
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(total_rank):
                    val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(val - 1.0) > 1e-6:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        rank_schoolbook = n * m * p
        U_school = [[0] * rank_schoolbook for _ in range(n * m)]
        V_school = [[0] * rank_schoolbook for _ in range(m * p)]
        W_school = [[0] * rank_schoolbook for _ in range(p * n)]
        
        for r in range(rank_schoolbook):
            i = r // (m * p)
            rem = r % (m * p)
            j = rem // p
            k = rem % p
            
            idx_u = i * m + j
            idx_v = j * p + k
            idx_w = k * n + i
            
            U_school[idx_u][r] = 1
            V_school[idx_v][r] = 1
            W_school[idx_w][r] = 1
        
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
