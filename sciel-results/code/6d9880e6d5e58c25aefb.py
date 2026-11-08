import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Helper to get U, V, W for 2x2x2 multiplication
    def get_2x2x2_strassen():
        # n=2, m=2, p=2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices:
        # A: 00(0), 01(1), 10(2), 11(3)
        # B: 00(0), 01(1), 10(2), 11(3)
        # C: 00(0), 10(1), 01(2), 11(3)  (Trace: k*n+i. n=2.
        # C_00: k=0,i=0 -> 0
        # C_10: k=0,i=1 -> 1
        # C_01: k=1,i=0 -> 2
        # C_11: k=1,i=1 -> 3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen Terms
        # 1: (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # 2: (c+d)(e+h)  <-- Wait, standard Strassen M2 is (c+d)(e+h)?
        # Let's use the set:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e+h)
        # M3 = (a+b)(h-i) -- No, B is 2x2.
        
        # I will use this specific verified set:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)(B00 + B01)
        # M3 = (A00 - A11)(B10 + B11)
        # M4 = (A00 + A01)B11
        # M5 = (A10 - A00)B00
        # M6 = A11(B01 - B00)
        # M7 = A00(B10 - B11)
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1; V[1, 1] = 1
        W[1, 1] = 1 # C10
        
        # M3
        U[0, 2] = 1; U[3, 2] = -1
        V[2, 2] = 1; V[3, 2] = 1
        W[2, 2] = 1 # C01
        
        # M4
        U[0, 3] = 1; U[1, 3] = 1
        V[3, 3] = 1
        W[3, 3] = 1 # C11
        
        # M5
        U[2, 4] = 1; U[0, 4] = -1
        V[0, 4] = 1
        W[0, 4] = -1 # C00
        
        # M6
        U[3, 5] = 1
        V[1, 5] = 1; V[0, 5] = -1
        W[0, 5] = 1 # C00
        
        # M7
        U[0, 6] = 1
        V[2, 6] = 1; V[3, 6] = -1
        W[0, 6] = 1 # C00
        
        # Verify
        # Target Tensor T[i,j,k] = 1 if i,j,k in 0..1
        # Decomposed T'[i,j,k] = sum_r U[i*2+j, r] V[j*2+k, r] W[k*2+i, r]
        
        U_mat = U
        V_mat = V
        W_mat = W
        
        # Check a few points
        # C_00 (i=0, k=0) = A00 B00 + A01 B10
        # W index for C_00 is 0.
        # Sum_r U[0*2+0, r] V[0*2+0, r] W[0*2+0, r] + U[0*2+1, r] V[1*2+0, r] W[0*2+0, r]
        # Term 0 (A00, B00):
        # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
        # r=4: U[0,4]=-1, V[0,4]=1, W[0,4]=-1 -> 1
        # r=5: U[0,5]=0
        # r=6: U[0,6]=1, V[0,6]=0
        # Sum = 1 + 1 = 2? Should be 1.
        
        # This set is likely wrong.
        # I will return rank 8 direct to be safe, or just use the previous 44 rank if 42 fails.
        # But 44 was failing too.
        
        # Let's use a simple rank 4 for 2x2x2?
        # No, 2x2x2 min rank is 7.
        
        # I will use the previous program's logic but fixed?
        # The previous program had 4 blocks of 2x2x3.
        # I will stick to the 6 blocks of 2x2x2 plan.
        
        # If Strassen verification fails, use Rank 8 Direct.
        # Rank 8 Direct for 2x2x2:
        U_dir = np.zeros((4, 8))
        V_dir = np.zeros((4, 8))
        W_dir = np.zeros((4, 8))
        
        # r=0: A00 B00 -> C00
        U_dir[0, 0] = 1; V_dir[0, 0] = 1; W_dir[0, 0] = 1
        # r=1: A01 B10 -> C00
        U_dir[1, 1] = 1; V_dir[2, 1] = 1; W_dir[0, 1] = 1
        # r=2: A00 B01 -> C01
        U_dir[0, 2] = 1; V_dir[1, 2] = 1; W_dir[2, 2] = 1
        # r=3: A01 B11 -> C01
        U_dir[1, 3] = 1; V_dir[3, 3] = 1; W_dir[2, 3] = 1
        # r=4: A10 B00 -> C10
        U_dir[2, 4] = 1; V_dir[0, 4] = 1; W_dir[1, 4] = 1
        # r=5: A11 B10 -> C10
        U_dir[3, 5] = 1; V_dir[2, 5] = 1; W_dir[1, 5] = 1
        # r=6: A10 B01 -> C11
        U_dir[2, 6] = 1; V_dir[1, 6] = 1; W_dir[3, 6] = 1
        # r=7: A11 B11 -> C11
        U_dir[3, 7] = 1; V_dir[3, 7] = 1; W_dir[3, 7] = 1
        
        # Verify Direct
        # C00: r=0 (1*1*1) + r=1 (1*1*1) = 2? No.
        # C00 = A00 B00 + A01 B10.
        # r=0: U[0,0]=1 (A00), V[0,0]=1 (B00), W[0,0]=1 (C00). Product 1.
        # r=1: U[1,1]=1 (A01), V[2,1]=1 (B10), W[0,1]=1 (C00). Product 1.
        # Sum = 2. Should be 1.
        # Ah, the tensor identity is sum_r U V W = T.
        # T[0,0,0] = 1.
        # My direct decomposition gives 2.
        # Why?
        # U[0,0]=1 corresponds to A[0,0].
        # V[0,0]=1 corresponds to B[0,0].
        # W[0,0]=1 corresponds to C[0,0].
        # So term 0 is A00 B00 C00.
        # Term 1 is A01 B10 C00.
        # Sum is A00 B00 + A01 B10.
        # This IS C00.
        # So why did I think it was 2?
        # Because I summed the coefficients.
        # The value is 1*1*1 + 1*1*1 = 2?
        # No, the formula is $\sum_r U_{i',r} V_{j',r} W_{k',r}$.
        # For T[0,0,0]:
        # r=0: U[0,0] V[0,0] W[0,0] = 1*1*1 = 1.
        # r=1: U[1,1] V[2,1] W[0,1] = 0*1*1 = 0 (U[0,1] is 0).
        # Wait, U[0,1] is 0.
        # So sum is 1. Correct.
        
        # So Direct Rank 8 works.
        # I will use Direct Rank 8 for 2x2x2 to ensure correctness.
        # Total Rank = 6 * 8 = 48.
        # Previous was 44 (and failed).
        # 48 > 44.
        # But 44 failed.
        # I need a working 44 or lower.
        
        # Let's try to fix the 2x2x3 rank 11?
        # Or just use 48.
        # The goal is to fix the program.
        # 48 is a valid score.
        
        return U_dir, V_dir, W_dir

    # Assemble 2x4 * 4x6
    # 6 blocks of 2x2x2
    # A is 2x4. Split cols into 2 blocks of 2.
    # B is 4x6. Split rows into 2 blocks of 2, cols into 3 blocks of 2.
    # C is 2x6. Split cols into 3 blocks of 2.
    
    U_total = np.zeros((8, 48))
    V_total = np.zeros((24, 48))
    W_total = np.zeros((12, 48))
    
    u0, v0, w0 = get_2x2x2_strassen()
    
    # Block (i, j) where i in 0..1 (A col block), j in 0..2 (B col block)
    # A block i: cols i*2, i*2+1. Rows 0,1.
    # B block i,j: rows i*2, i*2+1. Cols j*2, j*2+1.
    # C block j: cols j*2, j*2+1. Rows 0,1.
    
    for i in range(2):
        for j in range(3):
            block_idx = i * 3 + j
            start_r = block_idx * 8
            
            # U: A rows 0,1. Cols i*2, i*2+1.
            # Global A indices: 0,1,2,3,4,5,6,7.
            # Row 0: 0,1,2,3. Row 1: 4,5,6,7.
            # A[i*2, 0] is global idx i*2.
            # A[i*2+1, 0] is global idx i*2+1.
            # A[i*2, 1] is global idx i*2+2.
            # A[i*2+1, 1] is global idx i*2+3.
            
            # Local U indices: 0(A00), 1(A01), 2(A10), 3(A11).
            # Map local to global:
            # 0 -> i*2
            # 1 -> i*2+1
            # 2 -> i*2+4
            # 3 -> i*2+5
            
            U_total[i*2, start_r:start_r+8] = u0[0]
            U_total[i*2+1, start_r:start_r+8] = u0[1]
            U_total[i*2+4, start_r:start_r+8] = u0[2]
            U_total[i*2+5, start_r:start_r+8] = u0[3]
            
            # V: B rows i*2, i*2+1. Cols j*2, j*2+1.
            # Global B indices: Row 0: 0..5. Row 1: 6..11. Row 2: 12..17. Row 3: 18..23.
            # B[i*2, j*2] is global idx i*2*6 + j*2.
            # B[i*2, j*2+1] is global idx i*2*6 + j*2+1.
            # B[i*2+1, j*2] is global idx (i*2+1)*6 + j*2.
            # B[i*2+1, j*2+1] is global idx (i*2+1)*6 + j*2+1.
            
            # Local V indices: 0(B00), 1(B01), 2(B10), 3(B11).
            # Map local to global:
            # 0 -> i*2*6 + j*2
            # 1 -> i*2*6 + j*2+1
            # 2 -> (i*2+1)*6 + j*2
            # 3 -> (i*2+1)*6 + j*2+1
            
            V_total[i*2*6 + j*2, start_r:start_r+8] = v0[0]
            V_total[i*2*6 + j*2+1, start_r:start_r+8] = v0[1]
            V_total[(i*2+1)*6 + j*2, start_r:start_r+8] = v0[2]
            V_total[(i*2+1)*6 + j*2+1, start_r:start_r+8] = v0[3]
            
            # W: C rows 0,1. Cols j*2, j*2+1.
            # Global C indices: Row 0: 0..5. Row 1: 6..11.
            # Trace: k*n+i. n=2.
            # C[0, j*2] -> k=j*2, i=0 -> j*2*2 + 0 = 4*j.
            # C[0, j*2+1] -> k=j*2+1, i=0 -> (j*2+1)*2 + 0 = 4*j+2.
            # C[1, j*2] -> k=j*2, i=1 -> j*2*2 + 1 = 4*j+1.
            # C[1, j*2+1] -> k=j*2+1, i=1 -> (j*2+1)*2 + 1 = 4*j+3.
            
            # Local W indices: 0(C00), 1(C10), 2(C01), 3(C11).
            # Map local to global:
            # 0 -> 4*j
            # 1 -> 4*j+1
            # 2 -> 4*j+2
            # 3 -> 4*j+3
            
            W_total[4*j, start_r:start_r+8] = w0[0]
            W_total[4*j+1, start_r:start_r+8] = w0[1]
            W_total[4*j+2, start_r:start_r+8] = w0[2]
            W_total[4*j+3, start_r:start_r+8] = w0[3]

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
