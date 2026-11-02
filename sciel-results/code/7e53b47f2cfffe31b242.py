"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s evaluator kill
    n, m, p = DIMENSIONS

    # Default: schoolbook rank = n*m*p = 36
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    try:
        # Strategy: Winograd Convolution
        # We model the multiplication C = A * B where A is 2x3 and B is 3x6.
        # C[i, k] = sum_{j=0}^{2} A[i, j] * B[j, k].
        # For fixed i and k, this is a dot product of length 3.
        # However, we can view this as a convolution if we consider the structure.
        # Actually, for fixed i, the row A[i, :] is length 3.
        # The column B[:, k] is length 3.
        # This is just a dot product, not a convolution in the standard sense unless we embed it.
        
        # Let's try a different framing:
        # The task is 2x3 * 3x6.
        # We can split the 3x6 matrix B into two 3x3 blocks: B_left (cols 0-2) and B_right (cols 3-5).
        # C_left = A * B_left (2x3 * 3x3 -> 2x3)
        # C_right = A * B_right (2x3 * 3x3 -> 2x3)
        
        # The rank of 2x3 * 3x3 matrix multiplication is known to be 11 (using Strassen for 2x2x2 part + schoolbook for rest, or similar).
        # The current program already achieves rank 22 by doing this (11+11).
        
        # Can we do better?
        # The theoretical lower bound is max(n*m, m*p, p*n) = max(6, 18, 12) = 18.
        # Upper bound is 36.
        # Known results for 2x3x3: Rank is 11.
        # So 2x3x6 rank is at most 22.
        
        # Is there a decomposition of rank < 22?
        # Maybe we can share factors between the left and right blocks?
        # The A matrix is the same.
        # If we use Winograd for the 3x3 blocks, the rank is 7 for 3x3x3? No, 3x3x3 matrix mult rank is 19 (or 23?). 
        # Wait, 2x3x3 is different from 3x3x3.
        
        # Let's re-evaluate the "Winograd" hint.
        # "modeling the 2x3x6 multiplication as a linear convolution of length 3 and 6"
        # This suggests treating the j-dimension as a convolution kernel.
        
        # Let's try to construct a rank 18 decomposition? Or maybe 21?
        # The prompt mentions "potentially yielding a rank of 21".
        
        # Let's attempt to find a rank 21 decomposition.
        # If we can't find a known one, we might have to search or use a heuristic.
        # Given the time limit, a heuristic search for rank 21 is risky.
        
        # Let's stick to the known rank 22 (two 2x3x3 blocks of rank 11) unless we can prove rank 21.
        # However, the current program already does this and scores 1.0 (which is 36/36? No, score is seed/value. Seed is likely 36. Value is rank. So 36/22 = 1.63? But the current score is 1.0. 
        # Wait, the current program returns schoolbook (rank 36) if the sub-decomposition fails?
        # Let's check the code. 
        # "if not verify_sub... return best" (best is schoolbook).
        # The Strassen terms for 2x2x2 are hardcoded. 
        # Maybe the verification fails?
        
        # Let's debug the verification of the 2x3x3 rank 11 decomposition.
        # The Strassen terms provided in the current code:
        # They look like the standard 7-term Strassen for 2x2x2.
        # Let's verify them manually for a simple case.
        
        # If the current program returns rank 36 (score 1.0), it means the rank 22 attempt failed verification.
        # Why?
        # U_sub, V_sub, W_sub construction.
        # U_sub[i * m_sub + j, t] = ut[i * 2 + j]
        # m_sub = 3.
        # U_sub indices: i in 0..1, j in 0..2.
        # For Strassen terms (j in 0..1), it fills U_sub[i*3+j, t].
        # For schoolbook terms (j=2), it fills U_sub[i*3+2, t].
        
        # Let's check the Strassen terms again.
        # Term 0: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
        # This corresponds to (A00+A11)(B00+B11).
        # U00=1, U01=0, U10=0, U11=1.
        # V00=1, V01=0, V10=0, V11=1.
        # W00=1, W01=0, W10=0, W11=1.
        # This looks correct for the diagonal terms.
        
        # Term 1: U=[0,0,1,1], V=[1,0,0,0], W=[0,1,0,-1]
        # U10=1, U11=1. (A10+A11)
        # V00=1. (B00)
        # W01=1, W11=-1. (C10 += ..., C11 -= ...)
        # This looks like P2 = (A10+A11)B00.
        # C10 += P2, C11 -= P2? 
        # Standard Strassen:
        # P1 = (A11 - A00)(B00 + B11)
        # P2 = (A00 + A10)B00
        # P3 = A11(B11 - B01)
        # P4 = A00(B10 + B11)
        # P5 = (A01 + A11)B11
        # P6 = (A10 - A00)(B01 + B11)
        # P7 = (A01 - A10)(B00 + B10)
        
        # The provided terms in the code:
        # t=0: [1,0,0,1], [1,0,0,1], [1,0,0,1] -> (A00+A11)(B00+B11) ... Wait, this is not standard P1.
        # Standard P1 is (A11-A00)(B00+B11).
        # Let's check the sum of terms for C00.
        # C00 = P1 + P4 - P5 - P7.
        # C01 = P3 + P5.
        # C10 = P2 + P4.
        # C11 = P1 - P2 + P3 + P6.
        
        # Let's check the provided terms against this.
        # t=0: U=[1,0,0,1] (A00+A11), V=[1,0,0,1] (B00+B11), W=[1,0,0,1] (C00+C11? No, W indices are k*2+i. W[0]=C00, W[1]=C01? No, k is row of B/col of C? 
        # W[k*n+i]. n=2.
        # k=0: W[0]=C00, W[1]=C01? No, i is row of A/col of C?
        # C[i,k]. i in 0..1, k in 0..1.
        # W index = k*n + i.
        # k=0, i=0 -> W[0] = C00.
        # k=0, i=1 -> W[1] = C10? No, C[i,k]. i=1, k=0 -> C10.
        # k=1, i=0 -> W[2] = C01.
        # k=1, i=1 -> W[3] = C11.
        
        # So W vector [w0, w1, w2, w3] corresponds to [C00, C10, C01, C11].
        
        # Let's re-read the Strassen terms in the code.
        # t=0: W=[1, 0, 0, 1] -> C00 += 1, C11 += 1.
        # t=1: W=[0, 1, 0, -1] -> C10 += 1, C11 -= 1.
        # t=2: W=[0, 0, 1, 1] -> C01 += 1, C11 += 1.
        # t=3: W=[1, 1, 0, 0] -> C00 += 1, C10 += 1.
        # t=4: W=[-1, 0, 1, 0] -> C00 -= 1, C01 += 1.
        # t=5: W=[0, 0, 0, 1] -> C11 += 1.
        # t=6: W=[-1, 1, 0, 0] -> C00 -= 1, C10 += 1.
        
        # Let's check C00 sum:
        # t0: 1
        # t3: 1
        # t4: -1
        # t6: -1
        # Sum = 0. 
        # This is wrong. C00 should be A00B00 + A01B10.
        
        # It seems the Strassen terms in the current program are incorrect or I am misinterpreting the indices.
        # This explains why the verification fails and it falls back to schoolbook (rank 36).
        
        # My goal: Provide a CORRECT rank 22 decomposition (or better).
        
        # Let's construct a correct 2x3x3 rank 11 decomposition.
        # We can use the known rank 11 algorithm for 2x3x3.
        # Or, we can just use the correct Strassen terms for the 2x2x2 part and schoolbook for the rest.
        
        # Correct Strassen terms for 2x2x2 (indices: U[i*2+j], V[j*2+k], W[k*2+i]):
        # Note: W index k*2+i.
        # P1 = (A11-A00)(B00+B11)
        # U: A11=1, A00=-1 -> [ -1, 0, 0, 1 ] (i=0,j=0 is -1; i=1,j=1 is 1)
        # V: B00=1, B11=1 -> [ 1, 0, 0, 1 ]
        # W: C10 += 1, C11 -= 1, C01 += 1, C00 -= 1? 
        # Standard:
        # C00 = P4 - P5 - P7 + P1
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Let's map P1:
        # U1: [-1, 0, 0, 1] (A00=-1, A11=1)
        # V1: [1, 0, 0, 1] (B00=1, B11=1)
        # W1: C00 += 1, C10 += 1, C01 += 1, C11 -= 1?
        # Wait, P1 contributes to C10, C11, C01, C00?
        # P1 = (A11-A00)(B00+B11)
        # C00 += P1? No.
        # C00 = A00B00 + A01B10.
        # C10 = A10B00 + A11B10.
        # C01 = A00B01 + A01B11.
        # C11 = A10B01 + A11B11.
        
        # P1 = A11B00 + A11B11 - A00B00 - A00B11.
        # This term appears in:
        # C11: +A11B11.
        # C10: +A11B10? No, B10 is not in P1.
        
        # Let's use a verified set of Strassen terms.
        # From standard references:
        # P1 = (A11 - A00)(B00 + B11)
        # P2 = (A00 + A10)B00
        # P3 = A11(B11 - B01)
        # P4 = A00(B10 + B11)
        # P5 = (A01 + A11)B11
        # P6 = (A10 - A00)(B01 + B11)
        # P7 = (A01 - A10)(B00 + B10)
        
        # C00 = P4 - P5 - P7 + P1
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Let's build the vectors for these 7 terms for 2x2x2.
        # U index: i*2+j. V index: j*2+k. W index: k*2+i.
        
        # Term 1 (P1):
        # U: A11=1, A00=-1. Others 0. -> [ -1, 0, 0, 1 ]
        # V: B00=1, B11=1. Others 0. -> [ 1, 0, 0, 1 ]
        # W: C00+=1, C10+=1, C01+=1, C11-=1.
        # W indices: k*2+i.
        # C00 (k=0,i=0) -> idx 0: +1
        # C10 (k=0,i=1) -> idx 1: +1
        # C01 (k=1,i=0) -> idx 2: +1
        # C11 (k=1,i=1) -> idx 3: -1
        # W1: [1, 1, 1, -1]
        
        # Term 2 (P2):
        # U: A00=1, A10=1. -> [ 1, 0, 1, 0 ]
        # V: B00=1. -> [ 1, 0, 0, 0 ]
        # W: C10+=1, C11-=1.
        # C10 (k=0,i=1) -> idx 1: +1
        # C11 (k=1,i=1) -> idx 3: -1
        # W2: [0, 1, 0, -1]
        
        # Term 3 (P3):
        # U: A11=1. -> [ 0, 0, 0, 1 ]
        # V: B11=1, B01=-1. -> [ 0, -1, 0, 1 ]
        # W: C01+=1, C11+=1.
        # C01 (k=1,i=0) -> idx 2: +1
        # C11 (k=1,i=1) -> idx 3: +1
        # W3: [0, 0, 1, 1]
        
        # Term 4 (P4):
        # U: A00=1. -> [ 1, 0, 0, 0 ]
        # V: B10=1, B11=1. -> [ 0, 0, 1, 1 ]
        # W: C00+=1, C10+=1.
        # C00 (k=0,i=0) -> idx 0: +1
        # C10 (k=0,i=1) -> idx 1: +1
        # W4: [1, 1, 0, 0]
        
        # Term 5 (P5):
        # U: A01=1, A11=1. -> [ 0, 1, 0, 1 ]
        # V: B11=1. -> [ 0, 0, 0, 1 ]
        # W: C00-=1, C01+=1.
        # C00 (k=0,i=0) -> idx 0: -1
        # C01 (k=1,i=0) -> idx 2: +1
        # W5: [-1, 0, 1, 0]
        
        # Term 6 (P6):
        # U: A10=1, A00=-1. -> [ -1, 0, 1, 0 ]
        # V: B01=1, B11=1. -> [ 0, 1, 0, 1 ]
        # W: C11+=1.
        # C11 (k=1,i=1) -> idx 3: +1
        # W6: [0, 0, 0, 1]
        
        # Term 7 (P7):
        # U: A01=1, A10=-1. -> [ 0, 1, -1, 0 ]
        # V: B00=1, B10=1. -> [ 1, 0, 1, 0 ]
        # W: C00-=1.
        # C00 (k=0,i=0) -> idx 0: -1
        # W7: [-1, 0, 0, 0]
        
        # Let's verify C00: P4 - P5 - P7 + P1
        # W4[0]=1, W5[0]=-1, W7[0]=-1, W1[0]=1.
        # 1 - (-1) - (-1) + 1 = 4? 
        # Wait. C00 = P4 - P5 - P7 + P1.
        # Coefficients:
        # P4: +1
        # P5: -1 (so -1 * W5[0] = -1 * -1 = +1)
        # P7: -1 (so -1 * W7[0] = -1 * -1 = +1)
        # P1: +1 (so +1 * W1[0] = +1)
        # Sum = 1 + 1 + 1 + 1 = 4. 
        # This is wrong. The coefficients in the formula are for the terms P, not the W vectors directly.
        # The W vector contains the contribution of that term to the output.
        # So if C00 = P4 - P5 - P7 + P1, then the total W vector for C00 is W4 - W5 - W7 + W1.
        # W4[0]=1.
        # W5[0]=-1. So -W5[0] = 1.
        # W7[0]=-1. So -W7[0] = 1.
        # W1[0]=1.
        # Sum = 1+1+1+1 = 4.
        
        # Why? Because the Strassen algorithm is a decomposition of the TENSOR.
        # The tensor T[i,j,k] = 1 if i,j,k correspond to a valid multiplication.
        # The decomposition is T = sum_t U_t V_t W_t.
        # So for each t, we have a rank-1 tensor U_t * V_t * W_t.
        # The sum of these rank-1 tensors must equal the multiplication tensor.
        
        # So, I just need to sum the rank-1 tensors.
        # My W vectors above are the W_t vectors.
        # I need to check if sum_t (U_t[i] * V_t[j] * W_t[k]) equals 1 for valid (i,j,k) and 0 otherwise.
        
        # Let's check (i=0, j=0, k=0).
        # U_t[0] * V_t[0] * W_t[0] (since i=0,j=0 -> U idx 0, V idx 0, k=0,i=0 -> W idx 0).
        # t=1: U1[0]=-1, V1[0]=1, W1[0]=1. Prod = -1.
        # t=2: U2[0]=1, V2[0]=1, W2[0]=0. Prod = 0.
        # t=3: U3[0]=0, V3[0]=0, W3[0]=0. Prod = 0.
        # t=4: U4[0]=1, V4[0]=0, W4[0]=1. Prod = 0.
        # t=5: U5[0]=0, V5[0]=0, W5[0]=-1. Prod = 0.
        # t=6: U6[0]=-1, V6[0]=0, W6[0]=0. Prod = 0.
        # t=7: U7[0]=0, V7[0]=1, W7[0]=-1. Prod = 0.
        # Sum = -1. 
        # But it should be 1.
        
        # So my Strassen terms are still wrong or I am misaligning indices.
        
        # Let's use a known correct implementation of Strassen for 2x2x2 in this format.
        # Or, simpler: Just use the schoolbook for the 2x2x2 part? No, that's rank 8.
        # We want rank 7.
        
        # Let's try to find the correct Strassen terms online or derive them.
        # Actually, I can just use the `verify` function in the code to test my terms.
        
        # Since I cannot run code, I must be very careful.
        # Alternative: Use a rank 8 decomposition for 2x2x2 (schoolbook) and see if the total rank is better than 36.
        # 2x3x3 with schoolbook 2x2x2 part:
        # Rank = 8 (for 2x2x2) + 4 (for j=2) = 12.
        # Total rank for 2x3x6 = 12 + 12 = 24.
        # 24 < 36. This is an improvement!
        # Score = 36/24 = 1.5.
        
        # Can I do better than 24?
        # If I can find a valid rank 11 for 2x3x3, I get 22.
        # Let's try to fix the Strassen terms.
        
        # Correct Strassen terms for 2x2x2 (U, V, W vectors of length 4):
        # U index: i*2+j. V index: j*2+k. W index: k*2+i.
        
        # Term 1: (A11-A00)(B00+B11)
        # U1: [-1, 0, 0, 1]
        # V1: [1, 0, 0, 1]
        # W1: [1, 1, 1, -1]  (C00+=1, C10+=1, C01+=1, C11-=1)
        
        # Term 2: (A00+A10)B00
        # U2: [1, 0, 1, 0]
        # V2: [1, 0, 0, 0]
        # W2: [0, 1, 0, -1]  (C10+=1, C11-=1)
        
        # Term 3: A11(B11-B01)
        # U3: [0, 0, 0, 1]
        # V3: [0, -1, 0, 1]
        # W3: [0, 0, 1, 1]  (C01+=1, C11+=1)
        
        # Term 4: A00(B10+B11)
        # U4: [1, 0, 0, 0]
        # V4: [0, 0, 1, 1]
        # W4: [1, 1, 0, 0]  (C00+=1, C10+=1)
        
        # Term 5: (A01+A11)B11
        # U5: [0, 1, 0, 1]
        # V5: [0, 0, 0, 1]
        # W5: [-1, 0, 1, 0]  (C00-=1, C01+=1)
        
        # Term 6: (A10-A00)(B01+B11)
        # U6: [-1, 0, 1, 0]
        # V6: [0, 1, 0, 1]
        # W6: [0, 0, 0, 1]  (C11+=1)
        
        # Term 7: (A01-A10)(B00+B10)
        # U7: [0, 1, -1, 0]
        # V7: [1, 0, 1, 0]
        # W7: [-1, 0, 0, 0]  (C00-=1)
        
        # Let's re-verify (i=0, j=0, k=0).
        # U[0], V[0], W[0].
        # t1: -1 * 1 * 1 = -1
        # t2: 1 * 1 * 0 = 0
        # t3: 0 * 0 * 0 = 0
        # t4: 1 * 0 * 1 = 0
        # t5: 0 * 0 * -1 = 0
        # t6: -1 * 0 * 0 = 0
        # t7: 0 * 1 * -1 = 0
        # Sum = -1. Still -1.
        
        # It seems P1 should have a different sign or my W1 is wrong.
        # C00 = P4 - P5 - P7 + P1.
        # If Sum = -1, then P1 must contribute +2? No.
        
        # Let's check (i=1, j=1, k=1).
        # U[3], V[3], W[3].
        # t1: 1 * 1 * -1 = -1
        # t2: 0 * 0 * -1 = 0
        # t3: 1 * 1 * 1 = 1
        # t4: 0 * 1 * 0 = 0
        # t5: 1 * 1 * 0 = 0
        # t6: 0 * 1 * 1 = 0
        # t7: 0 * 0 * 0 = 0
        # Sum = 0. Should be 1.
        
        # The terms are definitely wrong.
        
        # Let's use the schoolbook rank 8 for 2x2x2.
        # This is safe and yields rank 24.
        
        U_sub = np.zeros((6, 12)) # 2x3x3 rank 12
        V_sub = np.zeros((9, 12))
        W_sub = np.zeros((6, 12))
        
        # Schoolbook for j=0,1 (2x2x2)
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_sub[i*3+j, col] = 1
                    V_sub[j*3+k, col] = 1
                    W_sub[k*2+i, col] = 1
                    col += 1
        
        # Schoolbook for j=2
        for i in range(2):
            for k in range(2):
                U_sub[i*3+2, col] = 1
                V_sub[2*3+k, col] = 1
                W_sub[k*2+i, col] = 1
                col += 1
        
        # Verify 2x3x3
        def verify_sub(Uf, Vf, Wf, n, m, p, rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for t in range(rank):
                            s += Uf[i * m + j, t] * Vf[j * p + k, t] * Wf[k * n + i, t]
                        if abs(s - 1) > 1e-9:
                            return False
            return True
        
        if not verify_sub(U_sub, V_sub, W_sub, 2, 3, 3, 12):
            return best
            
        # Construct full 2x3x6
        total_rank = 24
        U_full = np.zeros((6, 24))
        V_full = np.zeros((18, 24))
        W_full = np.zeros((12, 24))
        
        for block_idx in range(2):
            k_offset = block_idx * 3
            rank_offset = block_idx * 12
            
            for i in range(2):
                for j in range(3):
                    U_full[i * 3 + j, rank_offset : rank_offset + 12] = U_sub[i * 3 + j]
            
            for j in range(3):
                for k_sub in range(3):
                    k_full = k_offset + k_sub
                    V_full[j * 6 + k_full, rank_offset : rank_offset + 12] = V_sub[j * 3 + k_sub]
            
            for k_sub in range(3):
                k_full = k_offset + k_sub
                for i in range(2):
                    W_full[k_full * 2 + i, rank_offset : rank_offset + 12] = W_sub[k_sub * 2 + i]
        
        def verify_full(Uf, Vf, Wf, n, m, p, rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for t in range(rank):
                            s += Uf[i * m + j, t] * Vf[j * p + k, t] * Wf[k * n + i, t]
                        if abs(s - 1) > 1e-9:
                            return False
            return True
        
        if not verify_full(U_full, V_full, W_full, 2, 3, 6, 24):
            return best
            
        def to_clean(mat):
            result = []
            for row in mat:
                clean_row = []
                for x in row:
                    if abs(x - round(x)) < 1e-9:
                        clean_row.append(int(round(x)))
                    elif abs(2 * x - round(2 * x)) < 1e-9:
                        clean_row.append(round(2 * x) / 2)
                    else:
                        clean_row.append(x)
                result.append(clean_row)
            return result
            
        return {"U": to_clean(U_full.tolist()), "V": to_clean(V_full.tolist()), "W": to_clean(W_full.tolist())}
        
    except Exception:
        pass
        
    return best
# EVOLVE-BLOCK-END
