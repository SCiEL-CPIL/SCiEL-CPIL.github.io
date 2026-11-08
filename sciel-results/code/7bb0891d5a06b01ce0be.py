# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # Strassen's 7 rank-1 tensors for 2x2x2
    # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i (trace convention)
    
    # Correct Strassen decomposition for 2x2x2
    # Reference: Strassen (1969)
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10+b11)
    # M4 = a11(b00-b10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Let's verify the W vectors for these formulas.
    # W vector for each M_r has 4 entries [w00, w10, w01, w11] corresponding to C[0,0], C[1,0], C[0,1], C[1,1]
    # Note: W index is k*2+i.
    # C[0,0] -> k=0, i=0 -> idx 0
    # C[1,0] -> k=0, i=1 -> idx 1
    # C[0,1] -> k=1, i=0 -> idx 2
    # C[1,1] -> k=1, i=1 -> idx 3
    
    # M1: C00+1, C11+1 -> w=[1,0,0,1]
    # M2: C10+1, C11-1 -> w=[0,1,0,-1]
    # M3: C01+1, C11+1 -> w=[0,0,1,1]
    # M4: C00+1, C10+1 -> w=[1,1,0,0]
    # M5: C00-1, C01+1 -> w=[-1,0,1,0]
    # M6: Not used in C? 
    # M7: C00-1, C11+1 -> w=[-1,0,0,1]
    
    # If M6 is not used, then we only have 6 terms?
    # No, the rank is the number of terms in the decomposition.
    # If M6 is not in the linear combination, it's not a term.
    # But Strassen is rank 7.
    # Let's check if M6 is needed.
    # If I use 6 terms, rank is 6. But rank 2x2x2 is 7.
    # So M6 MUST be used.
    
    # Let's check the formulas again.
    # Maybe C10 = M2 + M4 is wrong.
    # C10 = a10*b00 + a11*b10.
    # M2 = (a10+a11)b00 = a10*b00 + a11*b00.
    # M4 = a11(b00-b10) = a11*b00 - a11*b10.
    # M2 + M4 = a10*b00 + 2*a11*b00 - a11*b10.
    # This is not C10.
    
    # Correct formulas:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4  <-- This is the standard one I see everywhere. Why is it failing my check?
    # Ah, maybe my M2 or M4 is defined differently.
    # Standard Strassen:
    # M2 = (a10 + a11) b00
    # M4 = a11 (b00 - b10)
    # Sum = a10*b00 + a11*b00 + a11*b00 - a11*b10 = a10*b00 + 2*a11*b00 - a11*b10.
    # C10 = a10*b00 + a11*b10.
    # Difference: 2*a11*b00 - 2*a11*b10.
    
    # I suspect the standard formulas I'm recalling are for a different permutation or I am missing a term.
    # Actually, there are multiple variants of Strassen.
    
    # To be safe, I will use the schoolbook decomposition for the 2x2 blocks but combined in a way that might allow sharing? No, that's hard.
    
    # Instead, I will use the known rank-7 decomposition from a reliable source.
    # One such source is the `tensordot` examples or specific papers.
    
    # Let's try this set of 7 tensors which is known to work:
    # (u, v, w)
    # 1. [1,0,1,0], [1,0,0,1], [1,0,0,1]
    # 2. [0,0,1,1], [1,0,0,0], [0,1,0,-1]
    # 3. [1,0,0,0], [0,0,1,1], [0,0,1,1]
    # 4. [0,0,0,1], [1,0,-1,0], [1,1,0,0]
    # 5. [1,1,0,0], [0,0,0,1], [-1,0,1,0]
    # 6. [-1,0,1,0], [1,1,0,0], [0,0,0,0] -> This is the issue.
    
    # I will fall back to schoolbook rank 96 if I cannot verify Strassen.
    # But I can try to use a rank-8 decomposition for 2x2x2 (schoolbook) and see if I can merge any? No.
    
    # Let's just use the schoolbook rank 96 for now, but with a more efficient construction?
    # No, the score is 1.0 (rank 96). I need to beat 96.
    
    # What if I use the fact that 4x4x6 is equivalent to 4x4x4 + 4x4x2?
    # Rank 4x4x4 is 49. Rank 4x4x2 is 16 (schoolbook). Total 65.
    # This is a huge improvement.
    # I need the rank-49 decomposition for 4x4x4.
    # This is complex to hardcode.
    
    # Alternative: Use the block Strassen with rank 84.
    # I will implement the verification. If it fails, I return schoolbook.
    # To increase the chance of Strassen working, I will use a set of tensors that I can verify in the code.
    
    # Let's define the 7 Strassen tensors for 2x2x2 and verify them against the multiplication tensor.
    # If they don't match, I will use schoolbook for the 2x2 blocks (rank 8 per block, total 96).
    
    strassen_tensors = [
        ([1,0,1,0], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        ([1,0,0,0], [0,0,1,1], [0,0,1,1]),
        ([0,0,0,1], [1,0,-1,0], [1,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,0]), # Placeholder, likely wrong
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,1])
    ]
    
    # Verify Strassen tensors
    T_strassen = np.zeros((4, 4, 4))
    for u, v, w in strassen_tensors:
        T_strassen += np.einsum('i,j,k->ijk', u, v, w)
    
    T_expected_2x2 = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_expected_2x2[i*2+j, j*2+k, k*2+i] = 1
    
    use_strassen = np.allclose(T_strassen, T_expected_2x2, atol=1e-9)
    
    if use_strassen:
        rank = 12 * 7
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        
        col = 0
        for i_block in range(2):
            for j_block in range(3):
                for k_block in range(2):
                    for u_local, v_local, w_local in strassen_tensors:
                        for i in range(2):
                            for j in range(2):
                                a_idx = (i_block*2 + i)*4 + (k_block*2 + j)
                                U[a_idx, col] += u_local[i*2 + j]
                        for i in range(2):
                            for j in range(2):
                                b_idx = (k_block*2 + i)*6 + (j_block*2+j)
                                V[b_idx, col] += v_local[i*2 + j]
                        for i in range(2):
                            for j in range(2):
                                c_idx = (j_block*2 + j)*4 + (i_block*2+i)
                                W[c_idx, col] += w_local[i*2 + j]
                        col += 1
    else:
        # Fallback to schoolbook
        rank = 96
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1.0
                    V[j*6+k, col] = 1.0
                    W[k*4+i, col] = 1.0
                    col += 1

    # Verify full decomposition
    T_full = np.zeros((16, 24, 24))
    for i in range(4):
        for j in range(4):
            for k in range(6):
                T_full[i*4+j, j*6+k, k*4+i] = 1.0

    T_decomp = np.zeros((16, 24, 24))
    for c in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    
    if not np.allclose(T_full, T_decomp, atol=1e-9):
        # If verification fails, return schoolbook
        rank = 96
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1.0
                    V[j*6+k, col] = 1.0
                    W[k*4+i, col] = 1.0
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
