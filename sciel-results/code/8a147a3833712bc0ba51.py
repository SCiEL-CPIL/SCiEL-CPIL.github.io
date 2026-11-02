"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS
    # The target is 2x3 @ 3x9 -> 2x9
    # We embed this into 3x3 @ 3x9 -> 3x9
    # The 3x9 output is decomposed into three 3x3 blocks: Y1, Y2, Y3
    # Y1 = A * B1
    # Y2 = A * B2
    # Y3 = A * B3
    # Then we map the 3x3 result back to 2x2 for each block? 
    # Wait, the mapping is:
    # Let A be 2x3. Let B be 3x9.
    # Let A_ext be 3x3 where A_ext[0:2, :] = A and A_ext[2, :] = A[0, :] + A[1, :]
    # Let B_ext be 3x9 where B_ext[:, :] = B (since B is 3x9, no change needed for columns, 
    # but the rows of B correspond to the columns of A. The rows of A_ext are 3, so B must be 3x9.
    # This fits perfectly.
    # Let C_ext = A_ext * B_ext (3x9)
    # C_ext[0:2, :] = A * B (the desired 2x9 result)
    # C_ext[2, :] = (A[0,:] + A[1,:]) * B = A[0,:]*B + A[1,:]*B = C[0,:] + C[1,:]
    # So the third row of C_ext is redundant. We only need to compute C_ext[0:2, :].
    # However, the tensor decomposition computes all of C_ext.
    # We can decompose the 3x3x9 multiplication tensor.
    # 3x3x9 = (3x3x3) x (3x3) ? No, 9 is 3*3.
    # Let's split the 9 columns into 3 groups of 3.
    # B_ext = [B1 | B2 | B3] where B1, B2, B3 are 3x3.
    # C_ext = [C1 | C2 | C3] where Ci = A_ext * Bi (3x3).
    # So we need to compute 3 separate 3x3 matrix multiplications.
    # Each 3x3 multiplication can be done with rank 5 (Strassen-like for 3x3? 
    # Actually, the rank of 3x3x3 is 5? No, Strassen is 2x2x2 rank 7.
    # For 3x3x3, the rank is known to be 19? Or is it lower?
    # Wait, the rank of 3x3 matrix multiplication is 19? 
    # Let's check: 2x2 is 7. 3x3 is at most 3^3=27. 
    # Actually, the rank of 3x3x3 is 19? No, I recall it might be 19 or 20.
    # Let's look up or derive. 
    # Actually, I can just use the schoolbook for 3x3x3 which is rank 9.
    # If I use rank 9 for each of the 3 blocks, total rank = 3 * 9 = 27.
    # But the original problem is 2x3x9 which is rank 54.
    # So rank 27 is a huge improvement!
    
    # Is the rank of 3x3x3 really 19? 
    # Let's try to find a decomposition for 3x3x3.
    # Actually, the rank of 3x3x3 is 19 is a known result? 
    # Let's check: The rank of 2x2x2 is 7.
    # The rank of 3x3x3 is at most 19? 
    # I don't have the exact decomposition memorized. 
    # Let's use the schoolbook for 3x3x3 (rank 9) to be safe.
    # Rank 27 is better than 54.
    
    # Let's construct the rank 27 decomposition.
    # We have 3 blocks of 3x3 multiplications.
    # For each block b (0, 1, 2):
    #   We compute Ci = A_ext * Bi.
    #   A_ext is 3x3, Bi is 3x3, Ci is 3x3.
    #   Schoolbook for 3x3x3:
    #   For i in 0..2, j in 0..2, k in 0..2:
    #     U[i*3+j] = 1, V[j*3+k] = 1, W[k*3+i] = 1
    #   This uses 9 terms per block.
    #   Total 27 terms.
    
    # Let's map the indices.
    # U shape: (3*3, 27) = (9, 27)
    # V shape: (3*9, 27) = (27, 27)
    # W shape: (9*3, 27) = (27, 27)
    
    # Wait, the output is 3x9, so W should be (9*3, 27) = (27, 27).
    # The input A_ext is 3x3, so U is (3*3, 27) = (9, 27).
    # The input B_ext is 3x9, so V is (3*9, 27) = (27, 27).
    
    # Let's build it.
    U = np.zeros((9, 27), dtype=int)
    V = np.zeros((27, 27), dtype=int)
    W = np.zeros((27, 27), dtype=int)
    
    col = 0
    for b in range(3):  # 3 blocks
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # U index: i*3 + j
                    # V index: j*9 + (b*3 + k)  # B_ext is 3x9, row j, column b*3+k
                    # W index: (b*3 + k)*3 + i   # C_ext is 3x9, row b*3+k? No.
                    # C_ext is 3x9. Row i, column b*3+k.
                    # W index: i*9 + (b*3 + k)
                    U[i*3 + j, col] = 1
                    V[j*9 + (b*3 + k), col] = 1
                    W[i*9 + (b*3 + k), col] = 1
                    col += 1
    
    # Now we have a decomposition for 3x3x9.
    # But we need 2x3x9.
    # The output C_ext is 3x9. We only need the first 2 rows.
    # The W matrix has 27 rows (3x9). We only need the first 18 rows (2x9).
    # However, the task requires W to have shape (p*n, rank) = (9*2, rank) = (18, rank).
    # So we just take the first 18 rows of W.
    # U has shape (9, 27). But we need U to have shape (n*m, rank) = (2*3, rank) = (6, rank).
    # A_ext is 3x3. We only need the first 2 rows of A_ext.
    # So we take the first 6 rows of U.
    # V has shape (27, 27). We need V to have shape (m*p, rank) = (3*9, rank) = (27, rank).
    # B_ext is 3x9. This matches exactly.
    
    # So:
    U_final = U[:6, :]
    V_final = V
    W_final = W[:18, :]
    
    # Check if we're within deadline before returning
    if time.time() < deadline:
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    else:
        # Fallback: return schoolbook decomposition
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
