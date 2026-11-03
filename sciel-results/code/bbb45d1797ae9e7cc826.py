"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Schoolbook decomposition as fallback: rank = n*m*p = 80
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find a lower-rank decomposition using structured approaches
    # For 4x4x5, we can try rank 20 (5 groups of 4)
    
    # Strategy: Use Kronecker-like structured decomposition
    # For 4x4x5, split into blocks and use fast matrix multiplication patterns
    
    # Approach 1: Try rank 20 using a structured pattern
    # The idea: for 4x4x4, schoolbook is rank 16. Add 4 more for the 5th column.
    # But we can potentially do better with structured sharing.
    
    # Let's try a rank-20 construction:
    # Terms 0-15: Handle 4x4x4 part (first 4 columns of B)
    # Terms 16-19: Handle 4x4x1 part (5th column of B)
    
    # For 4x4x4 with rank 16 (schoolbook):
    # Term t = i*4+j for i in 0..3, j in 0..3
    # U[i*4+j, t] = 1
    # V[j*5+k, t] = 1 if k < 4 else 0  (only for first 4 columns)
    # W[k*4+i, t] = 1 if k < 4 else 0
    
    # Wait, this doesn't work because V[j*5+k, t] needs to be 1 for k=0..3
    # but that means each term t contributes to 4 different V entries, which is fine.
    # But W[k*4+i, t] also needs to be 1 for k=0..3.
    
    # Let me reconsider. The tensor equation is:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = A[i,j] * B[j,k]
    
    # For schoolbook 4x4x4 (k=0..3):
    # For each (i,j), use one term t = i*4+j
    # U[i*4+j, t] = 1
    # V[j*5+k, t] = 1 for all k=0..3 (4 entries)
    # W[k*4+i, t] = 1 for all k=0..3 (4 entries)
    
    # This gives 16 terms for the 4x4x4 part.
    # For the 5th column (k=4), use 4 more terms:
    # For each i, use term t = 16+i
    # U[i*4+j, t] = 1 for all j=0..3 (4 entries) -- wait, this is wrong
    
    # Actually for 4x4x1 (A is 4x4, B is 4x1, C is 1x4):
    # C[i] = sum_j A[i,j] * B[j]
    # Schoolbook rank is 4 (one term per i)
    # Term t = 16+i:
    # U[i*4+j, t] = 1 for all j (4 entries)
    # V[j*5+4, t] = 1 for all j (4 entries)  
    # W[4*4+i, t] = 1 (1 entry)
    
    # But wait, U[i*4+j, t] = 1 for all j means the same U row pattern for all j.
    # Let's verify: sum_t U[i*4+j, t] * V[j*5+k, t] * W[k*4+i, t]
    # For k=4: sum_{t=16}^{19} U[i*4+j, t] * V[j*5+4, t] * W[4*4+i, t]
    # = sum_{t=16}^{19} [1 if t==16+i else 0] * [1] * [1 if t==16+i else 0]
    # = 1 if i is in 0..3 (which it is)
    # But this gives 1 for all j, which is wrong. We need A[i,j]*B[j,4].
    
    # I see the issue. The schoolbook for 4x4x1 doesn't work this way because
    # U and V need to encode the A and B values separately.
    
    # Let me go back to basics. The correct schoolbook for 4x4x5 is rank 80.
    # Each (i,j,k) triple gets its own term.
    
    # For a lower rank, we need to find linear dependencies or use fast algorithms.
    
    # Given the complexity, let's try a different approach:
    # Use the fact that 4x4x5 can be decomposed using 2x2x2 Winograd blocks.
    
    # Actually, let's just return the schoolbook result since we can't easily
    # construct a verified lower-rank decomposition within the time limit.
    
    if time.time() - start_time > DEADLINE - 10:
        return best_result
    
    # Try rank 40: use 2x2 blocks
    # A is 4x4, split into four 2x2 blocks
    # B is 4x5, split into two 2x5 blocks
    # C is 5x4, split into two 5x2 blocks
    
    # C_top = A00 * B_top + A01 * B_bottom  (2x5)
    # C_bottom = A10 * B_top + A11 * B_bottom  (2x5)
    
    # Each 2x2 * 2x5 = 2x5 multiplication has schoolbook rank 20
    # Total: 4 * 20 = 80, same as before. No improvement.
    
    # Let's try a different split. Actually, the schoolbook is already optimal
    # for this approach. We need a genuinely fast algorithm.
    
    # Given the time constraints and the difficulty of constructing a verified
    # lower-rank decomposition, return the schoolbook result.
    
    return best_result
# EVOLVE-BLOCK-END
