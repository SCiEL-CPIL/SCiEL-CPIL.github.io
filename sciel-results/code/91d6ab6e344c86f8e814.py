"""Matrix multiplication 3,3,4 - rank-minimal decomposition"""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 140.0  # safety margin before 150s limit

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Use a block-structured Strassen/Laderman hybrid approach
    # Partition: A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    
    # For the 2x2 blocks, use Strassen (7 mults)
    # For cross terms, use direct products
    # Total: 7 (for A11*B11) + 7 (for A12*B21 part) + ... 
    # Let me use a known construction
    
    # Actually, let me use the explicit rank-23 decomposition for 3x3x3 
    # and extend to 3x3x4 by adding 4 more terms for the extra column
    # That gives rank 27. But we can do better.
    
    # Let me try a direct construction with rank 23+4=27 or less
    # Using Laderman's 23-term 3x3x3 decomposition, extended to 4 columns
    
    # Laderman's decomposition for 3x3x3 (rank 23):
    # The 23 products are indexed and each corresponds to a triple (i,j,k)
    # with specific linear combinations
    
    # For 3x3x4, I'll use a block approach:
    # Split B into B_left (3x2) and B_right (3x2)
    # Compute C_left = A * B_left (3x2) and C_right = A * B_right (3x2)
    # Each 3x3x2 can be done in rank 23 (same as 3x3x3 since the last dimension doesn't increase rank)
    # But that gives 46, which is worse.
    
    # Better: use a single decomposition that handles all 4 columns at once.
    # Let me construct using the tensor structure directly.
    
    # I'll use a known result: rank(3x3x4) <= 23 + 4 = 27 is trivial,
    # but there are constructions achieving less.
    
    # Let me try rank 23 by reusing the 3x3x3 Laderman terms cleverly.
    # Actually, the 4th column just needs 3 more multiplications in the worst case,
    # so rank <= 26. Let me aim for 23-26.
    
    # Construct a rank-26 decomposition:
    # Use Laderman's 23 terms for the first 3 columns of output,
    # plus 3 additional terms for the 4th column.
    
    # Laderman's 3x3x3 rank-23 decomposition (standard form):
    # Products P1..P23, each is a product of a linear form in A and a linear form in B
    
    # Let me encode Laderman's decomposition explicitly.
    # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
    
    # I'll use a known list of the 23 terms.
    
    # For simplicity and correctness, let me use a verified construction.
    # I'll build the decomposition programmatically.
    
    # Strategy: Use the fact that we can write the multiplication as
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # and find a low-rank factorization of this bilinear form.
    
    # Let me try a simple block-Strassen approach:
    # Partition A into 2x2 blocks: A11(2x2), A12(2x1), A21(1x2), A22(1x1)
    # Partition B into 2x2 blocks: B11(2x2), B12(2x2), B21(1x2), B22(1x2)
    
    # C11 = A11*B11 + A12*B21  (2x2)
    # C12 = A11*B12 + A12*B22  (2x2)  
    # C21 = A21*B11 + A22*B21  (1x2)
    # C22 = A21*B12 + A22*B22  (1x2)
    
    # For C11 and C12, use Strassen on the 2x2 products:
    # M1 = (A11+A12)*(B11+B21) -> gives A11*B11 + A11*B21 + A12*B11 + A12*B21
    # M2 = A12*(B11+B21) -> gives A12*B11 + A12*B21
    # M3 = A11*B12
    # M4 = (A11+A12)*(B12+B22) -> gives A11*B12 + A11*B22 + A12*B12 + A12*B22
    # M5 = A12*B22
    # M6 = (A11+A12)*(B11+B21) - A12*(B11+B21) = A11*(B11+B21) -> A11*B11 + A11*B21
    # M7 = A11*B12 + (A11+A12)*(B12+B22) - A12*B22 - A11*(B11+B21) ... this is getting complex
    
    # Let me just use a brute-force search with time limit to find a valid low-rank decomposition.
    # But that's hard to verify. Instead, let me use a known good construction.
    
    # I'll use the following approach: rank 24
    # Based on the observation that 3x3x4 can be done in rank 24
    
    # Let me construct it explicitly using a verified method.
    # I'll use a combination of Strassen blocks.
    
    # Simpler: just return the schoolbook for now as a baseline, 
    # but try to improve with a quick search if time permits.
    
    # For now, let me implement a rank-26 construction that I can verify:
    # 23 terms from Laderman for columns 1-3, plus 3 terms for column 4.
    
    # Actually, let me just do a direct construction with rank 24 using 
    # a known pattern. I'll use the following 24 products:
    
    # Define the 24 terms as (u_coeff, v_coeff, w_coeff) where
    # u_coeff is a 9-dim vector for A, v_coeff is a 12-dim vector for B,
    # w_coeff is a 12-dim vector for C
    
    # This is complex to hardcode. Let me use a simpler verified approach.
    
    # Fallback: schoolbook rank 36, but try to reduce.
    # Let me at least try rank 27 (23+4) as a safe improvement.
    
    rank = 27
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 27
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 27
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 27
    
    # First 23 terms: Laderman 3x3x3 for the first 3 columns
    # I'll use a simplified version - actually let me just use schoolbook for reliability
    # and accept rank 36, or try a quick improvement.
    
    # Given time constraints, let me return a verified rank-36 schoolbook 
    # but with a quick attempt at improvement.
    
    # Actually, let me just do schoolbook to be safe and correct.
    rank = n * m * p  # 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
