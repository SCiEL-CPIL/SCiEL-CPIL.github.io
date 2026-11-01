"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _verify_fast(U, V, W, n, m, p, rank):
    """
    Fast verification using integer arithmetic.
    U, V, W are lists of lists of integers (representing 2*x).
    The tensor identity is sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k.
    With scaled integers (2*x), the sum must be 8 (since 2^3 * 1 = 8).
    """
    # Pre-compute products U[i,j,c] * V[j,k,c] for all i,j,k,c
    # Dimensions: n=2, m=2, p=5
    # U: 4 x rank
    # V: 10 x rank
    # W: 10 x rank
    
    # For each c, we have a term T_c = U_c * V_c * W_c (outer product)
    # We need sum_c T_c[i,j,k] == 8 for all i,j,k.
    
    # Efficient check:
    # For each c, compute the 2x2x5 tensor contribution.
    # Sum them up.
    
    # To speed up, we can check each (i,j,k) independently but vectorized-ish.
    # Since rank is small (<20), simple loops are fine.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 8:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    Strategy:
    1. Use Strassen's algorithm for the first 2 columns (k=0,1) of the 2x5 output.
       This handles the 2x2x2 sub-tensor with rank 7.
    2. The remaining 3 columns (k=2,3,4) of the 2x5 output are 2x2x3 tensors.
       The full tensor is the sum of the 2x2x2 part and the 2x2x3 part.
       Wait, the tensor is T[i,j,k] = 1 for all i,j,k.
       So T = T_2x2x2 (for k=0,1) + T_2x2x3 (for k=2,3,4).
       
       Actually, it's easier to think of the full 2x2x5 tensor as a single object.
       We can split the output index k into two parts: k1 in {0,1} and k2 in {2,3,4}.
       
       Let's use a block decomposition approach.
       We know a rank-7 decomposition for the 2x2x2 identity tensor.
       Let's call the factors for this part U1, V1, W1 (with rank 7).
       U1 has shape (4, 7), V1 has shape (10, 7), W1 has shape (10, 7).
       
       For the Strassen part (k=0,1):
       The output indices for W are k*n+i.
       For k=0,1 and i=0,1, the W indices are 0,1,2,3.
       For k=2,3,4 and i=0,1, the W indices are 4,5,6,7,8,9.
       
       We can define U1, V1, W1 such that they only act on the k=0,1 outputs.
       Specifically, W1[c][k*n+i] is non-zero only if k is 0 or 1.
       
       Then we need to find U2, V2, W2 of rank r2 such that:
       sum_{c in 0..6} U1[i,j,c]V1[j,k,c]W1[k,i,c] + sum_{c in 7..7+r2-1} U2[i,j,c]V2[j,k,c]W2[k,i,c] = 1
       
       For k=0,1, the second sum must be 0.
       For k=2,3,4, the first sum is 0 (since W1 is 0 there), so the second sum must be 1.
       
       So we need a decomposition of the 2x2x3 identity tensor (for k=2,3,4) into rank r2.
       And we need U2, V2 to be "orthogonal" to the W1 support in a way that they don't interfere?
       No, the condition is just that the total sum is 1.
       
       If we set W2 to be non-zero ONLY for k=2,3,4, then the interference is automatically 0 for k=0,1.
       And for k=2,3,4, W1 is 0, so we just need the 2x2x3 part to be 1.
       
       So the problem reduces to finding the rank of the 2x2x3 identity tensor!
       Let's call this rank R_2x2x3.
       Then the total rank is 7 + R_2x2x3.
       
       What is the rank of the 2x2x3 identity tensor?
       Lower bound: max(2*2, 2*3, 3*2) = max(4, 6, 6) = 6.
       Schoolbook rank: 2*2*3 = 12.
       
       Let's search for the rank of the 2x2x3 identity tensor.
       If we can find rank 6, total rank is 13.
       If rank 7, total rank is 14.
       
       Let's hardcode the Strassen factors for 2x2x2.
       Strassen's 7 products for C = A*B:
       M1 = (A11 + A22)(B11 + B22)
       M2 = (A21 + A22)B11
       M3 = A11(B21 - B22)
       M4 = A22(B12 - B11)
       M5 = (A11 + A12)B22
       M6 = (A21 - A11)(B11 + B12)
       M7 = (A12 - A22)(B21 + B22)
       
       C11 = M1 + M4 - M5 + M7
       C12 = M3 + M5
       C21 = M2 + M4
       C22 = M1 - M2 + M3 + M6
       
       We need to map this to U, V, W.
       U[i,j,c] corresponds to the coefficient of A[i,j] in product c.
       V[j,k,c] corresponds to the coefficient of B[j,k] in product c.
       W[k,i,c] corresponds to the coefficient of product c in C[k,i]. Note: C[k,i] is the element at row k, col i?
       Wait, the task says "output factor follows the official trace convention W[k*n+i,r]".
       And the tensor is T[i,j,k]. Usually T[i,j,k] = A[i,j]B[j,k] -> C[i,k].
       So C[i,k] = sum_j A[i,j]B[j,k].
       The index in W is k*n+i. This suggests the output matrix is indexed as C[k,i]?
       Let's re-read carefully: "multiplying a 2 by 2 matrix by a 2 by 5 matrix".
       A is 2x2 (n=2, m=2). B is 2x5 (m=2, p=5).
       Result C is 2x5 (n=2, p=5).
       Standard matmul: C[i,k] = sum_j A[i,j]B[j,k].
       The task says W index is k*n+i.
       If n=2, indices are:
       k=0, i=0 -> 0
       k=0, i=1 -> 1
       k=1, i=0 -> 2
       k=1, i=1 -> 3
       ...
       This implies the output matrix is stored with k as the major index? i.e., C[k,i]?
       Or is it just a flat index?
       If C is 2x5, standard row-major is i*p+k.
       If C is 5x2, standard row-major is k*n+i.
       The problem says "2 by 2 matrix by a 2 by 5 matrix".
       Usually this means A(2x2) * B(2x5) = C(2x5).
       If the output factor W has index k*n+i, and n=2, p=5, then the index range is 0..9.
       If it's C[i,k], index would be i*5+k.
       If it's C[k,i], index would be k*2+i.
       The prompt says "W[k*n+i,r]". This strongly implies the output is treated as a p x n matrix (5x2) or the indices are swapped.
       Let's assume the tensor T[i,j,k] corresponds to A[i,j] * B[j,k] contributing to the output element at position (k,i) in the W matrix's row index.
       So W_row = k*n + i.
       This means the output matrix C is effectively C[k,i] = sum_j A[i,j]B[j,k].
       This is the transpose of the standard C[i,k].
       Does this change the rank? No, rank is invariant under permutation of indices.
       So we can just swap the roles of i and k in the output if needed, or just be careful with indices.
       
       Let's stick to the tensor definition:
       Sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
       
       For the 2x2x2 sub-problem (k in {0,1}):
       We need Sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for i,j in {0,1}, k in {0,1}.
       This is exactly the matrix multiplication tensor for 2x2 * 2x2 = 2x2.
       The rank is 7.
       
       For the 2x2x3 sub-problem (k in {2,3,4}):
       We need Sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for i,j in {0,1}, k in {2,3,4}.
       Let's map k' = k - 2, so k' in {0,1,2}.
       This is the tensor for 2x2 * 2x3 = 2x3 (with output indexed by k',i).
       We need to find the rank of this specific tensor.
       
       Let's search for the rank of the 2x2x3 identity tensor.
       Lower bound 6.
       
       I will implement a search for the 2x2x3 tensor rank.
       If found rank r2, total rank = 7 + r2.
       
       Let's try to find r2.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # 1. Construct Strassen decomposition for the 2x2x2 part (k=0,1)
    # U1: 4x7, V1: 10x7, W1: 10x7
    # Indices:
    # U: i*m+j (0..3)
    # V: j*p+k (0..9)
    # W: k*n+i (0..9)
    
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(10)]
    W1 = [[0]*7 for _ in range(10)]
    
    # Strassen products
    # M1: (A11+A22)(B11+B22)
    # A indices: 11->0, 22->1 (if 0-indexed: 00->0, 11->1)
    # Let's use 0-indexed:
    # A[0,0]=A00, A[0,1]=A01, A[1,0]=A10, A[1,1]=A11
    # B[0,0]=B00, B[0,1]=B01, B[1,0]=B10, B[1,1]=B11
    # C[0,0]=C00, C[0,1]=C01, C[1,0]=C10, C[1,1]=C11
    
    # M1 = (A00+A11)(B00+B11)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1  # B00 is j=0,k=0 -> 0*5+0=0. B11 is j=1,k=1 -> 1*5+1=6?
    # Wait, V index is j*p+k.
    # For 2x2x2, p=2.
    # j=0,k=0 -> 0
    # j=0,k=1 -> 1
    # j=1,k=0 -> 2
    # j=1,k=1 -> 3
    # But our full V has p=5.
    # So for the sub-problem k in {0,1}, the V indices are:
    # j=0,k=0 -> 0
    # j=0,k=1 -> 1
    # j=1,k=0 -> 5
    # j=1,k=1 -> 6
    # Let's correct V1 indices.
    
    # Reset V1
    V1 = [[0]*7 for _ in range(10)]
    
    # M1: (A00+A11)(B00+B11)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[6][0] = 1 # B00 (j=0,k=0), B11 (j=1,k=1)
    # W1: C00 (k=0,i=0 -> 0), C01 (k=1,i=0 -> 2), C10 (k=0,i=1 -> 1), C11 (k=1,i=1 -> 3)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W1[0][0] = 1  # C00
    W1[2][0] = 0  # C01
    W1[1][0] = 0  # C10
    W1[3][0] = 1  # C11
    
    # M2: (A10+A11)B00
    U1[2][1] = 1; U1[3][1] = 1 # A10 (i=1,j=0 -> 1*2+0=2), A11 (i=1,j=1 -> 3)
    V1[0][1] = 1               # B00
    W1[1][1] = 1               # C10
    W1[3][1] = -1              # C11
    
    # M3: A00(B10-B11)
    U1[0][2] = 1               # A00
    V1[5][2] = 1; V1[6][2] = -1 # B10 (j=1,k=0 -> 5), B11 (j=1,k=1 -> 6)
    W1[2][2] = 1               # C01
    W1[3][2] = 1               # C11
    
    # M4: A11(B01-B00)
    U1[3][3] = 1               # A11
    V1[1][3] = 1; V1[0][3] = -1 # B01 (j=0,k=1 -> 1), B00 (j=0,k=0 -> 0)
    W1[0][3] = 1               # C00
    W1[1][3] = 1               # C10
    
    # M5: (A00+A01)B11
    U1[0][4] = 1; U1[1][4] = 1 # A00 (0), A01 (i=0,j=1 -> 1)
    V1[6][4] = 1               # B11
    W1[0][4] = -1              # C00
    W1[2][4] = 1               # C01
    
    # M6: (A10-A00)(B00+B01)
    U1[2][5] = 1; U1[0][5] = -1 # A10 (2), A00 (0)
    V1[0][5] = 1; V1[1][5] = 1  # B00 (0), B01 (1)
    W1[3][5] = 1               # C11
    
    # M7: (A01-A11)(B10+B11)
    U1[1][6] = 1; U1[3][6] = -1 # A01 (1), A11 (3)
    V1[5][6] = 1; V1[6][6] = 1  # B10 (5), B11 (6)
    W1[0][6] = 1               # C00
    W1[3][6] = 1               # C11
    
    # Verify Strassen part for k=0,1
    # We'll do this in the final verification.
    
    # 2. Search for rank of 2x2x3 identity tensor (k=2,3,4)
    # We need U2, V2, W2 such that sum_c U2[i,j,c]V2[j,k,c]W2[k,i,c] = 1 for k in {2,3,4}.
    # And 0 for k in {0,1}.
    # We can enforce W2[k,i,c] = 0 for k in {0,1}.
    # So W2 only has non-zeros at indices 4..9.
    # Indices for k=2,3,4:
    # k=2: 4,5
    # k=3: 6,7
    # k=4: 8,9
    
    # Let's search for r2.
    # Lower bound 6.
    
    best_r2 = 12 # Schoolbook
    best_U2 = None
    best_V2 = None
    best_W2 = None
    
    # Values to try
    values = [-2, -1, 0, 1, 2]
    
    for r2 in range(6, 12):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Random search
        # To speed up, we can fix some structure or use smarter sampling.
        # But let's try random first with a tight loop.
        
        # Optimization: Since the tensor is "identity", we can try to find a decomposition
        # that is a sum of rank-1 tensors.
        
        max_attempts = 50000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            U2 = [[random.choice(values) for _ in range(r2)] for _ in range(4)]
            V2 = [[random.choice(values) for _ in range(r2)] for _ in range(10)]
            W2 = [[0]*r2 for _ in range(10)]
            
            # Fill W2 only for k=2,3,4
            for i in range(2):
                for k in range(2, 5):
                    idx = k*2 + i
                    W2[idx] = [random.choice(values) for _ in range(r2)]
            
            # Check
            # We only need to check k=2,3,4 for value 1, and k=0,1 for value 0.
            # But since W2 is 0 for k=0,1, the sum is 0 automatically.
            # So we only check k=2,3,4.
            
            ok = True
            for i in range(2):
                for j in range(2):
                    for k in range(2, 5):
                        s = 0
                        for c in range(r2):
                            s += U2[i*2+j][c] * V2[j*5+k][c] * W2[k*2+i][c]
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                found = True
                best_r2 = r2
                best_U2 = U2
                best_V2 = V2
                best_W2 = W2
                break
        
        if found:
            break
    
    # Combine U1, U2
    total_rank = 7 + best_r2
    U = [[0]*total_rank for _ in range(4)]
    V = [[0]*total_rank for _ in range(10)]
    W = [[0]*total_rank for _ in range(10)]
    
    # Copy U1, V1, W1 to first 7 columns
    for i in range(4):
        for c in range(7):
            U[i][c] = U1[i][c]
    for i in range(10):
        for c in range(7):
            V[i][c] = V1[i][c]
            W[i][c] = W1[i][c]
    
    # Copy U2, V2, W2 to next best_r2 columns
    for i in range(4):
        for c in range(best_r2):
            U[i][7+c] = best_U2[i][c]
    for i in range(10):
        for c in range(best_r2):
            V[i][7+c] = best_V2[i][c]
            W[i][7+c] = best_W2[i][c]
    
    # Final verification with exact integers
    # Note: U, V, W are integers. The product sum must be 1.
    # My _verify_fast checks for 8 because it assumed scaling.
    # Here, the entries are integers from {-2,-1,0,1,2}.
    # The tensor identity is sum U*V*W = 1.
    # So we check for 1.
    
    # Let's write a simple verifier for this specific case.
    def verify_final(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    if not verify_final(U, V, W, n, m, p, total_rank):
        # Fallback to schoolbook if something went wrong
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        total_rank = r
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
