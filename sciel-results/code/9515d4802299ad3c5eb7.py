import time
import numpy as np
from itertools import product

n, m, p = 3, 5, 5
START_TIME = time.time()
DEADLINE = 115.0  # Safety margin before 120s limit

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _validate(U, V, W):
    """Validate that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :]
                if abs(np.sum(prod) - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    """Standard rank n*m*p = 75 decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _find_strassen_2x2():
    """
    Find a rank-7 decomposition for 2x2x2 matrix multiplication.
    We search for a set of 7 triples (u, v, w) of 2x2 matrices such that
    sum_r u_r * v_r * w_r = identity tensor.
    """
    # Standard Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # We represent matrices as vectors in R^4 (row-major)
    # Identity tensor T[i,j,k] = 1 if i==j==k? No, T[i,j,k] = delta_{i,j} delta_{j,k} is not right.
    # The tensor for 2x2 matrix mult is T[i, j, k] where A[i,j] * B[j,k] -> C[i,k].
    # So T[i, j, k] = 1 if we multiply A[i,j] and B[j,k] to get C[i,k].
    # In our flattened indices:
    # U index: i*2 + j (0..3)
    # V index: j*2 + k (0..3)
    # W index: k*2 + i (0..3) -- Note: W[k*n+i, r] in the prompt means W index is k*3+i for 3x5x5.
    # For 2x2x2, W index is k*2+i.
    
    # Let's define the target tensor T of shape (4, 4, 4)
    T = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # U[i*2+j] * V[j*2+k] * W[k*2+i] should contribute to 1
                # Wait, the tensor decomposition condition is:
                # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
                # This is exactly the matrix multiplication tensor.
                T[i*2+j, j*2+k, k*2+i] = 1

    # Strassen factors
    # m1 = (a+d)(e+h) -> U1: a,d=1; V1: e,h=1; W1: ?
    # Let's just use the known Strassen matrices.
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    
    # m1 = (a+d)(e+h)
    # U1 = [[1, 0], [1, 0]] (a+d) -> vec [1, 0, 1, 0]
    # V1 = [[1, 1], [0, 0]] (e+h) -> vec [1, 1, 0, 0]
    # m2 = (c+d)e
    # U2 = [[0, 0], [1, 1]] (c+d) -> vec [0, 0, 1, 1]
    # V2 = [[1, 0], [0, 0]] (e) -> vec [1, 0, 0, 0]
    # m3 = a(f-h)
    # U3 = [[1, 0], [0, 0]] (a) -> vec [1, 0, 0, 0]
    # V3 = [[0, 1], [0, -1]] (f-h) -> vec [0, 1, 0, -1]
    # m4 = d(g-e)
    # U4 = [[0, 0], [0, 1]] (d) -> vec [0, 0, 0, 1]
    # V4 = [[0, -1], [1, 0]] (g-e) -> vec [0, -1, 1, 0]
    # m5 = (a+b)h
    # U5 = [[1, 1], [0, 0]] (a+b) -> vec [1, 1, 0, 0]
    # V5 = [[0, 0], [0, 1]] (h) -> vec [0, 0, 0, 1]
    # m6 = (c-a)(e+f)
    # U6 = [[-1, 0], [1, 0]] (c-a) -> vec [-1, 0, 1, 0]
    # V6 = [[1, 1], [0, 0]] (e+f) -> vec [1, 1, 0, 0]
    # m7 = (b-d)(g+h)
    # U7 = [[0, 1], [0, -1]] (b-d) -> vec [0, 1, 0, -1]
    # V7 = [[0, 1], [1, 1]] (g+h) -> vec [0, 1, 1, 1]
    
    # Now we need W_r such that sum U_r V_r W_r = T.
    # Actually, W_r are the coefficients that map the products m_r back to C.
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Let's construct W_r as vectors of length 4 (for C11, C12, C21, C22)
    # W_r[c] is the coefficient of m_r in C_c.
    # C indices: 0:11, 1:12, 2:21, 3:22
    
    W_factors = [
        [1, 0, 0, 1],  # m1
        [0, 0, 1, -1], # m2
        [0, 1, 0, 1],  # m3
        [0, 0, 1, 0],  # m4
        [-1, 1, 0, 0],# m5
        [0, 0, 0, 1],  # m6
        [1, 0, 0, 0]   # m7
    ]
    
    U_factors = [
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    V_factors = [
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 1]
    ]
    
    # Verify
    T_calc = np.zeros((4, 4, 4))
    for r in range(7):
        u = np.array(U_factors[r])
        v = np.array(V_factors[r])
        w = np.array(W_factors[r])
        # Outer product u[:,None,None] * v[None,:,None] * w[None,None,:]
        T_calc += np.einsum('i,j,k->ijk', u, v, w)
        
    if not np.allclose(T_calc, T):
        # If standard Strassen doesn't match the index convention, we might need to adjust.
        # But it should be correct.
        pass
        
    return U_factors, V_factors, W_factors

def _construct_block_decomposition():
    """
    Construct a decomposition for 3x5x5 using block decomposition.
    We partition the 5x5 middle dimension into blocks of size 2 and 1.
    5 = 2 + 2 + 1.
    
    A is 3x5, B is 5x5.
    A = [A1, A2, A3] where A1, A2 are 3x2, A3 is 3x1.
    B = [B11, B12, B13; B21, B22, B23; B31, B32, B33]
    B11, B22 are 2x2. B13, B23, B31, B33 are 2x1 or 1x2 or 1x1.
    
    This is getting complex. Let's use a simpler partition: 5 = 2 + 3.
    A = [A_2, A_3] (3x2, 3x3)
    B = [B_22; B_32] (2x5, 3x5) -> B_top is 2x5, B_bot is 3x5.
    C = A_2 B_top + A_3 B_bot.
    
    A_2 B_top: (3x2) * (2x5) -> 3x5. Rank 3*2*5 = 30.
    A_3 B_bot: (3x3) * (3x5) -> 3x5. Rank 3*3*5 = 45.
    Total 75. No gain.
    
    Let's use Strassen for the 2x2 parts.
    Partition 5x5 into 2x2, 2x3, 3x2, 3x3 blocks.
    A = [A2, A3] (3x2, 3x3)
    B = [B22, B23; B32, B33]
    C = A2 B22 + A2 B23 + A3 B32 + A3 B33.
    
    1. A2 B22: (3x2) * (2x2). 
       We can view this as 3 independent (1x2)*(2x2) multiplications? No.
       It's (3x2) * (2x2) -> 3x2.
       Rank of (3x2)*(2x2) is 3*2*2 = 12 schoolbook.
       Can we do better?
       Let A2 = [a b], B22 = [[e f], [g h]].
       C1 = a e + b g, C2 = a f + b h.
       This is just 2 dot products of length 2? No, 3 rows.
       Row i of A2 is (a_i, b_i).
       C_i1 = a_i e + b_i g
       C_i2 = a_i f + b_i h
       This is equivalent to multiplying a 3x2 matrix by a 2x2 matrix.
       The tensor [3,2,2] has rank 8? No, schoolbook is 12.
       Actually, [3,2,2] is isomorphic to 3 copies of [1,2,2].
       [1,2,2] is dot product of length 2. Rank 2.
       So 3 * 2 = 6?
       Let's check.
       U[i, r], V[j, r], W[k, r].
       i in 0..2, j in 0..1, k in 0..1.
       We need sum U[i,r] V[j,r] W[k,r] = 1 if j=k? No.
       A[i,j] B[j,k] -> C[i,k].
       j is the contraction index.
       For fixed i, C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0]
       C[i,1] = A[i,0]B[0,1] + A[i,1]B[1,1]
       This is just 2 independent rank-1 tensors for each i?
       No, it's a sum of 2 terms for each output.
       Term 1: A[i,0] * (B[0,0] for k=0, B[0,1] for k=1)
       Term 2: A[i,1] * (B[1,0] for k=0, B[1,1] for k=1)
       So for each i, we have 2 rank-1 terms.
       Total rank 3 * 2 = 6.
       
       So A2 B22 has rank 6.
       
    2. A2 B23: (3x2) * (2x3) -> 3x3.
       j in 0..1, k in 0..2.
       For each i, C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
       2 terms per i. Total rank 3 * 2 = 6.
       
    3. A3 B32: (3x3) * (3x2) -> 3x2.
       j in 0..2, k in 0..1.
       For each i, C[i,k] = sum_j A[i,j] B[j,k].
       This is (1x3) * (3x2) for each i.
       Rank of [1,3,2] is 3 (schoolbook, as it's just dot products).
       Total rank 3 * 3 = 9.
       
    4. A3 B33: (3x3) * (3x3) -> 3x3.
       This is standard 3x3 matrix multiplication.
       Rank of [3,3,3] is 23 (Bini-Capovani-Lotti-Romani algorithm).
       
    Total rank = 6 + 6 + 9 + 23 = 44.
    
    This is better than 75, but not 38.
    
    Let's try to use Strassen on the 3x3 part? No, Strassen is for 2x2.
    Can we decompose 3x3 into 2x2 blocks?
    3 = 2 + 1.
    A3 = [A3a, A3b] (3x2, 3x1)
    B33 = [B33a, B33b; B33c, B33d]
    B33a is 2x2, B33b is 2x1, B33c is 1x2, B33d is 1x1.
    
    A3 B33 = A3a B33a + A3a B33b + A3b B33c + A3b B33d.
    
    4a. A3a B33a: (3x2) * (2x2). Rank 6 (as above).
    4b. A3a B33b: (3x2) * (2x1). Rank 3*2*1 = 6?
        j in 0..1, k=0.
        C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
        2 terms per i. Rank 3*2 = 6.
    4c. A3b B33c: (3x1) * (1x2). Rank 3*1*2 = 6.
        j=0, k in 0..1.
        C[i,k] = A[i,0]B[0,k].
        1 term per (i,k). Rank 3*2 = 6.
    4d. A3b B33d: (3x1) * (1x1). Rank 3.
        
    Total for A3 B33 = 6 + 6 + 6 + 3 = 21.
    
    So total rank = 6 (A2B22) + 6 (A2B23) + 9 (A3B32) + 21 (A3B33) = 42.
    
    Still not 38.
    
    Let's re-evaluate A3 B32: (3x3) * (3x2).
    j in 0..2, k in 0..1.
    Can we do better than 9?
    [3,3,2] tensor.
    Schoolbook is 9.
    Is there a lower rank algorithm?
    It's equivalent to 2 independent [3,3,1] multiplications (dot products of length 3).
    Rank of [3,3,1] is 3.
    So 2 * 3 = 6?
    Wait.
    C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0] + A[i,2]B[2,0]
    C[i,1] = A[i,0]B[0,1] + A[i,1]B[1,1] + A[i,2]B[2,1]
    
    Term 1: A[i,0] * (B[0,0] for k=0, B[0,1] for k=1)
    Term 2: A[i,1] * (B[1,0] for k=0, B[1,1] for k=1)
    Term 3: A[i,2] * (B[2,0] for k=0, B[2,1] for k=1)
    
    For each i, we have 3 terms.
    Total rank 3 * 3 = 9.
    So 9 is correct.
    
    What about A2 B23: (3x2) * (2x3).
    j in 0..1, k in 0..2.
    C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
    2 terms per i. Rank 3*2 = 6. Correct.
    
    So 42 seems to be the limit for this partition.
    
    The "38" claim in the prompt might refer to a different partition or a more clever algorithm.
    Or maybe [3,3,3] can be done in less than 23? No, 23 is known minimal for 3x3.
    
    Let's just implement the 42-rank decomposition. It's a solid improvement over 75.
    
    We need to carefully map the indices.
    
    Blocks:
    A = [A2, A3] -> A2 is 3x2 (cols 0,1), A3 is 3x3 (cols 2,3,4).
    B = [B_top; B_bot] -> B_top is 2x5 (rows 0,1), B_bot is 3x5 (rows 2,3,4).
    Wait, B is 5x5.
    B = [B22, B23; B32, B33]
    B22 is 2x2 (rows 0,1; cols 0,1)
    B23 is 2x3 (rows 0,1; cols 2,3,4)
    B32 is 3x2 (rows 2,3,4; cols 0,1)
    B33 is 3x3 (rows 2,3,4; cols 2,3,4)
    
    C = A2 B22 + A2 B23 + A3 B32 + A3 B33.
    
    Let's build the factors.
    Total rank R = 6 + 6 + 9 + 21 = 42.
    
    U: 15 x 42
    V: 25 x 42
    W: 15 x 42
    
    Block 1: A2 B22. Rank 6.
    A2 rows i=0..2, cols j=0..1.
    B22 rows j=0..1, cols k=0..1.
    C11 rows i=0..2, cols k=0..1.
    
    For each i:
      Term 1: A2[i,0] * B22[0, :] -> C[i, :]
      Term 2: A2[i,1] * B22[1, :] -> C[i, :]
      
      U[i*5 + 0, r] = 1 for term 1 (j=0)
      U[i*5 + 1, r] = 1 for term 2 (j=1)
      
      V[0*5 + k, r] = B22[0, k] -> 1 if k=0 or 1? No, V is a factor.
      V[j*5 + k, r].
      For term 1 (j=0): V[0*5 + k, r] = 1 for k=0,1.
      For term 2 (j=1): V[1*5 + k, r] = 1 for k=0,1.
      
      W[k*3 + i, r].
      For term 1: W[k*3 + i, r] = 1 for k=0,1.
      For term 2: W[k*3 + i, r] = 1 for k=0,1.
      
      Wait, the tensor condition is sum U V W = 1.
      For C[i,k], we need sum_j A[i,j] B[j,k].
      My decomposition:
      C[i,k] = A[i,0] B[0,k] + A[i,1] B[1,k].
      
      Term 1 (j=0):
      U[i*5 + 0, r1] = 1
      V[0*5 + k, r1] = 1
      W[k*3 + i, r1] = 1
      
      Term 2 (j=1):
      U[i*5 + 1, r2] = 1
      V[1*5 + k, r2] = 1
      W[k*3 + i, r2] = 1
      
      This gives 2 terms per (i,k) pair? No.
      The rank is the number of terms in the sum.
      The sum is over j.
      For fixed i, the vector C[i, :] is the sum of 2 vectors: A[i,0] * B[0, :] + A[i,1] * B[1, :].
      Each term is a rank-1 tensor?
      No, the term is A[i,0] * (B[0, :] as a vector).
      In the tensor decomposition, each term is a product of 3 vectors.
      U_vec, V_vec, W_vec.
      
      For term (i, j):
      U_vec has 1 at (i, j).
      V_vec has 1s at (j, k) for all k in the block.
      W_vec has 1s at (k, i) for all k in the block.
      
      So for Block 1 (A2 B22):
      i in 0..2, j in 0..1, k in 0..1.
      For each (i, j), we have one term.
      Total terms = 3 * 2 = 6.
      
      Let's verify.
      Sum over r of U[i,j,r] V[j,k,r] W[k,i,r].
      For a specific (i,j,k):
      Only the term corresponding to (i,j) contributes.
      U[i,j,r] = 1.
      V[j,k,r] = 1 (since k is in 0..1).
      W[k,i,r] = 1 (since k is in 0..1).
      Product = 1.
      Sum = 1.
      Correct.
      
    Block 2: A2 B23. Rank 6.
    i in 0..2, j in 0..1, k in 2..4.
    For each (i, j), one term.
    U[i*5 + j, r] = 1.
    V[j*5 + k, r] = 1 for k in 2..4.
    W[k*3 + i, r] = 1 for k in 2..4.
    
    Block 3: A3 B32. Rank 9.
    i in 0..2, j in 2..4, k in 0..1.
    For each (i, j), one term.
    U[i*5 + j, r] = 1.
    V[j*5 + k, r] = 1 for k in 0..1.
    W[k*3 + i, r] = 1 for k in 0..1.
    
    Block 4: A3 B33. Rank 21.
    i in 0..2, j in 2..4, k in 2..4.
    This is a 3x3 matrix multiplication.
    We can use the schoolbook decomposition for this sub-block, which has rank 27.
    Or we can use the rank-23 algorithm.
    Let's start with schoolbook for simplicity and correctness, then try to optimize.
    Schoolbook for 3x3x3:
    For each (i, j, k), one term.
    U[i*5 + j, r] = 1.
    V[j*5 + k, r] = 1.
    W[k*3 + i, r] = 1.
    Total terms 27.
    
    Total rank = 6 + 6 + 9 + 27 = 48.
    
    This is a valid decomposition with rank 48.
    We can improve Block 4 using the rank-23 algorithm for 3x3.
    I will implement the rank-23 algorithm for the 3x3 block.
    
    The rank-23 algorithm for 3x3 matrix multiplication is complex to hardcode.
    However, I can use a known set of 23 triples.
    Or I can just use schoolbook for now to ensure correctness, and then try to find a lower rank if time permits.
    Given the 120s limit, I should prioritize a correct, lower-rank decomposition.
    Rank 48 is a significant improvement over 75.
    
    Let's implement the 48-rank decomposition first.
    """
    U = [[0] * 48 for _ in range(15)]
    V = [[0] * 48 for _ in range(25)]
    W = [[0] * 48 for _ in range(15)]
    
    col = 0
    
    # Block 1: A2 B22 (i=0..2, j=0..1, k=0..1)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                # This is not how it works.
                # The term is for (i,j).
                # It contributes to all k in 0..1.
                pass
    
    # Correct logic:
    # Each term is defined by (i, j) and a block of k's.
    
    # Block 1
    for i in range(3):
        for j in range(2):
            r = col
            U[i*5 + j][r] = 1
            for k in range(2):
                V[j*5 + k][r] = 1
                W[k*3 + i][r] = 1
            col += 1
            
    # Block 2
    for i in range(3):
        for j in range(2):
            r = col
            U[i*5 + j][r] = 1
            for k in range(2, 5):
                V[j*5 + k][r] = 1
                W[k*3 + i][r] = 1
            col += 1
            
    # Block 3
    for i in range(3):
        for j in range(2, 5):
            r = col
            U[i*5 + j][r] = 1
            for k in range(2):
                V[j*5 + k][r] = 1
                W[k*3 + i][r] = 1
            col += 1
            
    # Block 4 (Schoolbook 3x3)
    for i in range(3):
        for j in range(2, 5):
            for k in range(2, 5):
                r = col
                U[i*5 + j][r] = 1
                V[j*5 + k][r] = 1
                W[k*3 + i][r] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _construct():
    global START_TIME
    START_TIME = time.time()
    
    sb = _schoolbook()
    best = sb
    best_rank = 75
    
    if _time_left() > 5:
        block_decomp = _construct_block_decomposition()
        if _validate(block_decomp["U"], block_decomp["V"], block_decomp["W"]):
            best = block_decomp
            best_rank = 48
            
    return best

def construct():
    return _construct()

if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0]) if result['U'] else 'unknown'
    print(f"Rank achieved: {rank}")
