import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = delta_{i,k} for A[i,j]*B[j,k] = C[i,k]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Tensor entry T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i and j==j' and i==i' ... 
    # Actually the multiplication tensor: T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}... 
    # No. The tensor for matrix mult: T[i,j,k] for the 3-index version is different.
    # Let me use the standard form: The multiplication tensor M_{n,m,p} has entries
    # T[(i,j),(j',k),(k',i')] = 1 if j==j' and i==i' and k==k', else 0.
    # Wait no. For C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
    # So the tensor is T[i,j,k] = A[i,j]*B[j,k]*C[i,k] but as a trilinear form:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k] = 0 for the bilinear form...
    # Actually the tensor decomposition means: for each rank-1 term r,
    # U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] should sum to give delta.
    # The condition is: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # where T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i with the SAME j in a and b,
    # and the SAME i in a and c... 
    # Let me just check: T[(i,j),(j,k),(k,i)] = 1 (using same indices), 0 otherwise.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute sum_r U[a,r] * V[b,r] * W[c,r] for all a,b,c
    # This is a tensor of shape (n*m, m*p, p*n)
    # Use einsum: sum_r U[a,r] V[b,r] W[c,r]
    result = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target[a, b, c] = 1.0
    
    if not np.allclose(result, target, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_2x2(n=2, m=2, p=2):
    """Strassen decomposition for 2x2 x 2x2 matrix multiplication, rank 7."""
    # Standard Strassen: 7 products
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[2][0] = -1  # C11+=M1, C22-=M1... wait
    # C11 = M1+M4-M5+M7, C22 = M1-M2+M3+M6
    # So W for M1: C11 gets +1, C22 gets +1
    W[0][0] = 1; W[3][0] = 1  # C[0,0]=C11, C[1,1]=C22
    # Wait, W index: k*n+i. For 2x2: W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    # C11 is W[0], C12 is W[1], C21 is W[2], C22 is W[3]
    
    # Let me redo:
    # M1: U: a11(0),a22(3) both +1; V: b11(0),b22(3) both +1; W: C11(0)+1, C22(3)+1
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    
    # M2 = (a21+a22)b11
    # U: a21(2)+1, a22(3)+1; V: b11(0)+1; W: C21(2)+1, C22(3)-1
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    
    # M3 = a11(b12-b22)
    # U: a11(0)+1; V: b12(1)+1, b22(3)-1; W: C12(1)+1, C22(3)+1
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1
    
    # M4 = a22(b21-b11)
    # U: a22(3)+1; V: b21(2)+1, b11(0)-1; W: C11(0)+1, C21(2)+1
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    
    # M5 = (a11+a12)b22
    # U: a11(0)+1, a12(1)+1; V: b22(3)+1; W: C11(0)-1, C12(1)+1
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    
    # M6 = (a21-a11)(b11+b12)
    # U: a21(2)+1, a11(0)-1; V: b11(0)+1, b12(1)+1; W: C22(3)+1
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    
    # M7 = (a12-a22)(b21+b22)
    # U: a12(1)+1, a22(3)-1; V: b21(2)+1, b22(3)+1; W: C11(0)+1
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1
    
    return U, V, W

def _rank28_decomposition():
    """
    Rank-28 decomposition for 2x3 x 3x5 matrix multiplication.
    
    Strategy: Split the 5 columns of B into two parts: 2 columns and 3 columns.
    - Part 1: 2x3 x 3x2 -> 2x2 output. Use a rank-7 decomposition.
    - Part 2: 2x3 x 3x3 -> 2x3 output. Use schoolbook rank-9.
    Total: 7 + 9 = 16? 
    
    Actually, let me think about this more carefully. The issue is that when we split,
    we're not just doing independent subproblems - we need to be careful about how
    the factors combine.
    
    Alternative approach that gives rank 28:
    Use the fact that 2x3 x 3x5 can be decomposed using Strassen-like techniques
    on the 3x3 blocks.
    
    Let me try a different split:
    - Split the middle dimension 3 into 2+1
    - This gives terms that need to be combined
    
    Actually, the simplest verified approach: use the schoolbook for the 2x3x3 part (rank 9)
    and Strassen for the 2x3x2 part (rank 7), total 16.
    But I need to verify this works.
    
    Let me try yet another approach: just verify if rank 28 works with a specific construction.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 6 x 28
    V = [[0]*rank for _ in range(m*p)]  # 15 x 28
    W = [[0]*rank for _ in range(p*n)]  # 10 x 28
    
    # I'll construct this by combining:
    # 1) Strassen for the first 2x2 block of the output (columns 0-1 of B)
    # 2) Schoolbook for the remaining 2x3 block (columns 2-4 of B)
    
    # For part 1: 2x3 x 3x2
    # A is 2x3, B1 is 3x2 (first 2 columns of B)
    # We need rank-7 factors for this
    # The 2x3 x 3x2 tensor: T[(i,j),(j,k),(k,i)] = 1
    # where i in {0,1}, j in {0,1,2}, k in {0,1}
    # This is equivalent to 2x2x2 with an extra j index.
    # We can embed Strassen here.
    
    # Actually, let me just use a known rank-7 for 2x3x2:
    # Split j into j1 (0,1) and j2 (2)
    # For j in {0,1}: standard 2x2x2 Strassen (rank 7) but with only 2 of the 3 j-indices
    # For j=2: additional terms
    
    # This is getting complicated. Let me just use the schoolbook for everything
    # and verify, then try to optimize.
    
    # For now, return a verified rank-30 schoolbook and see if we can improve.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try rank-28 decomposition
    result = _try_rank_28(n, m, p)
    if result is not None:
        if _verify(result['U'], result['V'], result['W'], n, m, p):
            return result
    
    # Fallback to schoolbook rank 30
    return _schoolbook(n, m, p)

def _try_rank_28(n, m, p):
    """Attempt rank-28 decomposition using block structure."""
    # Split B's 5 columns into 2+3
    # Part 1: 2x3 x 3x2 (output 2x2) - use Strassen rank 7
    # Part 2: 2x3 x 3x3 (output 2x3) - use schoolbook rank 9
    # Total: 16
    
    # For 2x3 x 3x2: 
    # A is 2x3, B is 3x2, C is 2x2
    # T[(i,j),(j,k),(k,i)] = 1 for i in 0,1; j in 0,1,2; k in 0,1
    # 
    # We can write this as: for each j, we have a 2x1 x 1x2 slice
    # Sum over j: A[:,j] * B[j,:] = C
    # 
    # Rank-7 construction: Use Strassen on the "effective" 2x2 by grouping j indices.
    # Let's define:
    # A1 = A[:,0], A2 = A[:,1], A3 = A[:,2] (each 2x1)
    # B1 = B[0,:], B2 = B[1,:], B3 = B[2,:] (each 1x2)
    # C = A1*B1 + A2*B2 + A3*B3
    #
    # This is like multiplying 2x1 by 1x2 three times and summing.
    # The tensor rank of sum of 3 outer products of 2x1 and 1x2...
    # 
    # Actually, this is the same as the 2x3x2 matrix multiplication tensor.
    # Known rank: 7 (same as 2x2x2 since the extra dimension can be handled).
    #
    # Let me construct it explicitly using Strassen-like formulas.
    # 
    # Define 7 products:
    # M1 = (A1+A2)(B1+B2)  [element-wise: (A1+A2) is 2x1, (B1+B2) is 1x2]
    # M2 = (A2+A3)B1
    # M3 = A1(B2-B3)
    # M4 = A3(B1-B2)  -- wait, this doesn't quite work
    # 
    # Let me use a different approach. The 2x3x2 tensor can be seen as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Use the following rank-7 decomposition (adapted from Strassen):
    # Let a_i = A[i,0], b_i = A[i,1], c_i = A[i,2] for i=0,1
    # Let p_j = B[j,0], q_j = B[j,1] for j=0,1,2
    # 
    # C[0,0] = a0*p0 + b0*p1 + c0*p2
    # C[0,1] = a0*q0 + b0*q1 + c0*q2
    # C[1,0] = a1*p0 + b1*p1 + c1*p2
    # C[1,1] = a1*q0 + b1*q1 + c1*q2
    # 
    # 7 products:
    # M1 = (a0+a1)(p0+p1)  -- no, these are scalars times scalars
    # Actually each "product" is an outer product of a 2-vector and a 2-vector.
    # 
    # Let me define:
    # u1 = (a0+a1, a0+a1) -- no, that's not right either.
    # 
    # I think the cleanest way is to just verify empirically.
    # Let me construct a candidate and verify it.
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Use schoolbook for the 2x3x3 part (columns 2,3,4 of B)
    # And a rank-7 construction for the 2x3x2 part (columns 0,1 of B)
    
    # Part 2: Schoolbook for columns 2,3,4 (3 columns)
    # A is 2x3, B_part is 3x3, C_part is 2x3
    # 9 terms
    col = 7  # Start after the 7 Strassen terms
    for i in range(n):
        for j in range(m):
            for k in range(2, p):  # k = 2,3,4
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Part 1: Rank-7 for columns 0,1
    # 2x3 x 3x2 tensor
    # A: 2x3 (indices i*m+j = 0,1,2,3,4,5 for i=0,1; j=0,1,2)
    # B: 3x2 (indices j*p+k = 0,1,2,3,4,5 for j=0,1,2; k=0,1)
    # C: 2x2 (indices k*n+i = 0,1,2,3 for k=0,1; i=0,1)
    # 
    # 7 rank-1 terms:
    # Term 0: U: A[0,0]=1, A[1,2]=1; V: B[0,0]=1, B[1,1]=1; W: C[0,0]=1, C[1,1]=1
    # This is getting too speculative. Let me just return None and use schoolbook.
    # I need a verified construction.
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Try to build a verified rank-28 decomposition
    result = _build_rank_28()
    if result is not None:
        if _verify(result['U'], result['V'], result['W'], n, m, p, tol=1e-9):
            return result

    # Fallback: schoolbook rank 30
    return _schoolbook(n, m, p)

def _build_rank_28():
    """
    Build a rank-28 decomposition for 2x3 x 3x5.
    
    Strategy: Split B (3x5) into B1 (3x2) and B2 (3x3).
    - C1 = A*B1: 2x3 x 3x2 -> 2x2, rank 7 (Strassen-like)
    - C2 = A*B2: 2x3 x 3x3 -> 2x3, rank 9 (schoolbook)
    - Total rank: 7 + 9 = 16? 
    
    Wait, 7+9=16, not 28. Let me reconsider.
    
    Actually, I think the issue is that the rank of 2x3x2 is not 7.
    Let me compute it properly.
    
    The tensor M_{2,3,2} has entries T[(i,j),(j,k),(k,i)] = 1.
    Number of entries: 6 * 6 * 4 = 144, but the tensor is sparse.
    Non-zero entries: for each (i,j,k) with i in {0,1}, j in {0,1,2}, k in {0,1}:
    T[i*m+j, j*p+k, k*n+i] = 1
    That's 2*3*2 = 12 non-zero entries.
    
    The rank is at least the size of the largest "slice" which is max(n*m, m*p, p*n) = max(6,6,4) = 6.
    And at most n*m*p = 12.
    
    Known result: rank(M_{2,3,2}) = 7. Let me verify by construction.
    
    Actually, I recall that rank(M_{2,3,2}) = 7 is NOT correct. Let me think...
    M_{2,2,2} has rank 7 (Strassen). M_{2,3,2} should have rank >= 7.
    Is it exactly 7? I'm not sure. Let me try to construct it.
    
    For now, let me just use schoolbook for both parts: 12 + 9 = 21? No, 2x3x2 schoolbook is 12, 2x3x3 schoolbook is 9. Total 21.
    But 21 < 28, so if I can verify it, that's better.
    
    Let me just use full schoolbook (rank 30) as the safe fallback and try to improve.
    """
    return None

def _verify(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    result = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(result, target, atol=tol)

def _schoolbook(n, m, p):
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
# EVOLVE-BLOCK-END
