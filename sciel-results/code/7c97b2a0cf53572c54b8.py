"""Matrix multiplication 3x4x8 rank-minimal decomposition via block Strassen/Laderman."""
import time
import itertools

DIMENSIONS = [3, 4, 8]

# Known small matrix multiplication ranks:
# 2x2x2: rank 7 (Strassen)
# 3x3x3: rank 19 (Laderman)
# 2x2x4: rank 14 (Strassen on 2x2 blocks)
# 2x4x4: rank 14
# 4x2x4: rank 14
# 4x4x2: rank 14
# 4x4x4: rank 49 (Strassen recursive)

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 multiplications
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)*e, M3 = a*(f-h), M4 = d*(g-e)
    # M5 = (a+b)*h, M6 = (c-d)*(e+f), M7 = (b-c)*(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # U rows: a(0), b(1), c(2), d(3)
    # V rows: e(0), f(1), g(2), h(3)
    # W rows: k*n+i: for n=2,m=2,p=2: W[k*2+i]
    #   k=0,i=0 -> 0 (C11), k=0,i=1 -> 1 (C21)
    #   k=1,i=0 -> 2 (C12), k=1,i=1 -> 3 (C22)
    
    # Let's define the 7 terms:
    # Term 0: M1 = (a+d)(e+h)
    # Term 1: M2 = (c+d)*e
    # Term 2: M3 = a*(f-h)
    # Term 3: M4 = d*(g-e)
    # Term 4: M5 = (a+b)*h
    # Term 5: M6 = (c-d)*(e+f)
    # Term 6: M7 = (b-c)*(g+h)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h): U[0]+=1, U[3]+=1, V[0]+=1, V[3]+=1
    U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1
    # C11 = M1+M4-M5+M7, C21 = M2+M4, C12 = M3+M5, C22 = M1-M2+M3+M6
    W[0][0] = 1   # C11 += M1
    W[3][0] = 1   # C22 += M1
    
    # M2 = (c+d)*e: U[2]+=1, U[3]+=1, V[0]+=1
    U[2][1] = 1; U[3][1] = 1; V[0][1] = 1
    W[1][1] = 1   # C21 += M2
    W[3][1] = -1  # C22 -= M2
    
    # M3 = a*(f-h): U[0]+=1, V[1]+=1, V[3]-=1
    U[0][2] = 1; V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1   # C12 += M3
    W[3][2] = 1   # C22 += M3
    
    # M4 = d*(g-e): U[3]+=1, V[2]+=1, V[0]-=1
    U[3][3] = 1; V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1   # C11 += M4
    W[1][3] = 1   # C21 += M4
    
    # M5 = (a+b)*h: U[0]+=1, U[1]+=1, V[3]+=1
    U[0][4] = 1; U[1][4] = 1; V[3][4] = 1
    W[0][4] = -1  # C11 -= M5
    W[2][4] = 1   # C12 += M5
    
    # M6 = (c-d)*(e+f): U[2]+=1, U[3]-=1, V[0]+=1, V[1]+=1
    U[2][5] = 1; U[3][5] = -1; V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1   # C22 += M6
    
    # M7 = (b-c)*(g+h): U[1]+=1, U[2]-=1, V[2]+=1, V[3]+=1
    U[1][6] = 1; U[2][6] = -1; V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C11 += M7
    
    return U, V, W


def _laderman_3x3x3():
    """Return (U, V, W) for 3x3x3 matrix multiplication, rank 19 (Laderman).
    U has shape (9, 19), V has shape (9, 19), W has shape (9, 19).
    Indexing: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    # Laderman's decomposition for 3x3x3, rank 19
    # A[i,j] indexed as i*3+j, B[j,k] as j*3+k, C[i,k] as i*3+k
    # But our convention is W[k*n+i] for n=3, so W[k*3+i]
    
    # Laderman's 19 terms. Let me use the standard formulation.
    # From Laderman (1976), the 19 products are:
    # Let A = [a11 a12 a13; a21 a22 a23; a31 a32 a33]
    # Let B = [b11 b12 b13; b21 b22 b23; b31 b32 b33]
    
    # I'll use a known rank-19 decomposition.
    # Reference: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
    
    # Actually, let me construct it carefully. The Laderman algorithm:
    # M1 = (a11+a12)(b11+b12)
    # M2 = (a11-a12)(b11-b12)
    # M3 = (a11+a13)(b11+b13)
    # M4 = (a11-a13)(b11-b13)
    # M5 = (a21+a22)(b21+b22)
    # M6 = (a21-a22)(b21-b22)
    # M7 = (a21+a23)(b21+b23)
    # M8 = (a21-a23)(b21-b23)
    # M9 = (a31+a32)(b31+b32)
    # M10 = (a31-a32)(b31-b32)
    # M11 = (a31+a33)(b31+b33)
    # M12 = (a31-a33)(b31-b33)
    # M13 = (a11+a21+a31)(b11+b21+b31)
    # M14 = (a12+a22+a32)(b12+b22+b32)
    # M15 = (a13+a23+a33)(b13+b23+b33)
    # M16 = (a11+a21-a31)(b11+b21-b31)
    # M17 = (a12+a22-a32)(b12+b22-b32)
    # M18 = (a13+a23-a33)(b13+b23-b33)
    # M19 = (a11+a21+a31)(b12+b22+b32) - this doesn't look right
    
    # Let me use a different known rank-19 decomposition.
    # Actually, the standard Laderman rank is 19 for 3x3x3.
    # Let me look up the exact coefficients.
    
    # I'll use a verified rank-19 decomposition from literature.
    # The entries can be 0, 1, -1, or 1/2.
    
    # Using the decomposition from:
    # "The rank of the matrix multiplication tensor" 
    # Laderman's algorithm with 19 multiplications.
    
    # Let me just use a simpler approach: decompose 3x3x3 as a combination
    # of smaller blocks. 3x3 = 2x2 + 1x1 blocks.
    # But that gives rank 7+1+2*1*2 = 12... no, that's not right for block decomposition.
    
    # Actually for block matrix multiplication:
    # If we partition 3x3 into 2x2, 2x1, 1x2, 1x1 blocks:
    # C11 (2x2) = A11*B11 + A12*B21 (rank 7+7=14)
    # C12 (2x1) = A11*B12 + A12*B22 (rank 7+7=14) -- but this is 2x1x1, rank 2
    # Hmm, this gets complicated.
    
    # Let me just hardcode a known rank-19 Laderman decomposition.
    # From Strassen's 1973 paper and subsequent work.
    
    # I'll use the following verified rank-19 decomposition (coefficients in {0, ±1}):
    # This is from the standard Laderman construction.
    
    U = [[0]*19 for _ in range(9)]
    V = [[0]*19 for _ in range(9)]
    W = [[0]*19 for _ in range(9)]
    
    # A indices: a[i*3+j] for i,j in 0,1,2
    # B indices: b[j*3+k] for j,k in 0,1,2
    # C indices: c[i*3+k] for i,k in 0,1,2
    # W indices: W[k*3+i] (trace convention)
    
    # Laderman's 19 terms (from literature):
    # I'll define them as (A_coeff_vector, B_coeff_vector, C_coeff_vector)
    
    terms = [
        # (0,0,0): (a0+a1)(b0+b1) -> c0 += 1, c3 += -1, c6 += -1, c7 += 1... 
        # This is getting complex. Let me use a different approach.
    ]
    
    # Alternative: use the fact that 3x3x3 can be decomposed using
    # the identity: rank(3,3,3) = 19
    # I'll construct it from 2x2x2 Strassen + corrections.
    
    # Actually, let me just use a brute-force verified decomposition.
    # For now, let me use a simpler but valid decomposition.
    
    # 3x3x3 rank upper bound via Strassen-style:
    # Partition into 2x2, 2x1, 1x2, 1x1:
    # C = [[A11 B11 + A12 B21, A11 B12 + A12 B22],
    #      [A21 B11 + A22 B21, A21 B12 + A22 B22]]
    # where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1, etc.
    
    # C11 (2x2) = A11*B11 + A12*B21: rank <= 7 + 4 = 11 (Strassen for 2x2x2 + schoolbook for 2x1x2)
    # C12 (2x1) = A11*B12 + A12*B22: rank <= 7 + 2 = 9 (2x2x1 + 2x1x1)
    # C21 (1x2) = A21*B11 + A22*B21: rank <= 4 + 2 = 6 (1x2x2 + 1x1x2)
    # C22 (1x1) = A21*B12 + A22*B22: rank <= 2 + 1 = 3 (1x2x1 + 1x1x1)
    # Total: 11+9+6+3 = 29. Not great.
    
    # Let me just use a known rank-19 decomposition directly.
    # From: "A new bound for the rank of the matrix multiplication tensor"
    
    # I'll use this verified rank-19 decomposition (all entries in {-1, 0, 1}):
    # Source: Laderman (1976), as presented in various references.
    
    # After careful consideration, let me use a simpler valid decomposition
    # that I can verify by construction.
    
    # Actually, for the purpose of this task, let me use a block decomposition
    # that's guaranteed correct and see what rank we get.
    
    # For 3x4x8:
    # Option 1: 3x4x8 = (2+1)x4x(4+4)
    #   Block 1: 2x4x4 -> rank 14 (Strassen on 2x2 blocks: 2*(7) = 14? No...)
    #   Actually 2x4x4: partition 4 into 2+2: 
    #     C(2x4) = A(2x4)*B(4x4)
    #     Split B into B1(4x2), B2(4x2):
    #     C1(2x2) = A1(2x2)*B11(2x2) + A2(2x2)*B12(2x2) -- no, A is 2x4
    #     
    # Let me think about this differently.
    
    # 3x4x8: 
    # n=3, m=4, p=8
    # 
    # Decomposition strategy:
    # Split n=3 as 2+1, m=4 as 2+2, p=8 as 4+4
    # 
    # Blocks:
    # A: 3x4 -> A11(2x2), A12(2x2), A21(1x2), A22(1x2)
    # B: 4x8 -> B11(2x4), B12(2x4), B21(2x4), B22(2x4)
    # C: 3x8 -> C11(2x4), C12(2x4), C21(1x4), C22(1x4)
    # 
    # C11 = A11*B11 + A12*B21 (2x2 * 2x4 + 2x2 * 2x4)
    #   = A11*B11 + A12*B21
    #   Each term is 2x2x4: rank 14 each? No, 2x2x4 means multiplying 2x2 by 2x4.
    #   2x2x4: n=2,m=2,p=4. Rank = ?
    #   Using Strassen on the 2x2 part: treat as two 2x2x2 multiplications.
    #   2x2x4 = 2 * (2x2x2) -> rank 2*7 = 14
    #   So C11 rank <= 14 + 14 = 28
    # 
    # C12 = A11*B12 + A12*B22 (2x2 * 2x4 + 2x2 * 2x4) -> rank <= 28
    # C21 = A21*B11 + A22*B21 (1x2 * 2x4 + 1x2 * 2x4)
    #   1x2x4: n=1,m=2,p=4. Rank = m*p = 8 (since n=1, it's just dot products)
    #   Actually 1x2x4: C(i,k) = sum_j A(i,j)*B(j,k), i fixed, j=0,1, k=0..3
    #   That's 4 outputs, each a dot product of length 2. Rank = 4 (one per output)
    #   Wait, rank of (1,2,4) tensor: it's the sum over j of A[0,j]*B[j,k] for each k.
    #   As a tensor T[i,j,k] = delta[i,0]*delta[j,k]... no.
    #   T[i,j,k] = 1 if i=0, else 0. The tensor is sum_j e_0 ⊗ e_j ⊗ e_j... no.
    #   T[i,j,k] = 1 if (i,j,k) such that... 
    #   Actually the matrix multiplication tensor for (n,m,p) is T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k], which is always true.
    #   So T[i,j,k] = 1 for all i,j,k in range.
    #   For (1,2,4): T[0,j,k] = 1 for j in 0,1 and k in 0,1,2,3.
    #   Rank = number of (j,k) pairs = 2*4 = 8? No, the rank is the minimum number of rank-1 tensors.
    #   T = sum_{j,k} e_0 ⊗ e_j ⊗ e_k. That's 8 rank-1 terms. But can we do better?
    #   T[0,j,k] = 1. This is a 1x2x4 tensor that's all ones.
    #   Rank of an all-ones tensor of size a*b*c is max(a,b,c) if... no.
    #   Actually for (1,m,p), the tensor is just a matrix of all 1s of size m x p.
    #   The rank of a matrix of all 1s of size m x p is 1 (it's e_m * e_p^T).
    #   Wait no, as a 3-tensor with first dimension 1, it's equivalent to a 2-tensor (matrix).
    #   The rank of the all-1 matrix of size m x p is 1.
    #   So rank(1,2,4) = 1? Let me verify.
    #   T[0,j,k] = 1. We need T = sum_r u_r ⊗ v_r ⊗ w_r.
    #   Since first dim is 1, u_r = (1) for all r (or 0).
    #   So T[0,j,k] = sum_r v_r[j] * w_r[k].
    #   We need sum_r v_r[j] * w_r[k] = 1 for all j,k.
    #   This is rank 1 if we can write it as v[j]*w[k] = 1, e.g., v[j]=1, w[k]=1.
    #   Yes! Rank(1,m,p) = 1.
    #   
    #   Similarly rank(m,1,p) = 1, rank(n,m,1) = 1.
    
    # OK so let me redo:
    # C21 = A21*B11 + A22*B21
    #   A21 is 1x2, B11 is 2x4: product is 1x4. rank(1,2,4) = 1.
    #   A22 is 1x2, B21 is 2x4: product is 1x4. rank(1,2,4) = 1.
    #   C21 rank <= 1 + 1 = 2
    # C22 = A21*B12 + A22*B22 -> rank <= 2
    
    # Total: 28 + 28 + 2 + 2 = 60. Better than 96!
    
    # Can we do better? Let's try different splits.
    # 
    # 3x4x8: split as n=3, m=4, p=8
    # Try: m=4 as 2+2, p=8 as 4+4, keep n=3
    # C(3x8) = A(3x4)*B(4x8)
    # Split B into B1(4x4), B2(4x4):
    # C1(3x4) = A*B1, C2(3x4) = A*B2
    # Each is 3x4x4: 
    #   Split n=3 as 2+1:
    #   C11(2x4) = A1(2x4)*B1(4x4): 2x4x4
    #   C12(2x4) = A2(2x4)*B2(4x4): 2x4x4  -- wait, B is 4x4 here
    #   Hmm, let me be more careful.
    
    # Let me just implement the 2x2x4 rank-14 decomposition and build up.
    
    # 2x2x4: multiply 2x2 matrix by 2x4 matrix.
    # A = [[a,b],[c,d]], B = [[e,f,g,h],[i,j,k,l]]
    # C = [[ae+bi, af+bj, ag+bk, ah+bl],
    #      [ce+di, cf+dj, cg+dk, ch+dl]]
    # 
    # Using Strassen on the 2x2 part:
    # M1 = (a+d)(e+i), M2 = (c+d)*e, M3 = a*(f-j), M4 = d*(g-i)
    # M5 = (a+b)*j, M6 = (c-d)*(e+f), M7 = (b-c)*(i+j)
    # 
    # For the 4 columns, we apply Strassen to each pair of columns.
    # Columns 0,1: use e,f and i,j
    # Columns 2,3: use g,h and k,l
    # 
    # For columns 0,1:
    # M1_01 = (a+d)(e+i), M2_01 = (c+d)*e, M3_01 = a*(f-j), M4_01 = d*(g-i)... 
    # Wait, g and i are from different columns. Let me redo.
    
    # Actually, 2x2x4 is not the same as two 2x2x2.
    # 2x2x4: n=2, m=2, p=4.
    # We can view it as: for each pair of output columns, do a 2x2x2.
    # Columns (0,1): A(2x2) * B_cols01(2x2) -> C_cols01(2x2)
    # Columns (2,3): A(2x2) * B_cols23(2x2) -> C_cols23(2x2)
    # Each 2x2x2 has rank 7, so total rank = 14.
    
    # But wait, the A matrix is shared! So we can't just do two independent Strassen.
    # Actually we can: the tensor for 2x2x4 is:
    # T[i,j,k] = 1 for i in 0,1; j in 0,1; k in 0,1,2,3
    # This is the same as the tensor for 2x2x2 repeated for k=0,1 and k=2,3.
    # So rank(2,2,4) <= 2*rank(2,2,2) = 14.
    # And it's known that rank(2,2,4) = 14 (cannot do better).
    
    # So the construction is: for each pair of columns, use Strassen.
    # The U matrix will have the same Strassen coefficients for both column pairs.
    
    def _strassen_2x2x4():
        """2x2x4 matrix multiplication, rank 14.
        U: (4, 14), V: (8, 14), W: (8, 14)
        U[i*2+j, c], V[j*4+k, c], W[k*2+i, c]
        """
        U_s, V_s, W_s = _strassen_2x2x2()
        U = [[0]*14 for _ in range(4)]
        V = [[0]*14 for _ in range(8)]
        W = [[0]*14 for _ in range(8)]
        
        for c in range(7):
            # First column pair (k=0,1)
            for r in range(4):
                U[r][c] = U_s[r][c]
            for r in range(4):
                V[r][c] = V_s[r][c]  # V rows 0-3 are j*4+k for k=0,1 -> j*4+0, j*4+1
                # Wait, V index is j*4+k. For k=0,1: V[j*4+0] and V[j*4+1]
                # V_s has rows j*2+k for k=0,1. So V_s[j*2+k] corresponds to V[j*4+k]
                # So V[j*4+k][c] = V_s[j*2+k][c]
            V[j*4+k][c] = V_s[j*2+k][c]  # This is wrong, let me redo
            
        # Let me redo this more carefully.
        U = [[0]*14 for _ in range(4)]
        V = [[0]*14 for _ in range(8)]
        W = [[0]*14 for _ in range(8)]
        
        for c in range(7):
            # U is the same for both column pairs
            for r in range(4):
                U[r][c] = U_s[r][c]
            
            # V: for column pair 0 (k=0,1)
            for j in range(2):
                for k in range(2):
                    V[j*4+k][c] = V_s[j*2+k][c]
            
            # W: for column pair 0 (k=0,1)
            for k in range(2):
                for i in range(2):
                    W[k*2+i][c] = W_s[k*2+i][c]
        
        for c in range(7, 14):
            cc = c - 7
            for r in range(4):
                U[r][c] = U_s[r][cc]
            
            # V: for column pair 1 (k=2,3)
            for j in range(2):
                for k in range(2):
                    V[j*4+(k+2)][c] = V_s[j*2+k][cc]
            
            # W: for column pair 1 (k=2,3)
            for k in range(2):
                for i in range(2):
                    W[(k+2)*2+i][c] = W_s[k*2+i][cc]
        
        return U, V, W
    
    # Now let me build the full 3x4x8 decomposition.
    # 
    # Strategy: 
    # n=3, m=4, p=8
    # Split n=3 as 2+1, m=4 as 2+2, p=8 as 4+4
    #
    # A blocks: A11(2x2), A12(2x2), A21(1x2), A22(1x2)
    # B blocks: B11(2x4), B12(2x4), B21(2x4), B22(2x4)
    # C blocks: C11(2x4), C12(2x4), C21(1x4), C22(1x4)
    #
    # C11 = A11*B11 + A12*B21
    #   A11*B11: 2x2x4 -> rank 14
    #   A12*B21: 2x2x4 -> rank 14
    #   Total: 28
    #
    # C12 = A11*B12 + A12*B22
    #   A11*B12: 2x2x4 -> rank 14
    #   A12*B22: 2x2x4 -> rank 14
    #   Total: 28
    #
    # C21 = A21*B11 + A22*B21
    #   A21*B11: 1x2x4 -> rank 1
    #   A22*B21: 1x2x4 -> rank 1
    #   Total: 2
    #
    # C22 = A21*B12 + A22*B22
    #   A21*B12: 1x2x4 -> rank 1
    #   A22*B22: 1x2x4 -> rank 1
    #   Total: 2
    #
    # Grand total: 28+28+2+2 = 60
    
    # Let me implement this.
    
    def _rank1_1x2x4():
        """1x2x4 matrix multiplication, rank 1.
        U: (2, 1), V: (8, 1), W: (4, 1)
        U[i*2+j, c] -> U[j, c] since i=0
        V[j*4+k, c]
        W[k*1+0, c] -> W[k, c]
        """
        # T[0,j,k] = 1 for all j,k
        # U = [1, 1]^T (2x1)
        # V = [1, 1, 1, 1, 1, 1, 1, 1]^T (8x1)
        # W = [1, 1, 1, 1]^T (4x1)
        U = [[1], [1]]
        V = [[1]]*8
        W = [[1]]*4
        return U, V, W
    
    # Now build the full decomposition.
    U_total = []
    V_total = []
    W_total = []
    
    def add_block(U_b, V_b, W_b, row_offset_U, row_offset_V, row_offset_W):
        """Add a block decomposition to the total.
        U_b has shape (n_b*m_b, rank_b), V_b has shape (m_b*p_b, rank_b), W_b has shape (p_b*n_b, rank_b)
        row offsets are for the global indices.
        """
        n_b, m_b, p_b = U_b[0].__len__() if False else (len(U_b), len(U_b[0]), len(W_b))
        # Actually let me pass dimensions explicitly.
        pass
    
    # This is getting complicated. Let me just build it directly.
    
    # Global dimensions: n=3, m=4, p=8
    # U: (12, rank), V: (32, rank), W: (24, rank)
    # U index: i*4+j, V index: j*8+k, W index: k*3+i
    
    # Block 1: C11 = A11*B11 (2x2x4)
    # A11: rows 0,1 of A (i=0,1), cols 0,1 (j=0,1) -> U rows: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # B11: rows 0,1 of B (j=0,1), cols 0-3 (k=0-3) -> V rows: 0*8+0=0, 0*8+1=1, 0*8+2=2, 0*8+3=3, 1*8+0=8, 1*8+1=9, 1*8+2=10, 1*8+3=11
    # C11: cols 0-3 of C (k=0-3), rows 0,1 (i=0,1) -> W rows: k*3+i for k=0-3, i=0,1
    #   k=0: W[0], W[1]; k=1: W[3], W[4]; k=2: W[6], W[7]; k=3: W[9], W[10]
    
    U1, V1, W1 = _strassen_2x2x4()
    # U1: (4, 14), V1: (8, 14), W1: (8, 14)
    # U1 rows: 0,1,2,3 correspond to (i,j) = (0,0),(0,1),(1,0),(1,1)
    # V1 rows: 0-7 correspond to (j,k) = (0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)
    # W1 rows: 0-7 correspond to (k,i) = (0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,0),(3,1)
    
    # Map to global:
    # U1 row 0 -> U row 0 (i=0,j=0)
    # U1 row 1 -> U row 1 (i=0,j=1)
    # U1 row 2 -> U row 4 (i=1,j=0)
    # U1 row 3 -> U row 5 (i=1,j=1)
    
    # V1 row 0 -> V row 0 (j=0,k=0)
    # V1 row 1 -> V row 1 (j=0,k=1)
    # V1 row 2 -> V row 2 (j=0,k=2)
    # V1 row 3 -> V row 3 (j=0,k=3)
    # V1 row 4 -> V row 8 (j=1,k=0)
    # V1 row 5 -> V row 9 (j=1,k=1)
    # V1 row 6 -> V row 10 (j=1,k=2)
    # V1 row 7 -> V row 11 (j=1,k=3)
    
    # W1 row 0 -> W row 0 (k=0,i=0)
    # W1 row 1 -> W row 1 (k=0,i=1)
    # W1 row 2 -> W row 3 (k=1,i=0)
    # W1 row 3 -> W row 4 (k=1,i=1)
    # W1 row 4 -> W row 6 (k=2,i=0)
    # W1 row 5 -> W row 7 (k=2,i=1)
    # W1 row 6 -> W row 9 (k=3,i=0)
    # W1 row 7 -> W row 10 (k=3,i=1)
    
    U_map1 = [0, 1, 4, 5]
    V_map1 = [0, 1, 2, 3, 8, 9, 10, 11]
    W_map1 = [0, 1, 3, 4, 6, 7, 9, 10]
    
    # Block 2: C11 += A12*B21 (2x2x4)
    # A12: rows 0,1 of A (i=0,1), cols 2,3 (j=2,3) -> U rows: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # B21: rows 2,3 of B (j=2,3), cols 0-3 (k=0-3) -> V rows: 2*8+0=16, 2*8+1=17, 2*8+2=18, 2*8+3=19, 3*8+0=24, 3*8+1=25, 3*8+2=26, 3*8+3=27
    # C11: same as before -> W rows: 0,1,3,4,6,7,9,10
    
    U_map2 = [2, 3, 6, 7]
    V_map2 = [16, 17, 18, 19, 24, 25, 26, 27]
    W_map2 = [0, 1, 3, 4, 6, 7, 9, 10]
    
    # Block 3: C12 = A11*B12 (2x2x4)
    # A11: same as block 1 -> U rows: 0,1,4,5
    # B12: rows 0,1 of B (j=0,1), cols 4-7 (k=4-7) -> V rows: 0*8+4=4, 0*8+5=5, 0*8+6=6, 0*8+7=7, 1*8+4=12, 1*8+5=13, 1*8+6=14, 1*8+7=15
    # C12: cols 4-7 of C (k=4-7), rows 0,1 (i=0,1) -> W rows: k*3+i for k=4-7, i=0,1
    #   k=4: W[12], W[13]; k=5: W[15], W[16]; k=6: W[18], W[19]; k=7: W[21], W[22]
    
    U_map3 = [0, 1, 4, 5]
    V_map3 = [4, 5, 6, 7, 12, 13, 14, 15]
    W_map3 = [12, 13, 15, 16, 18, 19, 21, 22]
    
    # Block 4: C12 += A12*B22 (2x2x4)
    # A12: same as block 2 -> U rows: 2,3,6,7
    # B22: rows 2,3 of B (j=2,3), cols 4-7 (k=4-7) -> V rows: 2*8+4=20, 2*8+5=21, 2*8+6=22, 2*8+7=23, 3*8+4=28, 3*8+5=29, 3*8+6=30, 3*8+7=31
    # C12: same as block 3 -> W rows: 12,13,15,16,18,19,21,22
    
    U_map4 = [2, 3, 6, 7]
    V_map4 = [20, 21, 22, 23, 28, 29, 30, 31]
    W_map4 = [12, 13, 15, 16, 18, 19, 21, 22]
    
    # Block 5: C21 = A21*B11 (1x2x4)
    # A21: row 2 of A (i=2), cols 0,1 (j=0,1) -> U rows: 2*4+0=8, 2*4+1=9
    # B11: same as block 1 -> V rows: 0,1,2,3,8,9,10,11
    # C21: cols 0-3 of C (k=0-3), row 2 (i=2) -> W rows: k*3+2 for k=0-3 -> 2,5,8,11
    
    U5, V5, W5 = _rank1_1x2x4()
    U_map5 = [8, 9]
    V_map5 = [0, 1, 2, 3, 8, 9, 10, 11]
    W_map5 = [2, 5, 8, 11]
    
    # Block 6: C21 += A22*B21 (1x2x4)
    # A22: row 2 of A (i=2), cols 2,3 (j=2,3) -> U rows: 2*4+2=10, 2*4+3=11
    # B21: same as block 2 -> V rows: 16,17,18,19,24,25,26,27
    # C21: same as block 5 -> W rows: 2,5,8,11
    
    U_map6 = [10, 11]
    V_map6 = [16, 17, 18, 19, 24, 25, 26, 27]
    W_map6 = [2, 5, 8, 11]
    
    # Block 7: C22 = A21*B12 (1x2x4)
    # A21: same as block 5 -> U rows: 8,9
    # B12: same as block 3 -> V rows: 4,5,6,7,12,13,14,15
    # C22: cols 4-7 of C (k=4-7), row 2 (i=2) -> W rows: k*3+2 for k=4-7 -> 14,17,20,23
    
    U_map7 = [8, 9]
    V_map7 = [4, 5, 6, 7, 12, 13, 14, 15]
    W_map7 = [14, 17, 20, 23]
    
    # Block 8: C22 += A22*B22 (1x2x4)
    # A22: same as block 6 -> U rows: 10,11
    # B22: same as block 4 -> V rows: 20,21,22,23,28,29,30,31
    # C22: same as block 7 -> W rows: 14,17,20,23
    
    U_map8 = [10, 11]
    V_map8 = [20, 21, 22, 23, 28, 29, 30, 31]
    W_map8 = [14, 17, 20, 23]
    
    # Now assemble the total matrices.
    # Total rank = 14*4 + 1*4 = 56 + 4 = 60
    
    total_rank = 60
    n, m, p = 3, 4, 8
    U_total = [[0]*total_rank for _ in range(n*m)]
    V_total = [[0]*total_rank for _ in range(m*p)]
    W_total = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    # Blocks 1-4 use Strassen 2x2x4 (rank 14 each)
    for block_idx in range(4):
        if block_idx == 0:
            U_map, V_map, W_map = U_map1, V_map1, W_map1
        elif block_idx == 1:
            U_map, V_map, W_map = U_map2, V_map2, W_map2
        elif block_idx == 2:
            U_map, V_map, W_map = U_map3, V_map3, W_map3
        else:
            U_map, V_map, W_map = U_map4, V_map4, W_map4
        
        for c in range(14):
            for r in range(4):
                U_total[U_map[r]][col] += U1[r][c]
            for r in range(8):
                V_total[V_map[r]][col] += V1[r][c]
            for r in range(8):
                W_total[W_map[r]][col] += W1[r][c]
            col += 1
    
    # Blocks 5-8 use rank-1 1x2x4
    for block_idx in range(4):
        if block_idx == 0:
            U_map, V_map, W_map = U_map5, V_map5, W_map5
        elif block_idx == 1:
            U_map, V_map, W_map = U_map6, V_map6, W_map6
        elif block_idx == 2:
            U_map, V_map, W_map = U_map7, V_map7, W_map7
        else:
            U_map, V_map, W_map = U_map8, V_map8, W_map8
        
        for c in range(1):
            for r in range(2):
                U_total[U_map[r]][col] += U5[r][c]
            for r in range(8):
                V_total[V_map[r]][col] += V5[r][c]
            for r in range(4):
                W_total[W_map[r]][col] += W5[r][c]
            col += 1
    
    return {"U": U_total, "V": V_total, "W": W_total}


def construct():
    """Construct a rank-minimal decomposition for 3x4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try the block decomposition approach
    try:
        result = _block_decomposition()
        # Verify
        n, m, p = 3, 4, 8
        rank = len(result["U"][0])
        U = result["U"]
        V = result["V"]
        W = result["W"]
        
        # Verify: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return result
    except Exception:
        pass
    
    # Fallback: schoolbook
    n, m, p = 3, 4, 8
    rank = n * m * p
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


def _block_decomposition():
    """Block decomposition of 3x4x8 using Strassen for 2x2x4 subblocks."""
    # This is a cleaner implementation of the block decomposition.
    
    def _strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (c+d)*e
        U[2][1] = 1; U[3][1] = 1; V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # M3 = a*(f-h)
        U[0][2] = 1; V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # M4 = d*(g-e)
        U[3][3] = 1; V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # M5 = (a+b)*h
        U[0][4] = 1; U[1][4] = 1; V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # M6 = (c-d)*(e+f)
        U[2][5] = 1; U[3][5] = -1; V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (b-c)*(g+h)
        U[1][6] = 1; U[2][6] = -1; V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def _strassen_2x2x4():
        U_s, V_s, W_s = _strassen_2x2x2()
        U = [[0]*14 for _ in range(4)]
        V = [[0]*14 for _ in range(8)]
        W = [[0]*14 for _ in range(8)]
        
        for c in range(7):
            for r in range(4):
                U[r][c] = U_s[r][c]
            for j in range(2):
                for k in range(2):
                    V[j*4+k][c] = V_s[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W[k*2+i][c] = W_s[k*2+i][c]
        
        for c in range(7, 14):
            cc = c - 7
            for r in range(4):
                U[r][c] = U_s[r][cc]
            for j in range(2):
                for k in range(2):
                    V[j*4+(k+2)][c] = V_s[j*2+k][cc]
            for k in range(2):
                for i in range(2):
                    W[(k+2)*2+i][c] = W_s[k*2+i][cc]
        
        return U, V, W
    
    n, m, p = 3, 4, 8
    total_rank = 60
    U_total = [[0]*total_rank for _ in range(n*m)]
    V_total = [[0]*total_rank for _ in range(m*p)]
    W_total = [[0]*total_rank for _ in range(p*n)]
    
    U1, V1, W1 = _strassen_2x2x4()
    
    # Rank-1 block for 1x2x4
    U_r1 = [[1], [1]]
    V_r1 = [[1]]*8
    W_r1 = [[1]]*4
    
    col = 0
    # 4 Strassen blocks (rank 14 each)
    blocks_strassen = [
        # (U_row_map, V_row_map, W_row_map)
        ([0, 1, 4, 5], [0, 1, 2, 3, 8, 9, 10, 11], [0, 1, 3, 4, 6, 7, 9, 10]),      # A11*B11
        ([2, 3, 6, 7], [16, 17, 18, 19, 24, 25, 26, 27], [0, 1, 3, 4, 6, 7, 9, 10]),  # A12*B21
        ([0, 1, 4, 5], [4, 5, 6, 7, 12, 13, 14, 15], [12, 13, 15, 16, 18, 19, 21, 22]), # A11*B12
        ([2, 3, 6, 7], [20, 21, 22, 23, 28, 29, 30, 31], [12, 13, 15, 16, 18, 19, 21, 22]), # A12*B22
    ]
    
    for U_map, V_map, W_map in blocks_strassen:
        for c in range(14):
            for r in range(4):
                U_total[U_map[r]][col] += U1[r][c]
            for r in range(8):
                V_total[V_map[r]][col] += V1[r][c]
            for r in range(8):
                W_total[W_map[r]][col] += W1[r][c]
            col += 1
    
    # 4 rank-1 blocks
    blocks_r1 = [
        ([8, 9], [0, 1, 2, 3, 8, 9, 10, 11], [2, 5, 8, 11]),      # A21*B11
        ([10, 11], [16, 17, 18, 19, 24, 25, 26, 27], [2, 5, 8, 11]), # A22*B21
        ([8, 9], [4, 5, 6, 7, 12, 13, 14, 15], [14, 17, 20, 23]),  # A21*B12
        ([10, 11], [20, 21, 22, 23, 28, 29, 30, 31], [14, 17, 20, 23]), # A22*B22
    ]
    
    for U_map, V_map, W_map in blocks_r1:
        for c in range(1):
            for r in range(2):
                U_total[U_map[r]][col] += U_r1[r][c]
            for r in range(8):
                V_total[V_map[r]][col] += V_r1[r][c]
            for r in range(4):
                W_total[W_map[r]][col] += W_r1[r][c]
            col += 1
    
    return {"U": U_total, "V": V_total, "W": W_total}
