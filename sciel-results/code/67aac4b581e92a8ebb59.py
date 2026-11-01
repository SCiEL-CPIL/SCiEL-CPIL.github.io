"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START

def _to_matrix(frac_list):
    """Convert list of lists of Fractions to numpy array of floats."""
    return np.array([[float(x) for x in row] for row in frac_list], dtype=np.float64)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # The tensor T for (n x m) * (m x p) -> (n x p)
    # Indexing: A[i,j], B[j,k], C[i,k]
    # T[i,j,k,l] = delta_{j,l} where i in [0,n), j in [0,m), k in [0,p), l in [0,m)
    # Actually the standard tensor is T[i,j,k] = delta_{j,k} for (i,j) x (j,k) -> (i,k)
    # Let's use the convention from the task:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * ... 
    # Wait, let me re-read the task.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Standard tensor decomposition for matrix mult:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # In bilinear form: C = sum_r U_r A V_r W_r^T ... no.
    # The tensor T has indices (i, j, k) where A is (i,j), B is (j,k), C is (i,k).
    # T[i,j,k] = 1 if the multiplication involves A[i,j]*B[j,k].
    # Actually, the matrix multiplication tensor is T[i,j,k] = delta_{j,k} ... no.
    # Let's think of it as a 3-tensor T with indices (i, j, k) where:
    # A is n x m (indices i, j)
    # B is m x p (indices j, k)
    # C is n x p (indices i, k)
    # T[i, j, k] = 1 if j == j (always true), so T[i,j,k] = 1 for all valid i,j,k? No.
    # The tensor is T[i,j,k] where the multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor entries are T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right either. The tensor represents the bilinear map.
    # T is a 3-tensor of size (n*p, n*m, m*p) or similar.
    
    # Let's use the standard formulation:
    # The matrix multiplication tensor T of size (n, m, p) has entries T[i,j,k] = 1 
    # where i indexes rows of A (and C), j indexes columns of A (and rows of B), 
    # k indexes columns of B (and C).
    # So T is of shape (n, m, p) with all entries 1? No.
    # Actually, T[i,j,k] = 1 means that A[i,j] * B[j,k] contributes to C[i,k].
    # Since every such product contributes, T[i,j,k] = 1 for all i,j,k.
    # But that's a rank-1 tensor (all ones), which is wrong.
    
    # Let me reconsider. The matrix multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i^C e_j^A e_k^B delta_{j,k} ... no.
    # T[i,j,k] where i is row of C/A, j is col of A/row of B, k is col of C/B.
    # The condition is that j must match, so T[i,j,k] = 1 if the indices are consistent.
    # Actually, for matrix multiplication, T[i,j,k] = 1 for all valid i, j, k because
    # every A[i,j] multiplies with B[j,k] to give C[i,k]. The "tensor" here is just
    # the structure of which products are taken.
    
    # I think the correct way is: the tensor T has entries T[i,j,k] = 1 for all 
    # i in [0,n), j in [0,m), k in [0,p). This is because the bilinear form is
    # <A, B> -> C where C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is of shape (n*p, n*m, m*p) in flattened form, or (n, m, p) in 
    # 3D form where T[i,j,k] = 1.
    
    # Wait, that's still all ones. Let me look at this differently.
    # The tensor for matrix multiplication M_{n,m,p} is:
    # T[i,j,k] = delta_{j,j} ... which is 1. So T is the all-ones tensor?
    # No, that can't be right because the rank would be 1.
    
    # I think I'm confusing myself. Let me use the standard definition:
    # The matrix multiplication tensor T_{n,m,p} is a 3-tensor with entries:
    # T[i,j,k] = 1 if i is a row index of A, j is a column index of A / row index of B,
    # k is a column index of B, and the product A[i,j]*B[j,k] is part of C[i,k].
    # Since this is always true for valid indices, T[i,j,k] = 1 for all i,j,k.
    # But this gives a rank-1 tensor, which is clearly wrong.
    
    # Ah, I see the issue. The tensor is NOT all ones. The tensor T is defined such that
    # C[i,k] = sum_{j} A[i,j] B[j,k]. In tensor notation, this is:
    # T = sum_{i,j,k} e_i^{(n)} \otimes e_j^{(m)} \otimes e_k^{(p)} * (something)
    # The "something" is 1, so T[i,j,k] = 1. But this is a rank-1 tensor!
    
    # I must be misunderstanding the tensor structure. Let me think again.
    # Actually, the matrix multiplication tensor is NOT all ones. It's a specific 
    # tensor where T[i,j,k] = 1 if and only if the indices match in the multiplication.
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid i, j, k because
    # every entry of A multiplies with the corresponding entry of B.
    
    # Wait, I think the issue is that the tensor is of shape (n, m, p) and T[i,j,k] = 1
    # for all i, j, k. This IS a rank-1 tensor (the outer product of three all-ones vectors).
    # But matrix multiplication has rank > 1, so something is wrong.
    
    # Let me look at this from the bilinear map perspective. The bilinear map is:
    # (A, B) -> C where C = A * B.
    # This is a bilinear map from R^{n*m} x R^{m*p} -> R^{n*p}.
    # The associated tensor T is in R^{n*p} \otimes R^{n*m} \otimes R^{m*p}.
    # T_{(i,k), (i',j), (j',k')} = 1 if i=i', k=k', j=j'.
    # So T[(i,k), (i',j), (j',k')] = delta_{i,i'} delta_{k,k'} delta_{j,j'}.
    # This is NOT all ones; it's a sparse tensor!
    
    # So the tensor T has shape (n*p, n*m, m*p) and T[(i,k), (i',j), (j',k')] = 1 
    # if i=i', k=k', j=j'.
    
    # In the decomposition:
    # C[(i,k)] = sum_r U[(i,k), r] * ... no, that's not right either.
    
    # Let me use the standard tensor decomposition for bilinear maps.
    # A bilinear map f: V x W -> Z can be represented as a tensor T in V* \otimes W* \otimes Z.
    # The rank of T is the minimum number of rank-1 tensors needed to represent T.
    # For matrix multiplication, T is in (R^{n*p})^* \otimes (R^{n*m})^* \otimes (R^{m*p})^*.
    # Wait, no. The bilinear map is (A, B) -> C = A*B.
    # So T is in (R^{n*m})^* \otimes (R^{m*p})^* \otimes (R^{n*p}).
    # T[a, b, c] where a indexes A, b indexes B, c indexes C.
    # T[a, b, c] = 1 if A[a] * B[b] contributes to C[c].
    # A[a] where a = i*m + j (i row, j col of A)
    # B[b] where b = j*p + k (j row, k col of B)
    # C[c] where c = i*p + k (i row, k col of C)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = i*p+k for some i, j, k.
    # So T[a, b, c] = 1 if (a mod m) == (b // p) and (a // m) == (c // p) and (b mod p) == (c mod p).
    
    # Now, the decomposition with rank r:
    # T[a, b, c] = sum_{r=1}^r U[a, r] * V[b, r] * W[c, r]
    # where U is (n*m) x r, V is (m*p) x r, W is (n*p) x r.
    
    # Wait, the task says W has shape (p*n, rank) and follows convention W[k*n+i, r].
    # So W is indexed by (k, i) where k is col of C, i is row of C.
    # C[c] where c = k*n + i? That would mean c = k*n + i, so i = c mod n, k = c // n.
    # But C is n x p, so c should be i*p + k, not k*n + i.
    # The task says "output factor follows the official trace convention W[k*n+i,r]".
    # So W is indexed by (k*n + i, r) where i is row, k is col.
    # This means W has p*n rows, indexed by k*n + i.
    
    # Let me re-read: "W[k*n+i,r]" where presumably i is in [0,n) and k is in [0,p).
    # So W has p*n rows. The row index is k*n + i.
    # This is a different ordering than c = i*p + k.
    
    # So the decomposition is:
    # C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... but j is summed over?
    # No, the bilinear map is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form:
    # T[(i,j), (j,k), (i,k)] = 1
    # Decomposition:
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 if i=i', k=k', j=j'.
    
    # So: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{i,i'} delta_{k,k'} delta_{j,j'}
    
    # This is the constraint we need to satisfy.
    
    # For now, let's try the Strassen-based approach.
    # Strassen's algorithm for 2x2 x 2x2 has rank 7.
    # For 2x2 x 2x6, we can split the 2x6 into three 2x2 blocks.
    # So we need three 2x2 x 2x2 multiplications, each of rank 7, total rank 21.
    
    # Let's implement Strassen's algorithm for 2x2 x 2x2 and then extend to 2x2 x 2x6.
    
    # Strassen's 7 multiplications for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2 x 2x6, we have A is 2x2, B is 2x6 = [B1, B2, B3] where each Bi is 2x2.
    # C = A * B = [A*B1, A*B2, A*B3], each 2x2.
    # We can use Strassen for each block, but that's 3*7 = 21 multiplications.
    # Alternatively, we can share some computations.
    
    # Let's just use 3 independent Strassen decompositions for rank 21.
    
    # First, let's define the Strassen decomposition for 2x2 x 2x2.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # The 7 intermediate products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For the tensor decomposition, we need U, V, W matrices.
    # U has 4 rows (for A11, A12, A21, A22) and 7 columns.
    # V has 4 rows (for B11, B12, B21, B22) and 7 columns.
    # W has 4 rows (for C11, C12, C21, C22) and 7 columns.
    
    # U[i, r] where i indexes A entries, r indexes the 7 products.
    # V[j, r] where j indexes B entries, r indexes the 7 products.
    # W[k, r] where k indexes C entries, r indexes the 7 products.
    
    # For M1 = (A11 + A22)(B11 + B22):
    # U[A11, M1] = 1, U[A22, M1] = 1
    # V[B11, M1] = 1, V[B22, M1] = 1
    # W[C11, M1] = 1, W[C22, M1] = 1
    
    # For M2 = (A21 + A22)B11:
    # U[A21, M2] = 1, U[A22, M2] = 1
    # V[B11, M2] = 1
    # W[C21, M2] = 1, W[C22, M2] = -1
    
    # And so on.
    
    # Let's code this up.
    
    def strassen_2x2():
        """Return U, V, W for 2x2 x 2x2 Strassen decomposition."""
        # A indices: 0=A11, 1=A12, 2=A21, 3=A22
        # B indices: 0=B11, 1=B12, 2=B21, 3=B22
        # C indices: 0=C11, 1=C12, 2=C21, 3=C22
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11 + A22)(B11 + B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21 + A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = A11(B12 - B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A22(B21 - B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A11 + A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[1][6] = 1
        
        return U, V, W
    
    # Now for 2x2 x 2x6:
    # A is 2x2, indexed as: 0=A11, 1=A12, 2=A21, 3=A22 (row-major)
    # B is 2x6, indexed as: 0=B11, 1=B12, 2=B13, 3=B14, 4=B15, 5=B16, 6=B21, 7=B22, 8=B23, 9=B24, 10=B25, 11=B26
    # C is 2x6, indexed as: 0=C11, 1=C12, 2=C13, 3=C14, 4=C15, 5=C16, 6=C21, 7=C22, 8=C23, 9=C24, 10=C25, 11=C26
    # But W uses convention W[k*n+i, r] where i is row, k is col.
    # So W index = k*2 + i where i in {0,1}, k in {0,1,2,3,4,5}.
    # W[0] = C11 (i=0, k=0), W[1] = C21 (i=1, k=0), W[2] = C12 (i=0, k=1), W[3] = C22 (i=1, k=1), etc.
    
    # B index = j*p + k where j is row of B (0 or 1), k is col of B (0..5).
    # B[0] = B11, B[1] = B12, ..., B[5] = B16, B[6] = B21, ..., B[11] = B26.
    
    # A index = i*m + j where i is row of A (0 or 1), j is col of A (0 or 1).
    # A[0] = A11, A[1] = A12, A[2] = A21, A[3] = A22.
    
    # We split B into three 2x2 blocks: B1 = cols 0-1, B2 = cols 2-3, B3 = cols 4-5.
    # For each block, we use Strassen.
    
    U_total = [[0]*21 for _ in range(4)]
    V_total = [[0]*21 for _ in range(12)]
    W_total = [[0]*21 for _ in range(12)]
    
    U_str, V_str, W_str = strassen_2x2()
    
    for block in range(3):
        base = block * 7
        # A part: same for all blocks
        for i in range(4):
            for r in range(7):
                U_total[i][base + r] = U_str[i][r]
        
        # B part: block b uses cols 2b, 2b+1 of B
        # B indices for block b: 2b, 2b+1 (row 0), 6+2b, 6+2b+1 (row 1)
        # Strassen B indices: 0=B11, 1=B12, 2=B21, 3=B22
        # Map to actual B indices:
        # B11 -> 2b, B12 -> 2b+1, B21 -> 6+2b, B22 -> 6+2b+1
        b_map = [2*block, 2*block+1, 6+2*block, 6+2*block+1]
        for j in range(4):
            for r in range(7):
                V_total[b_map[j]][base + r] = V_str[j][r]
        
        # W part: block b produces C cols 2b, 2b+1
        # C indices: C11, C12, C21, C22 for this block
        # W index = k*2 + i where k is col, i is row
        # C11: i=0, k=2b -> W[2b*2+0] = W[4b]
        # C12: i=0, k=2b+1 -> W[(2b+1)*2+0] = W[4b+2]
        # C21: i=1, k=2b -> W[2b*2+1] = W[4b+1]
        # C22: i=1, k=2b+1 -> W[(2b+1)*2+1] = W[4b+3]
        w_map = [4*block, 4*block+2, 4*block+1, 4*block+3]
        for k in range(4):
            for r in range(7):
                W_total[w_map[k]][base + r] = W_str[k][r]
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        # Build the tensor T[a, b, c]
        # T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' and i=i', k=k', j=j'
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1
        
        # Check decomposition: T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is a 3-tensor contraction.
        # For each (a, b, c), check if sum_r U[a,r] V[b,r] W[c,r] == T[a, b, c]
        max_err = 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = sum(U_arr[a, r] * V_arr[b, r] * W_arr[c, r] for r in range(rank))
                    err = abs(val - T[a, b, c])
                    if err > max_err:
                        max_err = err
        return max_err
    
    err = verify(U_total, V_total, W_total, n, m, p)
    if err < 1e-9:
        return {"U": U_total, "V": V_total, "W": W_total}
    
    # If verification fails, fall back to schoolbook
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

# EVOLVE-BLOCK-END
