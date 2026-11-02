import numpy as np
import time

DIMENSIONS = [2, 2, 8]

# Strassen's 7 multiplications for 2x2 * 2x2
# A = [[a, b], [c, d]], B = [[e, f], [g, h]]
# P1 = (a+d)(e+h)
# P2 = (c+d)h
# P3 = a(f-h)
# P4 = d(g-e)
# P5 = (a+b)g
# P6 = (c-a)(e+f)
# P7 = (b-d)(h+g)
# C11 = P1 + P4 - P5 + P7
# C12 = P3 + P5
# C21 = P2 + P4
# C22 = P1 - P2 + P3 + P6

def _strassen_2x2x2_UVW():
    """
    Returns U, V, W for 2x2 * 2x2 = 2x2 using Strassen's rank-7 algorithm.
    U shape: (4, 7) - indices (i, j) where i in {0,1}, j in {0,1}
    V shape: (4, 7) - indices (j, k) where j in {0,1}, k in {0,1}
    W shape: (4, 7) - indices (k, i) where k in {0,1}, i in {0,1}
    
    Convention: W[k*n+i, r] for n=2
    """
    n = 2
    # U: row index = i*m + j = i*2 + j, where i is row of A, j is column of A
    # V: row index = j*p + k = j*2 + k, where j is column of A, k is row of B
    # W: row index = k*n + i = k*2 + i, where k is row of B, i is row of C
    
    U = np.zeros((4, 7), dtype=np.int32)
    V = np.zeros((4, 7), dtype=np.int32)
    W = np.zeros((4, 7), dtype=np.int32)
    
    # A elements: A[0,0]=a (idx 0), A[0,1]=b (idx 1), A[1,0]=c (idx 2), A[1,1]=d (idx 3)
    # B elements: B[0,0]=e (idx 0), B[0,1]=f (idx 1), B[1,0]=g (idx 2), B[1,1]=h (idx 3)
    # C elements: C[0,0] (k=0,i=0 idx 0), C[0,1] (k=1,i=0 idx 1), C[1,0] (k=0,i=1 idx 2), C[1,1] (k=1,i=1 idx 3)
    
    # P1 = (a+d)(e+h)
    # a = A[0,0], d = A[1,1] -> U[0,0]=1, U[3,0]=1
    # e = B[0,0], h = B[1,1] -> V[0,0]=1, V[3,0]=1
    # C11 += P1, C22 += P1 -> W[0,0]=1, W[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (c+d)h
    # c = A[1,0], d = A[1,1] -> U[2,1]=1, U[3,1]=1
    # h = B[1,1] -> V[3,1]=1
    # C21 += P2, C22 -= P2 -> W[2,1]=1, W[3,1]=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[3, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # P3 = a(f-h)
    # a = A[0,0] -> U[0,2]=1
    # f = B[0,1], h = B[1,1] -> V[1,2]=1, V[3,2]=-1
    # C12 += P3, C22 += P3 -> W[1,2]=1, W[3,2]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = d(g-e)
    # d = A[1,1] -> U[3,3]=1
    # g = B[1,0], e = B[0,0] -> V[2,3]=1, V[0,3]=-1
    # C11 += P4, C21 += P4 -> W[0,3]=1, W[2,3]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5 = (a+b)g
    # a = A[0,0], b = A[0,1] -> U[0,4]=1, U[1,4]=1
    # g = B[1,0] -> V[2,4]=1
    # C11 -= P5, C12 += P5 -> W[0,4]=-1, W[1,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[2, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6 = (c-a)(e+f)
    # c = A[1,0], a = A[0,0] -> U[2,5]=1, U[0,5]=-1
    # e = B[0,0], f = B[0,1] -> V[0,5]=1, V[1,5]=1
    # C22 += P6 -> W[3,5]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (b-d)(h+g)
    # b = A[0,1], d = A[1,1] -> U[1,6]=1, U[3,6]=-1
    # h = B[1,1], g = B[1,0] -> V[3,6]=1, V[2,6]=1
    # C11 += P7 -> W[0,6]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    
    Strategy: View the 2x8 matrix as 4 blocks of 2x2. Apply Strassen's rank-7
    algorithm to each block independently. This gives rank 4*7 = 28.
    
    U shape: (4, 28) - (i*m+j, block*7+r) where i,j in {0,1}, block in {0..3}, r in {0..6}
    V shape: (16, 28) - (j*p+k, block*7+r) where j in {0,1}, k in {0..7}, block in {0..3}, r in {0..6}
    W shape: (16, 28) - (k*n+i, block*7+r) where k in {0..7}, i in {0,1}, block in {0..3}, r in {0..6}
    
    Wait, let me reconsider the block structure.
    
    A is 2x2, B is 2x8. We can write B = [B0 B1 B2 B3] where each Bi is 2x2.
    C = A * B = [A*B0, A*B1, A*B2, A*B3] where each product is 2x2.
    
    For each block b (0..3):
      U contributes to rows (i*m+j) = (i*2+j) for the A-part
      V contributes to rows (j*p+k) = (j*8+k) for the B-part, where k is the row within the 2x8 B
        For block b, the B columns are 2*b and 2*b+1.
        So k in B is actually the row index of B (0 or 1), and the column within the block.
        Wait, V has shape (m*p, rank) = (2*8, rank) = (16, rank).
        Row index of V: j*p + k where j is column of A (0 or 1), k is row of B (0..7).
        
        For block b (columns 2b, 2b+1 of B):
        The B-part of this block is B[:, 2b:2b+2].
        In the tensor, B's indices are (j, k) where j is A's column, k is B's row.
        But B is 2x8, so B[j, col] where col is 0..7.
        
        Let me re-examine the tensor definition:
        (A B)_{i,k} = sum_{j} sum_{col} A_{i,j} B_{j,col}
        
        For 2x2 * 2x8:
        n=2, m=2, p=8
        A is n x m = 2x2
        B is m x p = 2x8
        C is n x p = 2x8
        
        U has shape (n*m, rank) = (4, rank). Row index: i*m + j = 2i + j
        V has shape (m*p, rank) = (16, rank). Row index: j*p + k = 8j + k, where k is 0..7 (column of B, which is row of C)
        
        Wait, the problem says: "W[k*n+i,r]" for the output factor.
        C is n x p = 2 x 8. C[i, col] where i is 0..1, col is 0..7.
        W has shape (p*n, rank) = (16, rank). Row index: k*n + i = 8k + i? No, that doesn't match.
        
        Let me re-read: "W[k*n+i,r]"
        p*n = 16. k ranges over p=8 values, i ranges over n=2 values.
        So W row index = k*2 + i where k is 0..7 (column index of C), i is 0..1 (row index of C).
        
        So C[i, k] = sum_r U[2i+j, r] * V[8j+k, r] * W[2k+i, r] ... no.
        
        The standard tensor decomposition:
        T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
        
        Here:
        T_{i,j,k} represents the coefficient in (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
        So T_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
        
        U has shape (n*m, rank): U[i*m+j, r]
        V has shape (m*p, rank): V[j*p+k, r]
        W has shape (p*n, rank): W[k*n+i, r]
        
        And the decomposition says:
        (A*B)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_r (sum_j A_{i,j} U[i*m+j, r]) * (sum_{j',k'} B_{j',k'} V[j'*p+k', r]) * W[k*n+i, r]
        
        Actually, the standard form is:
        T_{i,j,k} = sum_r U_{i,m+j,r} * V_{m+j,p+k,r} * W_{p+k,n+i,r}
        
        Let me just use the block approach.
        
        For block b (0..3), columns 2b and 2b+1 of B:
        Let B_b = B[:, 2b:2b+2] be the 2x2 block.
        C_b = A * B_b is 2x2.
        
        For each block b, we use Strassen's 7 multiplications.
        The rank is 4 * 7 = 28.
        
        For block b, multiplication index r is b*7 + s where s is 0..6.
        
        U[2i+j, b*7+s] = U_strassen[i*2+j, s]
        V[j*8+k, b*7+s] = V_strassen[j*2+k', s] where k' is the row within the block
          For block b, B's column k corresponds to k' = k - 2b if k is in {2b, 2b+1}.
          So if k = 2b, k' = 0; if k = 2b+1, k' = 1.
          V[8j+k, b*7+s] = V_strassen[2j+k', s] where k' = k - 2b.
          
        W[2k+i, b*7+s] = W_strassen[2k'+i, s] where k' is the column within C_b.
          For block b, C's column k corresponds to k' = k - 2b if k is in {2b, 2b+1}.
          So W[2k+i, b*7+s] = W_strassen[2(k-2b)+i, s] = W_strassen[2k'-i... wait.
          
        Let me just construct it carefully.
        """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 4 blocks of 2 columns each
    rank = num_blocks * 7
    rank = int(rank)
    
    U, V, W = _strassen_2x2x2_UVW()
    # U: (4, 7), V: (4, 7), W: (4, 7)
    
    U_out = np.zeros((n * m, rank), dtype=np.int32)
    V_out = np.zeros((m * p, rank), dtype=np.int32)
    W_out = np.zeros((p * n, rank), dtype=np.int32)
    
    for b in range(num_blocks):
        # Block b uses columns 2b and 2b+1 of B
        # U: row = i*m + j = 2i + j, where i in {0,1}, j in {0,1}
        #   U_out[2*i + j, b*7 + s] = U[i*2 + j, s]
        for i in range(n):
            for j in range(m):
                for s in range(7):
                    U_out[i * m + j, b * 7 + s] = U[i * 2 + j, s]
        
        # V: row = j*p + k = 8j + k, where j in {0,1}, k in {0..7}
        #   For block b, k = 2b + k' where k' in {0,1}
        #   V_out[8*j + (2*b + k'), b*7 + s] = V[2*j + k', s]
        for j in range(m):
            for k_prime in range(2):
                k = 2 * b + k_prime
                for s in range(7):
                    V_out[j * p + k, b * 7 + s] = V[j * 2 + k_prime, s]
        
        # W: row = k*n + i = 2k + i, where k in {0..7}, i in {0,1}
        #   For block b, k = 2b + k' where k' in {0,1}
        #   W_out[2*(2*b + k') + i, b*7 + s] = W[2*k_prime + i, s]
        for i in range(n):
            for k_prime in range(2):
                k = 2 * b + k_prime
                for s in range(7):
                    W_out[k * n + i, b * 7 + s] = W[k_prime * 2 + i, s]
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}


if __name__ == "__main__":
    # Quick verification
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = 2, 2, 8
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_{j',k'} B[j',k'] V[j'*p+k', r]) * W[k*n+i, r]
    # Actually, the correct formula is:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... no
    # The tensor T_{i,j,k} = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # And (A*B)_{i,k} = sum_j A_{i,j} * T_{i,j,k} * B_{j,k} ... no
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T_{i,j,k} should be 1 for all i,j,k, and the decomposition is:
    # 1 = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] ... no, that's not right either.
    
    # Let me just check numerically.
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for r in range(U.shape[1]):
                # Sum over j
                sum_A = 0.0
                sum_B = 0.0
                for j in range(m):
                    sum_A += A[i, j] * U[i * m + j, r]
                    sum_B += B[j, k] * V[j * p + k, r]
                total += sum_A * sum_B * W[k * n + i, r]
            C_computed[i, k] = total
    
    print("Max error:", np.max(np.abs(C_computed - C_expected)))
    print("Rank:", U.shape[1])
