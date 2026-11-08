import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Indices
    # A is n x m (3x5)
    # B is m x p (5x5)
    # C is p x n (5x3) -> Wait, task says W[k*n+i, r]. C is p x n.
    # Standard: C = A B. A (n x m), B (m x p). C (n x p).
    # Task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies C is indexed by k (row of C?) and i (col of C?)
    # Usually C[i,k] = sum A[i,j] B[j,k].
    # If W is p x n, then index is k*n + i.
    # So C[k,i] = sum B[k,j] A[j,i]?
    # Let's check the seed program.
    # Seed: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # Term: U[i*m+j] V[j*p+k] W[k*n+i].
    # Sum over j: U[i,j] V[j,k] W[k,i].
    # This computes (U V) [i,k] weighted by W?
    # No, it's a tensor decomposition.
    # The tensor is T[i,j,k] = delta_{i,k}? No.
    # Matrix mult tensor T[i,j,k] is 1 if...
    # The decomposition is: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = C[i,k]
    # Wait, the seed program has W[k*n+i].
    # So the output is indexed by k and i.
    # This corresponds to C[k,i].
    # Standard C[i,k] = sum A[i,j] B[j,k].
    # If the output is C[k,i], then it's C^T?
    # Or A is m x n?
    # Task: "multiplying a 3 by 5 matrix by a 5 by 5 matrix".
    # A (3x5), B (5x5). C (3x5).
    # Seed U (n*m) = 15. V (m*p) = 25. W (p*n) = 15.
    # If A is 3x5, n=3, m=5.
    # U index i*m+j -> i in 0..2, j in 0..4.
    # V index j*p+k -> j in 0..4, k in 0..4.
    # W index k*n+i -> k in 0..4, i in 0..2.
    # So the result is a matrix of size p x n = 5x3.
    # But A B is 3x5.
    # So the task expects the result to be transposed? Or A is 5x3?
    # "3 by 5 matrix by a 5 by 5 matrix".
    # Usually means (3x5) * (5x5) = 3x5.
    # But W is p*n = 5*3 = 15.
    # So the output dimension is 5x3.
    # This implies the multiplication is B * A?
    # Or the indices are defined such that C[k,i] = sum A[i,j] B[j,k].
    # If A is 3x5, B is 5x5, then C is 3x5.
    # If the output is 5x3, then it must be C^T.
    # I will follow the seed program's index convention exactly.
    # The seed program works for the tensor defined by those indices.
    
    # My decomposition must match this index structure.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    
    # I will build the factors for the 18 blocks identified.
    
    # Helper to add a block decomposition to the global U, V, W
    # Block A indices: i_start, i_end, j_start, j_end
    # Block B indices: j_start, j_end, k_start, k_end
    # Block C indices: k_start, k_end, i_start, i_end
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(A_i_range, A_j_range, B_j_range, B_k_range, rank, gen_func):
        # gen_func returns local U, V, W for this block
        # Local U shape: (len(A_i_range)*len(A_j_range), rank)
        # Local V shape: (len(B_j_range)*len(B_k_range), rank)
        # Local W shape: (len(B_k_range)*len(A_i_range), rank)
        
        local_U, local_V, local_W = gen_func()
        
        # Embed into global
        global_U = np.zeros((n*m, rank))
        global_V = np.zeros((m*p, rank))
        global_W = np.zeros((p*n, rank))
        
        for li, i in enumerate(A_i_range):
            for lj, j in enumerate(A_j_range):
                row = i*m + j
                local_row = li*len(A_j_range) + lj
                global_U[row, :] = local_U[local_row, :]
                
        for lj, j in enumerate(B_j_range):
            for lk, k in enumerate(B_k_range):
                row = j*p + k
                local_row = lj*len(B_k_range) + lk
                global_V[row, :] = local_V[local_row, :]
                
        for lk, k in enumerate(B_k_range):
            for li, i in enumerate(A_i_range):
                row = k*n + i
                local_row = lk*len(A_i_range) + li
                global_W[row, :] = local_W[local_row, :]
                
        U_list.append(global_U)
        V_list.append(global_V)
        W_list.append(global_W)
        
    # Ranges
    # i: 0,1,2. j: 0,1,2,3,4. k: 0,1,2,3,4.
    # Split i: [0,1], [2]
    # Split j: [0,1], [2,3], [4]
    # Split k: [0,1], [2,3], [4]
    
    i1 = [0, 1]; i2 = [2]
    j1 = [0, 1]; j2 = [2, 3]; j3 = [4]
    k1 = [0, 1]; k2 = [2, 3]; k3 = [4]
    
    # 1. A11 B11 (2x2 x 2x2) Strassen
    add_block(i1, j1, j1, k1, 7, lambda: strassen_222())
    # 2. A12 B21 (2x2 x 2x2) Strassen
    add_block(i1, j2, j2, k1, 7, lambda: strassen_222())
    # 3. A13 B31 (2x1 x 1x2) Rank 1
    add_block(i1, j3, j3, k1, 1, lambda: rank1_212())
    
    # 4. A11 B12 (2x2 x 2x2) Strassen
    add_block(i1, j1, j1, k2, 7, lambda: strassen_222())
    # 5. A12 B22 (2x2 x 2x2) Strassen
    add_block(i1, j2, j2, k2, 7, lambda: strassen_222())
    # 6. A13 B32 (2x1 x 1x2) Rank 1
    add_block(i1, j3, j3, k2, 1, lambda: rank1_212())
    
    # 7. A11 B13 (2x2 x 2x1) Rank 2
    add_block(i1, j1, j1, k3, 2, lambda: rank2_221())
    # 8. A12 B23 (2x2 x 2x1) Rank 2
    add_block(i1, j2, j2, k3, 2, lambda: rank2_221())
    # 9. A13 B33 (2x1 x 1x1) Rank 1
    add_block(i1, j3, j3, k3, 1, lambda: rank1_211())
    
    # 10. A21 B11 (1x2 x 2x2) Rank 2
    add_block(i2, j1, j1, k1, 2, lambda: rank2_122())
    # 11. A22 B21 (1x2 x 2x2) Rank 2
    add_block(i2, j2, j2, k1, 2, lambda: rank2_122())
    # 12. A23 B31 (1x1 x 1x2) Rank 1
    add_block(i2, j3, j3, k1, 1, lambda: rank1_112())
    
    # 13. A21 B12 (1x2 x 2x2) Rank 2
    add_block(i2, j1, j1, k2, 2, lambda: rank2_122())
    # 14. A22 B22 (1x2 x 2x2) Rank 2
    add_block(i2, j2, j2, k2, 2, lambda: rank2_122())
    # 15. A23 B32 (1x1 x 1x2) Rank 1
    add_block(i2, j3, j3, k2, 1, lambda: rank1_112())
    
    # 16. A21 B13 (1x2 x 2x1) Rank 2
    add_block(i2, j1, j1, k3, 2, lambda: rank2_121())
    # 17. A22 B23 (1x2 x 2x1) Rank 2
    add_block(i2, j2, j2, k3, 2, lambda: rank2_121())
    # 18. A23 B33 (1x1 x 1x1) Rank 1
    add_block(i2, j3, j3, k3, 1, lambda: rank1_111())
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    # Convert to list of lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Local generators

def strassen_222():
    # U (2x2), V (2x2), W (2x2)
    # Strassen variables
    # U indices: 0,1,2,3 (i*m+j)
    # V indices: 0,1,2,3 (j*p+k)
    # W indices: 0,1,2,3 (k*n+i)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A is 2x2. A11=U[0], A12=U[1], A21=U[2], A22=U[3]
    # B is 2x2. B11=V[0], B12=V[1], B21=V[2], B22=V[3]
    # C is 2x2. C11=W[0], C12=W[1], C21=W[2], C22=W[3]
    
    # Col 0: M1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[1,0]=-1; W[2,0]=-1; W[3,0]=1
    
    # Col 1: M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=1
    
    # Col 2: M3
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[0,2]=1; W[1,2]=1
    
    # Col 3: M4
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    W[2,3]=1; W[3,3]=1
    
    # Col 4: M5
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=1; W[3,4]=1
    
    # Col 5: M6
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[0,5]=1; W[2,5]=1
    
    # Col 6: M7
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[1,6]=1; W[3,6]=1
    
    return U, V, W

def rank1_212():
    # A 2x1, B 1x2. C 2x2.
    # A: U[0], U[1]. B: V[0]. C: W[0..3].
    # C = A B^T.
    # U[0]V[0]W[0] + U[1]V[0]W[2] ?
    # C[0,0] = A[0]B[0]. C[0,1] = 0. C[1,0] = A[1]B[0]. C[1,1] = 0.
    # Wait, B is 1x2. B[0,0], B[0,1].
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Here j is size 1.
    # C[i,k] = A[i,0] B[0,k].
    # This is outer product of A (col) and B (row).
    # Rank 1.
    U = np.zeros((2, 1))
    V = np.zeros((2, 1)) # V is j*p+k. j=0, k=0,1.
    W = np.zeros((4, 1)) # W is k*n+i. k=0,1. i=0,1.
    
    U[0,0]=1; U[1,0]=1
    V[0,0]=1; V[1,0]=1
    # W[0,0] (k=0,i=0) = 1?
    # W[2,0] (k=1,i=0) = 1?
    # W[1,0] (k=0,i=1) = 1?
    # W[3,0] (k=1,i=1) = 1?
    # C[i,k] = U[i] V[k] W[k,i]?
    # If U[0]=1, U[1]=1, V[0]=1, V[1]=1, W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1.
    # Then C[0,0] = U[0]V[0]W[0,0] = 1.
    # C[0,1] = U[0]V[1]W[1,0] = 1.
    # C[1,0] = U[1]V[0]W[2,0] = 1.
    # C[1,1] = U[1]V[1]W[3,0] = 1.
    # This gives all ones.
    # We need C[i,k] = A[i] B[k].
    # So U[i] = A[i], V[k] = B[k], W[k,i] = 1.
    U[0,0]=1; U[1,0]=1
    V[0,0]=1; V[1,0]=1
    W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1
    return U, V, W

def rank2_221():
    # A 2x2, B 2x1. C 2x1.
    # C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
    # Rank 2.
    U = np.zeros((4, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2)) # k=0,1? No, k is size 1.
    # B is 2x1. j=0,1. k=0.
    # V indices: j*p+k. p=1.
    # V[0] (j=0,k=0), V[1] (j=1,k=0).
    # W indices: k*n+i. k=0. n=2.
    # W[0] (k=0,i=0), W[1] (k=0,i=1).
    
    # Term 1: j=0. A[i,0] B[0,0].
    U[0,0]=1; U[2,0]=1
    V[0,0]=1
    W[0,0]=1; W[1,0]=1
    
    # Term 2: j=1. A[i,1] B[1,0].
    U[1,1]=1; U[3,1]=1
    V[1,1]=1
    W[0,1]=1; W[1,1]=1
    
    return U, V, W

def rank1_211():
    # A 2x1, B 1x1. C 2x1.
    U = np.zeros((2, 1))
    V = np.zeros((1, 1))
    W = np.zeros((2, 1))
    U[0,0]=1; U[1,0]=1
    V[0,0]=1
    W[0,0]=1; W[1,0]=1
    return U, V, W

def rank2_122():
    # A 1x2, B 2x2. C 1x2.
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k].
    U = np.zeros((2, 2))
    V = np.zeros((4, 2))
    W = np.zeros((2, 2))
    
    # j=0
    U[0,0]=1
    V[0,0]=1; V[1,0]=1
    W[0,0]=1; W[1,0]=1
    
    # j=1
    U[1,1]=1
    V[2,1]=1; V[3,1]=1
    W[0,1]=1; W[1,1]=1
    
    return U, V, W

def rank1_112():
    # A 1x1, B 1x2. C 1x2.
    U = np.zeros((1, 1))
    V = np.zeros((2, 1))
    W = np.zeros((2, 1))
    U[0,0]=1
    V[0,0]=1; V[1,0]=1
    W[0,0]=1; W[1,0]=1
    return U, V, W

def rank2_121():
    # A 1x2, B 2x1. C 1x1.
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((1, 2))
    
    U[0,0]=1
    V[0,0]=1
    W[0,0]=1
    
    U[1,1]=1
    V[1,1]=1
    W[0,1]=1
    
    return U, V, W

def rank1_111():
    U = np.zeros((1, 1))
    V = np.zeros((1, 1))
    W = np.zeros((1, 1))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    return U, V, W

if __name__ == "__main__":
    res = construct()
    print(len(res["U"][0]))

