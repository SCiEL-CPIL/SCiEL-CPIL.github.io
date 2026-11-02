import numpy as np

DIMENSIONS = [2, 2, 2]

def construct():
    """
    Returns a rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The tensor T for 2x2 matmul is defined such that:
    (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    
    A rank-r decomposition is a set of r triples (U_r, V_r, W_r) such that:
    T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    The interface requires:
    U: shape (n*m, rank) -> U[i*m + j, r]
    V: shape (m*p, rank) -> V[j*p + k, r]
    W: shape (p*n, rank) -> W[k*n + i, r]  (Official trace convention)
    
    For n=m=p=2:
    U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)  (since W[k*n+i])
               k=0,i=0 -> 0
               k=0,i=1 -> 1
               k=1,i=0 -> 2
               k=1,i=1 -> 3
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # r=0: M1
    # U: A00=1, A11=1
    U[0][0] = 1.0
    U[3][0] = 1.0
    # V: B00=1, B11=1
    V[0][0] = 1.0
    V[3][0] = 1.0
    # W: C00 += M1, C11 += M1
    # C00 is W[0*2+0]=W[0], C11 is W[1*2+1]=W[3]
    W[0][0] = 1.0
    W[3][0] = 1.0
    
    # r=1: M2
    # U: A10=1, A11=1
    U[2][1] = 1.0
    U[3][1] = 1.0
    # V: B00=1
    V[0][1] = 1.0
    # W: C10 += M2, C11 -= M2
    # C10 is W[0*2+1]=W[1], C11 is W[3]
    W[1][1] = 1.0
    W[3][1] = -1.0
    
    # r=2: M3
    # U: A00=1
    U[0][2] = 1.0
    # V: B01=1, B11=-1
    V[1][2] = 1.0
    V[3][2] = -1.0
    # W: C01 += M3, C11 += M3
    # C01 is W[1*2+0]=W[2], C11 is W[3]
    W[2][2] = 1.0
    W[3][2] = 1.0
    
    # r=3: M4
    # U: A11=1
    U[3][3] = 1.0
    # V: B10=1, B00=-1
    V[2][3] = 1.0
    V[0][3] = -1.0
    # W: C00 += M4, C10 += M4
    # C00 is W[0], C10 is W[1]
    W[0][3] = 1.0
    W[1][3] = 1.0
    
    # r=4: M5
    # U: A00=1, A01=1
    U[0][4] = 1.0
    U[1][4] = 1.0
    # V: B11=1
    V[3][4] = 1.0
    # W: C00 -= M5, C01 += M5
    # C00 is W[0], C01 is W[2]
    W[0][4] = -1.0
    W[2][4] = 1.0
    
    # r=5: M6
    # U: A10=1, A00=-1
    U[2][5] = 1.0
    U[0][5] = -1.0
    # V: B00=1, B01=1
    V[0][5] = 1.0
    V[1][5] = 1.0
    # W: C11 += M6
    # C11 is W[3]
    W[3][5] = 1.0
    
    # r=6: M7
    # U: A01=1, A11=-1
    U[1][6] = 1.0
    U[3][6] = -1.0
    # V: B10=1, B11=1
    V[2][6] = 1.0
    V[3][6] = 1.0
    # W: C00 += M7
    # C00 is W[0]
    W[0][6] = 1.0
    
    # Verification (Optional but recommended for robustness in complex tasks)
    # Construct tensors and check identity
    # T[i, j, k] = delta_{i,k} is NOT the tensor. 
    # The tensor for matmul is T[i, j, k] = 1 if the multiplication contributes to C[i,k] via A[i,j]B[j,k].
    # Actually, the tensor T satisfies: sum_j A[i,j]B[j,k] = C[i,k].
    # The decomposition is: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j] * V[j,k,r] B[j,k]) * W[k,i,r] ? 
    # No, the standard tensor decomposition is:
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # where T[i, j, k] is 1 if the term A[i,j]B[j,k] appears in the sum for C[i,k].
    # For matmul, C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i, j, k] = 1 for all i, j, k? No.
    # T is a 3-tensor. T[i, j, k] is the coefficient of A[i,j]B[j,k] in C[i,k].
    # Wait, the tensor indices are usually (i, j, k) corresponding to (row of A, col of A / row of B, col of B).
    # The product C[i,k] depends on A[i,j] and B[j,k].
    # So the tensor T has entries T[i, j, k] = 1 if the pair (A[i,j], B[j,k]) contributes to C[i,k].
    # Since every j contributes to C[i,k] via the sum, T[i, j, k] is 1 for all i, j, k?
    # Let's check:
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This means the tensor T has T[i, j, k] = 1 for all i, j, k in {0,1}.
    # Let's verify the decomposition on a specific entry, e.g., T[0,0,0] (contribution to C[0,0] from A[0,0]B[0,0]).
    # Sum_r U[0,0,r] V[0,0,r] W[0,0,r]
    # U[0,0,r] is U[0][r] (since i=0, j=0 -> idx 0)
    # V[0,0,r] is V[0][r] (since j=0, k=0 -> idx 0)
    # W[0,0,r] is W[0][r] (since k=0, i=0 -> idx 0)
    
    # r=0: U[0][0]=1, V[0][0]=1, W[0][0]=1 -> 1
    # r=1: U[0][1]=0 -> 0
    # r=2: U[0][2]=1, V[0][2]=0 -> 0
    # r=3: U[0][3]=0 -> 0
    # r=4: U[0][4]=1, V[0][4]=0 -> 0
    # r=5: U[0][5]=-1, V[0][5]=1, W[0][5]=0 -> 0
    # r=6: U[0][6]=0 -> 0
    # Sum = 1. Correct.
    
    # Check T[0,1,0] (contribution to C[0,0] from A[0,1]B[1,0])
    # U[0,1,r] is U[1][r] (i=0, j=1 -> idx 1)
    # V[1,0,r] is V[2][r] (j=1, k=0 -> idx 2)
    # W[0,0,r] is W[0][r]
    
    # r=0: U[1][0]=0 -> 0
    # r=1: U[1][1]=0 -> 0
    # r=2: U[1][2]=0 -> 0
    # r=3: U[1][3]=0 -> 0
    # r=4: U[1][4]=1, V[2][4]=0 -> 0
    # r=5: U[1][5]=0 -> 0
    # r=6: U[1][6]=1, V[2][6]=1, W[0][6]=1 -> 1
    # Sum = 1. Correct.
    
    # Check T[1,0,1] (contribution to C[1,1] from A[1,0]B[0,1])
    # U[1,0,r] is U[2][r] (i=1, j=0 -> idx 2)
    # V[0,1,r] is V[1][r] (j=0, k=1 -> idx 1)
    # W[1,1,r] is W[3][r] (k=1, i=1 -> idx 3)
    
    # r=0: U[2][0]=0 -> 0
    # r=1: U[2][1]=1, V[1][1]=0 -> 0
    # r=2: U[2][2]=0 -> 0
    # r=3: U[2][3]=0 -> 0
    # r=4: U[2][4]=0 -> 0
    # r=5: U[2][5]=1, V[1][5]=1, W[3][5]=1 -> 1
    # r=6: U[2][6]=0 -> 0
    # Sum = 1. Correct.
    
    # The decomposition is valid.
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
