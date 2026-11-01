"""
Tensor decomposition for 2x2 matrix multiplication.
Implements Strassen's algorithm, which achieves the theoretical minimum rank of 7.
"""
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-7 tensor decomposition for 2x2 matrix multiplication (Strassen).
    
    The decomposition factors the multiplication tensor T into U, V, W such that:
    T = sum_{c=1}^{rank} u_c v_c^T w_c^T
    
    Where:
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    The indices follow the convention:
    U[i*m + j, c] corresponds to A[i, j]
    V[j*p + k, c] corresponds to B[j, k]
    W[k*n + i, c] corresponds to C[i, k]
    
    For n=m=p=2:
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01], [b10, b11]]
    C = A * B
    
    Strassen's 7 products:
    M1 = (a00 + a11)(b00 + b11)
    M2 = (a10 + a11)b00
    M3 = a00(b01 - b11)
    M4 = a11(b10 - b00)
    M5 = (a00 + a01)b11
    M6 = (a10 - a00)(b00 + b01)
    M7 = (a01 - a11)(b10 + b11)
    
    Reconstruction:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index mapping:
    # A[i,j] -> U[i*m + j]
    # B[j,k] -> V[j*p + k]
    # C[i,k] -> W[k*n + i]
    
    # Product 1: M1 = (a00 + a11)(b00 + b11)
    # a00: U[0,0]=1, a11: U[3,0]=1
    # b00: V[0,0]=1, b11: V[3,0]=1
    # C00: W[0,0]=1, C11: W[3,0]=1
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # Product 2: M2 = (a10 + a11)b00
    # a10: U[2,1]=1, a11: U[3,1]=1
    # b00: V[0,1]=1
    # C10: W[1,1]=1, C11: W[3,1]=-1
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1; W[3, 1] += -1
    
    # Product 3: M3 = a00(b01 - b11)
    # a00: U[0,2]=1
    # b01: V[1,2]=1, b11: V[3,2]=-1
    # C01: W[2,2]=1, C11: W[3,2]=1
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] += -1
    W[2, 2] += 1; W[3, 2] += 1
    
    # Product 4: M4 = a11(b10 - b00)
    # a11: U[3,3]=1
    # b10: V[2,3]=1, b00: V[0,3]=-1
    # C00: W[0,3]=1, C10: W[1,3]=1
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] += -1
    W[0, 3] += 1; W[1, 3] += 1
    
    # Product 5: M5 = (a00 + a01)b11
    # a00: U[0,4]=1, a01: U[1,4]=1
    # b11: V[3,4]=1
    # C00: W[0,4]=-1, C01: W[2,4]=1
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] += -1; W[2, 4] += 1
    
    # Product 6: M6 = (a10 - a00)(b00 + b01)
    # a10: U[2,5]=1, a00: U[0,5]=-1
    # b00: V[0,5]=1, b01: V[1,5]=1
    # C11: W[3,5]=1
    U[2, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # Product 7: M7 = (a01 - a11)(b10 + b11)
    # a01: U[1,6]=1, a11: U[3,6]=-1
    # b10: V[2,6]=1, b11: V[3,6]=1
    # C00: W[0,6]=1
    U[1, 6] += 1; U[3, 6] += -1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    # Convert to lists as per typical interface expectation, though numpy arrays are often accepted
    # The prompt specifies "factor matrices of shapes...", lists are safer for strict typing if not specified as array
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
