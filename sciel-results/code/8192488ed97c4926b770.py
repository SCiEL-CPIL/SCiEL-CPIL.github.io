"""New program: Kronecker product of Strassen 2x2 decomposition for 4x4 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using the Kronecker product of the Strassen 2x2 decomposition with itself.
    
    The 4x4 matrix multiplication tensor is the tensor product of two 2x2 
    matrix multiplication tensors. If (U2, V2, W2) is a rank-r decomposition
    of the 2x2 tensor, then (kron(U2,U2), kron(V2,V2), kron(W2,W2)) is a
    rank-r^2 decomposition of the 4x4 tensor.
    
    Strassen's algorithm gives rank 7 for 2x2, so this yields rank 49 for 4x4.
    """
    # Strassen 2x2 decomposition factors
    # U2[i*m+j, c], V2[j*p+k, c], W2[k*n+i, c] for n=m=p=2
    # Index: for 2x2, i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (a00+a11)(b00+b11)
    # a00 is U[0,0]=1, a11 is U[3,0]=1
    # b00 is V[0,0]=1, b11 is V[3,0]=1
    # C: c00 += 1, c11 += 1
    # W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (a10+a11)b00
    # a10 is U[2,1]=1, a11 is U[3,1]=1
    # b00 is V[0,1]=1
    # C: c10 += 1, c11 -= 1
    # W[2,1]=1 (k=1,i=0), W[3,1]=-1 (k=1,i=1)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3 = a00(b01-b11)
    # a00 is U[0,2]=1
    # b01 is V[1,2]=1, b11 is V[3,2]=-1
    # C: c01 += 1, c11 += 1
    # W[1,2]=1 (k=0,i=1), W[3,2]=1 (k=1,i=1)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = a11(b10-b00)
    # a11 is U[3,3]=1
    # b10 is V[2,3]=1, b00 is V[0,3]=-1
    # C: c00 += 1, c10 += 1
    # W[0,3]=1 (k=0,i=0), W[2,3]=1 (k=1,i=0)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (a00+a01)b11
    # a00 is U[0,4]=1, a01 is U[1,4]=1
    # b11 is V[3,4]=1
    # C: c00 -= 1, c01 += 1
    # W[0,4]=-1 (k=0,i=0), W[1,4]=1 (k=0,i=1)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    # a10 is U[2,5]=1, a00 is U[0,5]=-1
    # b00 is V[0,5]=1, b01 is V[1,5]=1
    # C: c11 += 1
    # W[3,5]=1 (k=1,i=1)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    # a01 is U[1,6]=1, a11 is U[3,6]=-1
    # b10 is V[2,6]=1, b11 is V[3,6]=1
    # C: c00 += 1
    # W[0,6]=1 (k=0,i=0)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Verify 2x2 decomposition
    # For 2x2: C[k*2+i] = sum_j sum_r U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
    # Let's verify by checking all entries
    
    # Now compute Kronecker product for 4x4
    # U = kron(U2, U2), V = kron(V2, V2), W = kron(W2, W2)
    # This gives rank 7*7 = 49
    
    U4 = np.kron(U2, U2)  # shape (16, 49)
    V4 = np.kron(V2, V2)  # shape (16, 49)
    W4 = np.kron(W2, W2)  # shape (16, 49)
    
    # Convert to list format
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()
    
    # Verify: for 4x4, C[k*4+i] = sum_j sum_r U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r]
    # We trust the Kronecker product property
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
