import time
import numpy as np

START = time.time()
DEADLINE = START + 130.0

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _kron_decompose(U1, V1, W1, U2, V2, W2):
    """
    Compute the Kronecker product of two tensor decompositions.
    If T1 = sum u1_k v1_k w1_k and T2 = sum u2_k v2_k w2_k,
    then T1 x T2 = sum (u1_k x u2_k) (v1_k x v2_k) (w1_k x w2_k).
    """
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 * rank2
    
    dim_u1 = U1.shape[0]
    dim_v1 = V1.shape[0]
    dim_w1 = W1.shape[0]
    
    dim_u2 = U2.shape[0]
    dim_v2 = V2.shape[0]
    dim_w2 = W2.shape[0]
    
    U_out = np.zeros((dim_u1 * dim_u2, total_rank), dtype=np.int16)
    V_out = np.zeros((dim_v1 * dim_v2, total_rank), dtype=np.int16)
    W_out = np.zeros((dim_w1 * dim_w2, total_rank), dtype=np.int16)
    
    for k1 in range(rank1):
        for k2 in range(rank2):
            k = k1 * rank2 + k2
            U_out[:, k] = np.kron(U1[:, k1], U2[:, k2])
            V_out[:, k] = np.kron(V1[:, k1], V2[:, k2])
            W_out[:, k] = np.kron(W1[:, k1], W2[:, k2])
            
    return U_out, V_out, W_out

def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    U4, V4, W4 = _kron_decompose(U2, V2, W2, U2, V2, W2)
    return U4, V4, W4

def construct():
    """
    Construct rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses the direct sum of two rank-49 Strassen-based decompositions.
    """
    if time.time() > DEADLINE:
        return {"U": [], "V": [], "W": []} # Should not happen

    n, m, p = 4, 4, 8
    U4, V4, W4 = _build_4x4x4_tensors()
    
    total_rank = 98
    U = np.zeros((n * m, total_rank), dtype=np.int16)
    V = np.zeros((m * p, total_rank), dtype=np.int16)
    W = np.zeros((p * n, total_rank), dtype=np.int16)
    
    # Block 0: Output columns 0-3 (k in [0, 4))
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, :]
            
    # Block 1: Output columns 4-7 (k in [4, 8))
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j*8 + k, 49:98] = V4[j*4 + (k-4), :]
    for k in range(4, 8):
        for i in range(4):
            W[k*4 + i, 49:98] = W4[(k-4)*4 + i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
