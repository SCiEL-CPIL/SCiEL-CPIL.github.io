"""Rank-minimal tensor decomposition for 4x4 x 4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    rank = U.shape[1]
    Ua = np.array(U, dtype=np.float64).reshape(n, m, rank)
    Va = np.array(V, dtype=np.float64).reshape(m, p, rank)
    Wa = np.array(W, dtype=np.float64).reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.allclose(result, 1.0, atol=1e-9)

def _strassen_2x2():
    """Strassen decomposition of 2x2 matmul, rank 7."""
    r = 7
    U = np.zeros((4, r))  # U[i*2+j, c]
    V = np.zeros((4, r))  # V[j*2+k, c]
    W = np.zeros((4, r))  # W[k*2+i, c]
    
    # M1=(A11+A22)(B11+B22): C11+=1, C22+=1
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(A21+A22)B11: C21+=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
    # M3=A11(B12-B22): C12+=1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1
    # M4=A22(B21-B11): C11+=1, C21+=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5=(A11+A12)B22: C11-=1, C12+=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6=(A21-A11)(B11+B12): C22+=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7=(A12-A22)(B21+B22): C11+=1, C22-=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[3,6]=-1
    
    return U, V, W

def _kron_4x4(U2, V2, W2):
    """Kronecker product of 2x2 decomp to get 4x4 decomp, rank 49."""
    r2 = U2.shape[1]  # 7
    r4 = r2 * r2  # 49
    U4 = np.zeros((16, r4))
    V4 = np.zeros((16, r4))
    W4 = np.zeros((16, r4))
    
    for c1 in range(r2):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i in range(4):
                i1, i2 = i // 2, i % 2
                for j in range(4):
                    j1, j2 = j // 2, j % 2
                    U4[i*4+j, c] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
            for j in range(4):
                j1, j2 = j // 2, j % 2
                for k in range(4):
                    k1, k2 = k // 2, k % 2
                    V4[j*4+k, c] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
            for k in range(4):
                k1, k2 = k // 2, k % 2
                for i in range(4):
                    i1, i2 = i // 2, i % 2
                    W4[k*4+i, c] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
    
    return U4, V4, W4

def _replicate(U4, V4, W4, p):
    """Replicate 4x4x4 decomp p times for 4x4xp."""
    r4 = U4.shape[1]
    total_rank = r4 * p
    n, m = 4, 4
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for k in range(p):
        U[:, k*r4:(k+1)*r4] = U4
        V[k*m:(k+1)*m, k*r4:(k+1)*r4] = V4
        W[k*n:(k+1)*n, k*r4:(k+1)*r4] = W4
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 125
    
    # Build Strassen 2x2
    U2, V2, W2 = _strassen_2x2()
    
    # Verify 2x2
    if not _verify(U2, V2, W2, 2, 2, 2):
        raise ValueError("Strassen 2x2 verification failed")
    
    # Build 4x4 via Kronecker
    U4, V4, W4 = _kron_4x4(U2, V2, W2)
    
    # Verify 4x4
    if not _verify(U4, V4, W4, 4, 4, 4):
        raise ValueError("Kronecker 4x4 verification failed")
    
    # Replicate 8 times
    U8, V8, W8 = _replicate(U4, V4, W4, p)
    
    # Verify full
    if not _verify(U8, V8, W8, n, m, p):
        raise ValueError("Full 4x4x8 verification failed")
    
    return {"U": U8.tolist(), "V": V8.tolist(), "W": W8.tolist()}
# EVOLVE-BLOCK-END
