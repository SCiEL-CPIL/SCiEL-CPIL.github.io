import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2 * 2x2 -> 2x2 via Strassen, rank 7.
    U: (4,7) rows indexed by i*2+j
    V: (4,7) rows indexed by j*2+k
    W: (4,7) rows indexed by k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1  # C00 -= M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1  # C00 += M5, C11 -= M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = -1  # C11 -= M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1  # C00 -= M7, C01 += M7
    
    return U, V, W

def _build_4x4_strassen():
    """Build 4x4 * 4x4 -> 4x4 via Kronecker of Strassen, rank 49.
    U: (16, 49) rows i*4+j
    V: (16, 49) rows j*4+k
    W: (16, 49) rows k*4+i
    """
    U2, V2, W2 = _strassen_2x2()
    r = 49
    U = np.zeros((16, r))
    V = np.zeros((16, r))
    W = np.zeros((16, r))
    
    # We can use kron, but the index mapping needs to be careful.
    # U2 is 4x7. U is 16x49.
    # U[i*4+j, c1*7+c2] = U2[i*2+j1, c1] * U2[i2*2+j2, c2] where i=2*i1+i2, j=2*j1+j2.
    # This is exactly the Kronecker product structure if we reshape appropriately.
    
    # Let's just loop, it's fast enough for 49 terms.
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U part
            for i1 in range(2):
                for j1 in range(2):
                    u_val1 = U2[i1 * 2 + j1, c1]
                    if u_val1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u_val2 = U2[i2 * 2 + j2, c2]
                            if u_val2 == 0: continue
                            u_idx = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U[u_idx, c] += u_val1 * u_val2
                            
            # V part
            for j1 in range(2):
                for k1 in range(2):
                    v_val1 = V2[j1 * 2 + k1, c1]
                    if v_val1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v_val2 = V2[j2 * 2 + k2, c2]
                            if v_val2 == 0: continue
                            v_idx = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                            V[v_idx, c] += v_val1 * v_val2
                            
            # W part
            for k1 in range(2):
                for i1 in range(2):
                    w_val1 = W2[k1 * 2 + i1, c1]
                    if w_val1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w_val2 = W2[k2 * 2 + i2, c2]
                            if w_val2 == 0: continue
                            w_idx = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                            W[w_idx, c] += w_val1 * w_val2
                            
    return U, V, W

def _build_decomposition():
    """Build 4x4 * 4x8 -> 4x8 decomposition.
    Uses two 4x4*4x4 blocks.
    """
    n, m, p = 4, 4, 8
    U4, V4, W4 = _build_4x4_strassen()
    r4 = U4.shape[1] # 49
    r = 2 * r4 # 98
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # First 4x4 * 4x4: B columns 0-3, C columns 0-3
    U[:, 0:r4] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:r4] = V4[j * 4 + k, 0:r4]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:r4] = W4[k * 4 + i, 0:r4]
            
    # Second 4x4 * 4x4: B columns 4-7, C columns 4-7
    U[:, r4:2*r4] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j * 8 + k, r4:2*r4] = V4[j * 4 + (k - 4), 0:r4]
    for k in range(4, 8):
        for i in range(4):
            W[k * 4 + i, r4:2*r4] = W4[(k - 4) * 4 + i, 0:r4]
            
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8):
    """Verify tensor decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1"""
    r = U.shape[1]
    # Vectorized check
    # Create tensors of size (n, m, p, r)
    # U is (nm, r) -> reshape to (n, m, 1, r)
    U_t = U.reshape(n, m, 1, r)
    # V is (mp, r) -> reshape to (1, m, p, r)
    V_t = V.reshape(1, m, p, r)
    # W is (pn, r) -> reshape to (1, 1, p, r) ... wait, W index is k*n+i.
    # W[k, i] is at row k*n+i.
    # So W_t should be (1, 1, p, n) broadcast to (n, m, p, r)?
    # Let's just do the sum manually to be safe, or use einsum.
    
    # Einsum:
    # U[i,j,c], V[j,k,c], W[k,i,c]
    # U is (n, m, r)
    # V is (m, p, r)
    # W is (p, n, r)
    
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    # Sum over c: U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want result[i,j,k] to be 1.
    # np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    # Note: W index is k*n+i, so W_3d[k, i, c].
    # The contraction is over c, j, k? No.
    # We sum over c. The indices i, j, k are free.
    # U depends on i, j. V depends on j, k. W depends on k, i.
    # So we need to sum over c, and the result is a tensor T[i, j, k].
    # But j and k are also summed? No, in the tensor definition, T_{ijk} = sum_c ...
    # So the result is a 3D tensor of size (n, m, p).
    
    # However, einsum with repeated indices in input sums over them.
    # 'ijc,jkc,kic->ijk' sums over c. It does NOT sum over j or k.
    # Wait, j appears in U and V. If I write 'ijc,jkc,kic->ijk', numpy will sum over c, but j and k are free indices in the output.
    # Is this correct?
    # U[i,j,c] * V[j,k,c] * W[k,i,c].
    # Yes, j is shared between U and V, k between V and W, i between W and U.
    # But in the sum, we only sum over c. The indices i, j, k define the specific tensor element.
    # So the formula is correct.
    
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        # Check if all elements are 1
        if not np.allclose(result, 1.0, atol=1e-10):
            return False
        return True
    except Exception:
        return False

def _schoolbook(n=4, m=4, p=8):
    """Schoolbook decomposition, rank n*m*p."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    try:
        U, V, W = _build_decomposition()
        
        if time.time() > deadline:
            U, V, W = _schoolbook(n, m, p)
        elif not _verify(U, V, W, n, m, p):
            U, V, W = _schoolbook(n, m, p)
        else:
            # If we have time, we could try to reduce rank, but 98 is likely minimal.
            # We just return it.
            pass
            
        # Final verification
        if not _verify(U, V, W, n, m, p):
            U, V, W = _schoolbook(n, m, p)
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
