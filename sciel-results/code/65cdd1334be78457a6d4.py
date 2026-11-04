import numpy as np
import time

def _strassen_2x2x2():
    # Standard Strassen 2x2x2, rank 7. Indexing: 0=(0,0),1=(0,1),2=(1,0),3=(1,1)
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    return U2, V2, W2

def _block_specs():
    A_specs = [
        [(1,0),(1,3)], [(1,2),(1,3)], [(1,0)], [(1,3)],
        [(1,0),(1,1)], [(1,2),(-1,0)], [(1,1),(-1,3)],
    ]
    B_specs = [
        [(1,0),(1,3)], [(1,0)], [(1,1),(-1,3)], [(1,2),(-1,0)],
        [(1,3)], [(1,0),(1,1)], [(1,2),(1,3)],
    ]
    C_specs = [
        [(1,0),(1,3)], [(1,2),(-1,3)], [(1,1),(1,3)], [(1,0),(1,2)],
        [(-1,0),(1,1)], [(1,3)], [(1,0)],
    ]
    return A_specs, B_specs, C_specs

def _build(rank=98):
    n, m, p = 4, 4, 8
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()
    A_specs, B_specs, C_specs = _block_specs()
    for half in range(2):
        for kp in range(7):
            for ms in range(7):
                col = half*49 + kp*7 + ms
                for ca, bi in A_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c
                            U[gi*4+gj, col] += ca * U2[r*2+c, ms]
                for cb, bi in B_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            V[gi*8+gj, col] += cb * V2[r*2+c, ms]
                for cc, bi in C_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            W[gj*4+gi, col] += cc * W2[r*2+c, ms]
    return U, V, W

def _verify(U, V, W, n, m, p):
    # Vectorized verification
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    rank = U.shape[1]
    # Reshape to 3D tensors for easier indexing?
    # U[i, j, r], V[j, k, r], W[k, i, r]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # We want to check if sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    # This is a tensor contraction.
    # Let's do it loop over i, k and vectorize over j, r
    for i in range(n):
        for k in range(p):
            # U[i, :, :] is (m, rank)
            # V[:, k, :] is (m, rank)
            # W[k, i, :] is (rank,)
            # Sum over j and r: sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # This is not a simple dot product because of the j index.
            # Let M be a matrix where M[j, r] = U[i,j,r] * V[j,k,r]
            # Then we want sum_r W[k,i,r] * (sum_j M[j,r])
            # sum_j U[i,j,r] * V[j,k,r] is the r-th component of the Hadamard product sum?
            # No, it's the element-wise product summed over j.
            # Let H = U[i, :, :] * V[:, k, :]  (shape m x rank)
            # Sum over j: S = np.sum(H, axis=0) (shape rank,)
            # Then dot product with W[k, i, :]
            H = U_3d[i, :, :] * V_3d[:, k, :]
            S = np.sum(H, axis=0)
            val = np.dot(S, W_3d[k, i, :])
            if abs(val - 1.0) > 1e-6:
                return False
    return True

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 140.0
    
    U, V, W = _build(98)
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    R = U.shape[1]
    
    def get_tensor_matrix(U, V, W):
        # Returns matrix of size (n*p*m) x R
        # M[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # idx corresponds to (i, k, j)
        M = np.zeros((n*p*m, U.shape[1]))
        idx = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    M[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    idx += 1
        return M

    def is_half_integer(x, tol=1e-6):
        y = x * 2
        return abs(y - round(y)) < tol

    # Greedy removal with valid scaling
    for _ in range(R):
        if time.time() > deadline - 5.0:
            break
        
        # Compute tensor matrix
        M = get_tensor_matrix(U, V, W)
        
        # Find null space of M
        try:
            _, s, Vt = np.linalg.svd(M, full_matrices=True)
        except np.linalg.LinAlgError:
            break
            
        min_sv = s[-1]
        if min_sv > 1e-5:
            break
            
        # Try the last few null vectors (smallest singular values)
        found_reduction = False
        for null_idx in range(len(s)-1, max(len(s)-5, -1), -1):
            if s[null_idx] > 1e-5:
                continue
                
            c = Vt[null_idx, :]
            r0 = np.argmax(np.abs(c))
            if abs(c[r0]) < 1e-6:
                continue
                
            keep = [q for q in range(R) if q != r0]
            scales = []
            valid = True
            for r in keep:
                factor = 1.0 + c[r] / c[r0]
                if factor < 0:
                    s_val = np.cbrt(factor)
                else:
                    s_val = factor ** (1.0/3.0)
                
                if not is_half_integer(s_val):
                    valid = False
                    break
                scales.append(s_val)
            
            if valid:
                # Apply scaling
                for i, r in enumerate(keep):
                    U[:, r] *= scales[i]
                    V[:, r] *= scales[i]
                    W[:, r] *= scales[i]
                
                # Drop r0
                U = U[:, keep]
                V = V[:, keep]
                W = W[:, keep]
                R -= 1
                found_reduction = True
                break
                
        if not found_reduction:
            break
            
        if not _verify(U, V, W, n, m, p):
            # Rollback not possible easily, so break
            break

    if time.time() < deadline - 2.0:
        if not _verify(U, V, W, n, m, p):
            U, V, W = _build(98)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
