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
    # Tensor check: for each (i,k,j) output coefficient equals sum_r U[i*m+j,r]*V[k*m? ...]
    # U indexed by (i, j) -> i*m+j. V indexed by (k, j) -> k*p+j. W indexed by (k, i) -> k*n+i.
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                expected = 1.0
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - expected) > 1e-6:
                    return False
    return True

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 140.0
    U, V, W = _build(98)
    assert _verify(U, V, W, n, m, p), "base decomposition invalid"

    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    R = U.shape[1]
    
    def get_tensor_matrix(U, V, W):
        # Returns matrix of size (n*p*m) x R
        M = np.zeros((n*p*m, U.shape[1]))
        idx = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(U.shape[1]):
                        M[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    idx += 1
        return M

    def is_half_integer(x, tol=1e-6):
        # Check if x is close to a half-integer (k/2)
        y = x * 2
        return abs(y - round(y)) < tol

    # Greedy removal with valid scaling
    for _ in range(R):
        if time.time() > deadline - 5.0:
            break
        
        # Compute tensor matrix
        M = get_tensor_matrix(U, V, W)
        
        # Find null space of M (dependencies)
        # We want c such that M c = 0
        try:
            _, s, Vt = np.linalg.svd(M, full_matrices=True)
        except np.linalg.LinAlgError:
            break
            
        # The null space is spanned by the last rows of Vt if singular values are 0
        # Find the smallest singular value
        min_sv = s[-1]
        
        # If min_sv is not close to 0, no dependency
        if min_sv > 1e-5:
            break
            
        # Get the null vector (last row of Vt)
        c = Vt[-1, :]
        
        # Find the index with the largest absolute coefficient
        r0 = np.argmax(np.abs(c))
        
        # Check if c[r0] is non-zero
        if abs(c[r0]) < 1e-6:
            break
            
        # We have sum_r c_r t_r = 0
        # t_r0 = - sum_{r != r0} (c_r / c_r0) t_r
        # New sum = sum_{r != r0} (1 + c_r/c_r0) t_r
        # Scale factors s_r = (1 + c_r/c_r0)^(1/3)
        
        keep = [q for q in range(R) if q != r0]
        scales = []
        valid = True
        for r in keep:
            factor = 1.0 + c[r] / c[r0]
            # Check if factor is positive (for real cube root)
            if factor < 0:
                # Cube root of negative is negative, which is fine for real numbers
                # But we need to check half-integer constraint
                pass
            
            s_val = np.cbrt(factor)
            
            # Check if s_val is close to a half-integer
            if not is_half_integer(s_val):
                valid = False
                break
            scales.append(s_val)
            
        if not valid:
            # Try next null vector if possible, or break
            # For simplicity, break
            break
            
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
        
        # Verify
        if not _verify(U, V, W, n, m, p):
            # Rollback? We don't have a stack. 
            # This implies the scaling was incorrect or floating point error.
            # Since we checked half-integer, it should be exact.
            # If it fails, break.
            break

    if time.time() < deadline - 2.0:
        # final verify
        if not _verify(U, V, W, n, m, p):
            # Fallback to original if verification fails
            U, V, W = _build(98)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
