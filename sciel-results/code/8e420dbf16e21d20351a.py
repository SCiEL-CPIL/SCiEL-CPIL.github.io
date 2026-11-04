import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Build rank-28 baseline using Strassen + schoolbook
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28))
    V28 = np.zeros((m * p, rank_28))
    W28 = np.zeros((p * n, rank_28))
    
    col = 0
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        U28[i1*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        U28[i2*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1
        W28[k1*n+i2, col] = 1; W28[k2*n+i2, col] = -1
        col += 1
        U28[i1*m+j1, col] = 1
        V28[j1*p+k2, col] = 1; V28[j2*p+k2, col] = -1
        W28[k2*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        U28[i2*m+j2, col] = 1
        V28[j2*p+k1, col] = 1; V28[j1*p+k1, col] = -1
        W28[k1*n+i1, col] = 1; W28[k1*n+i2, col] = 1
        col += 1
        U28[i1*m+j1, col] = 1; U28[i1*m+j2, col] = 1
        V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = -1; W28[k2*n+i1, col] = 1
        col += 1
        U28[i2*m+j1, col] = 1; U28[i1*m+j1, col] = -1
        V28[j1*p+k1, col] = 1; V28[j1*p+k2, col] = 1
        W28[k2*n+i2, col] = 1
        col += 1
        U28[i1*m+j2, col] = 1; U28[i2*m+j2, col] = -1
        V28[j2*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U28[i*m+j, col] = 1
                    V28[j*p+k, col] = 1
                    W28[k*n+i, col] = 1
                    col += 1

    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    add_schoolbook([0, 1], [0, 1], [4])
    add_schoolbook([0, 1], [2], [0, 1])
    add_schoolbook([0, 1], [2], [2, 3])
    add_schoolbook([0, 1], [2], [4])
    
    # Verify rank-28
    def verify(U, V, W, rank):
        res = np.zeros((n, m, p))
        for c in range(rank):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
        return np.allclose(res, 1.0, atol=1e-9)
    
    assert verify(U28, V28, W28, rank_28)
    
    # Tensor slicing: compute the rank lower bound
    # The tensor T[i,j,k] = 1 for all i,j,k
    # Slice along i: T[0,:,:] and T[1,:,:] are 3x5 matrices of all 1s, rank 1
    # Slice along j: T[:,0,:], T[:,1,:], T[:,2,:] are 2x5 matrices of all 1s, rank 1
    # Slice along k: T[:,:,0], ..., T[:,:,4] are 2x3 matrices of all 1s, rank 1
    # So tensor slicing gives lower bound of 1, which is not helpful
    # The actual lower bound for 2x3x5 is known to be at least 24
    
    # Try to find rank 27 by attempting to merge factors
    # Use a more systematic approach: try all pairs and check if they can be merged
    best_rank = rank_28
    best_U, best_V, best_W = U28, V28, W28
    
    for c1 in range(rank_28 - 1):
        if time.time() > deadline - 30:
            break
        for c2 in range(c1 + 1, rank_28):
            if time.time() > deadline - 30:
                break
            
            u1 = U28[:, c1]
            u2 = U28[:, c2]
            v1 = V28[:, c1]
            v2 = V28[:, c2]
            w1 = W28[:, c1]
            w2 = W28[:, c2]
            
            if np.allclose(u1, 0) or np.allclose(u2, 0):
                continue
            if np.allclose(v1, 0) or np.allclose(v2, 0):
                continue
            if np.allclose(w1, 0) or np.allclose(w2, 0):
                continue
            
            # Check if u2 = alpha_u * u1
            try:
                alpha_u = np.dot(u2, u1) / np.dot(u1, u1)
                if not np.allclose(u2, alpha_u * u1, atol=1e-9):
                    continue
            except:
                continue
            
            try:
                alpha_v = np.dot(v2, v1) / np.dot(v1, v1)
                if not np.allclose(v2, alpha_v * v1, atol=1e-9):
                    continue
            except:
                continue
            
            try:
                alpha_w = np.dot(w2, w1) / np.dot(w1, w1)
                if not np.allclose(w2, alpha_w * w1, atol=1e-9):
                    continue
            except:
                continue
            
            factor = 1 + alpha_u * alpha_v * alpha_w
            if abs(factor) > 1e-9:
                merged_U = (1 + alpha_u * alpha_v * alpha_w) * u1
                if np.allclose(merged_U, np.round(merged_U * 2) / 2, atol=1e-9):
                    new_rank = rank_28 - 1
                    U_new = np.zeros((n * m, new_rank))
                    V_new = np.zeros((m * p, new_rank))
                    W_new = np.zeros((p * n, new_rank))
                    
                    new_col = 0
                    for c in range(rank_28):
                        if c == c2:
                            continue
                        if c == c1:
                            U_new[:, new_col] = merged_U
                            V_new[:, new_col] = v1
                            W_new[:, new_col] = w1
                        else:
                            U_new[:, new_col] = U28[:, c]
                            V_new[:, new_col] = V28[:, c]
                            W_new[:, new_col] = W28[:, c]
                        new_col += 1
                    
                    if verify(U_new, V_new, W_new, new_rank):
                        best_rank = new_rank
                        best_U, best_V, best_W = U_new, V_new, W_new
                        break
        if best_rank < rank_28:
            break
    
    # If we found a better rank, return it
    if best_rank < rank_28:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Otherwise return rank-28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
