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
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    R = U.shape[1]
    
    # Precompute flattened tensor vectors for faster dependency checks
    # vec(T_r) has length n*p*m = 128
    # Index: (i, k, j) -> i*p*m + k*m + j
    def get_tensor_vecs(U, V, W, R):
        vecs = np.zeros((R, n*p*m))
        for r in range(R):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            # T_r[i, j, k] = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
            # We need to map this to a flat vector indexed by (i, k, j)
            # idx = i*p*m + k*m + j
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        idx = i*p*m + k*m + j
                        vecs[r, idx] = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        return vecs

    vecs = get_tensor_vecs(U, V, W, R)
    
    removed = []
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        current_R = R - len(removed)
        # Re-index vecs to only active ones
        active_indices = [i for i in range(R) if i not in removed]
        active_vecs = vecs[active_indices]
        
        # Find a removable vector
        # We check if any vector is in the span of the others
        # This is equivalent to checking if the rank of the matrix of all active vectors
        # is less than the number of active vectors? No, that's for the span of all.
        # We need to check if a specific vector is in the span of the rest.
        
        # Efficient check: 
        # For each r, compute the rank of [active_vecs without r] and [active_vecs without r, r].
        # If rank is same, r is removable.
        
        # Optimization: Use SVD of the full active matrix.
        # If the matrix has rank K < current_R, there are dependencies.
        # We can find one dependent vector.
        
        try:
            _, s, Vt = np.linalg.svd(active_vecs, full_matrices=False)
        except np.linalg.LinAlgError:
            break
            
        # Tolerance for singular values
        tol = 1e-9 * max(active_vecs.shape)
        rank_svd = sum(1 for sv in s if sv > tol)
        
        if rank_svd < current_R:
            # There is at least one dependent vector.
            # Identify which one.
            # The right singular vectors corresponding to zero singular values are in the null space.
            # Any vector in the null space of the matrix is a linear combination of the rows?
            # No, we want a column to be a combination of others.
            
            # Simpler: Iterate and test removal.
            # Since current_R is small (< 100), this is fast enough.
            for idx_in_active, orig_idx in enumerate(active_indices):
                # Test if active_vecs[idx_in_active] is in span of others
                others = np.delete(active_vecs, idx_in_active, axis=0)
                if others.shape[0] == 0:
                    continue
                # Check rank of others vs rank of [others, vec]
                # Use lstsq to see if we can represent vec using others
                # Actually, just check if the rank increases when adding vec.
                try:
                    _, s_others, _ = np.linalg.svd(others, full_matrices=False)
                    rank_others = sum(1 for sv in s_others if sv > tol)
                    
                    # If rank_others == rank_svd (which is rank of all), then vec is in span.
                    # Wait, rank_svd is the rank of the full set.
                    # If removing a vector doesn't change the rank, it is removable.
                    if rank_others == rank_svd:
                        removed.append(orig_idx)
                        changed = True
                        # Update active lists
                        active_indices.remove(orig_idx)
                        active_vecs = np.delete(active_vecs, idx_in_active, axis=0)
                        break # Restart loop to re-evaluate
                except np.linalg.LinAlgError:
                    pass

    # Construct final matrices
    if removed:
        keep = [i for i in range(R) if i not in removed]
        U_final = U[:, keep]
        V_final = V[:, keep]
        W_final = W[:, keep]
    else:
        U_final = U
        V_final = V
        W_final = W
        
    # Final verification
    if time.time() < deadline - 1.0:
        assert _verify(U_final, V_final, W_final, n, m, p), "final decomposition invalid"
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
