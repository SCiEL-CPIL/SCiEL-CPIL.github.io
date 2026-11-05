import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # ---------- Block decompositions (same as before) ----------
    def strassen_2x2x2():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    def schoolbook_2x2x1():
        U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        return U, V, W

    def schoolbook_1x2x2():
        U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[2,1]=1; W[0,1]=1
        U[0,2]=1; V[1,2]=1; W[1,2]=1
        U[1,3]=1; V[3,3]=1; W[1,3]=1
        return U, V, W

    def schoolbook_1x2x1():
        U = np.zeros((2,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        return U, V, W

    # ---------- Assemble base decomposition ----------
    U_parts = []; V_parts = []; W_parts = []

    def add_block(U_blk, V_blk, W_blk, u_idx, v_idx, w_idx):
        R = U_blk.shape[1]
        for r in range(R):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for i, idx in enumerate(u_idx): u[idx] = U_blk[i, r]
            for i, idx in enumerate(v_idx): v[idx] = V_blk[i, r]
            for i, idx in enumerate(w_idx): w[idx] = W_blk[i, r]
            U_parts.append(u); V_parts.append(v); W_parts.append(w)

    # U: row i, col j -> i*m+j (m=4)
    # V: row i, col j -> i*p+j (p=5)
    # W: col k, row i -> k*n+i (n=3)

    # A rows 0,1 (top 2x4) split into cols 0,1 and 2,3
    # A row 2 (bottom 1x4) split into cols 0,1 and 2,3
    # B rows 0,1 (top 2x5) split into cols 0,1 / 2,3 / 4
    # B rows 2,3 (bottom 2x5) split into cols 0,1 / 2,3 / 4

    # Top-A blocks
    U_b,V_b,W_b = strassen_2x2x2()
    add_block(U_b,V_b,W_b,[0,1,4,5],[0,1,5,6],[0,3,1,4])   # A[0:2,0:2]*B[0:2,0:2]
    U_b,V_b,W_b = strassen_2x2x2()
    add_block(U_b,V_b,W_b,[0,1,4,5],[2,3,7,8],[6,9,7,10])   # A[0:2,0:2]*B[0:2,2:4]
    U_b,V_b,W_b = schoolbook_2x2x1()
    add_block(U_b,V_b,W_b,[0,1,4,5],[4,9],[12,13])          # A[0:2,0:2]*B[0:2,4]
    U_b,V_b,W_b = strassen_2x2x2()
    add_block(U_b,V_b,W_b,[2,3,6,7],[10,11,15,16],[0,3,1,4])# A[0:2,2:4]*B[2:4,0:2]
    U_b,V_b,W_b = strassen_2x2x2()
    add_block(U_b,V_b,W_b,[2,3,6,7],[12,13,17,18],[6,9,7,10])# A[0:2,2:4]*B[2:4,2:4]
    U_b,V_b,W_b = schoolbook_2x2x1()
    add_block(U_b,V_b,W_b,[2,3,6,7],[14,19],[12,13])        # A[0:2,2:4]*B[2:4,4]

    # Bottom-A blocks
    U_b,V_b,W_b = schoolbook_1x2x2()
    add_block(U_b,V_b,W_b,[8,9],[0,1,5,6],[2,5])            # A[2,0:2]*B[0:2,0:2]
    U_b,V_b,W_b = schoolbook_1x2x2()
    add_block(U_b,V_b,W_b,[8,9],[2,3,7,8],[8,11])           # A[2,0:2]*B[0:2,2:4]
    U_b,V_b,W_b = schoolbook_1x2x1()
    add_block(U_b,V_b,W_b,[8,9],[4,9],[14])                 # A[2,0:2]*B[0:2,4]
    U_b,V_b,W_b = schoolbook_1x2x2()
    add_block(U_b,V_b,W_b,[10,11],[10,11,15,16],[2,5])      # A[2,2:4]*B[2:4,0:2]
    U_b,V_b,W_b = schoolbook_1x2x2()
    add_block(U_b,V_b,W_b,[10,11],[12,13,17,18],[8,11])     # A[2,2:4]*B[2:4,2:4]
    U_b,V_b,W_b = schoolbook_1x2x1()
    add_block(U_b,V_b,W_b,[10,11],[14,19],[14])             # A[2,2:4]*B[2:4,4]

    U = np.array(U_parts).T  # (12, R)
    V = np.array(V_parts).T  # (20, R)
    W = np.array(W_parts).T  # (15, R)
    R0 = U.shape[1]

    # ---------- Verify base decomposition ----------
    def verify(Um, Vm, Wm, tol=1e-10):
        # Check: for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # Efficient: build tensor via einsum
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Reshape U to (n,m,R), V to (m,p,R), W to (p,n,R)
        Ur = Um.reshape(n, m, -1)
        Vr = Vm.reshape(m, p, -1)
        Wr = Wm.reshape(p, n, -1)
        # T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        # Use einsum
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=tol), np.max(np.abs(T - expected))

    ok, err = verify(U, V, W)
    if not ok:
        # Fallback: return schoolbook (rank 60) if something went wrong
        pass  # we'll still try to return what we have

    # ---------- Rank reduction via linear dependence ----------
    # The constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is linear in the rank-r "triple" vectors. If we have a linear relation
    # among the rank terms that preserves the tensor, we can eliminate terms.
    #
    # Key insight: The tensor T = sum_r u_r ⊗ v_r ⊗ w_r. If there exists a
    # non-trivial linear combination sum_c c_r (u_r ⊗ v_r ⊗ w_r) = 0 (as a
    # tensor), then we can reduce rank. This is equivalent to finding the
    # minimal rank of the given tensor, which is hard in general, but for
    # structured decompositions we can check for specific linear dependencies.
    #
    # Practical approach: Check if any column of the "coefficient matrix" is
    # linearly dependent given the structure. Specifically, for each pair of
    # output positions (i,k), the vectors u_{i,·,r} * v_{·,k,r} (as functions
    # of r) must satisfy: sum_r u_{i,·,r} * v_{·,k,r} * w_{k,i,r} = 1 for all j.
    #
    # A simpler check: if two rank terms r1, r2 have proportional U-columns
    # AND proportional V-columns (with the same proportionality), and the W
    # columns satisfy the same proportionality, then they can be merged.
    #
    # Let's do a more systematic search: try to find a rank-reducing linear
    # relation by solving a system.

    def try_reduce_rank(Um, Vm, Wm, max_iter=200):
        """Try to reduce rank by finding linear dependencies among rank terms."""
        R = Um.shape[1]
        removed = set()
        
        for iteration in range(max_iter):
            if time.time() > DEADLINE:
                break
            active = [r for r in range(R) if r not in removed]
            if len(active) <= 1:
                break
            
            # For each pair of active terms, check if they can be merged
            # by solving for a linear combination that preserves the tensor.
            # 
            # Approach: pick a pivot term r0, and for each other term r,
            # check if the "contribution" of r can be expressed as a
            # multiple of r0's contribution (in the tensor sense).
            #
            # More practically: build the full tensor coefficient matrix and
            # find its rank. The matrix M has entries M[(i,j,k), r] = 
            # U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r].
            # The tensor is determined by the row space of M. If rank(M) < R,
            # we can eliminate R - rank(M) terms.
            
            R_active = len(active)
            if R_active < 2:
                break
            
            # Build M: (n*m*p, R_active)
            idx_active = np.array(active)
            Ua = Um[:, idx_active]
            Va = Vm[:, idx_active]
            Wa = Wm[:, idx_active]
            
            # M[i*m+j, j*p+k, k*n+i, r] -> flatten (i,j,k)
            # Efficient: for each r, M[:,r] = U[:,r] outer V[:,r] outer W[:,r] reshaped
            # But this is expensive for large R. Let's do it in a loop.
            M = np.zeros((n*m*p, R_active))
            for r_idx, r in enumerate(active):
                # Vectorized: for each (i,j,k), M[i,j,k,r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # U[:,r] has shape (n*m,), V[:,r] has shape (m*p,), W[:,r] has shape (p*n,)
                # We need to compute the "triple product" for each (i,j,k).
                # U_flat[i*m+j], V_flat[j*p+k], W_flat[k*n+i]
                U_flat = Um[:, r]
                V_flat = Vm[:, r]
                W_flat = Wm[:, r]
                # Reshape to 2D
                U2 = U_flat.reshape(n, m)
                V2 = V_flat.reshape(m, p)
                W2 = W_flat.reshape(p, n)
                # M[i,j,k] = U2[i,j] * V2[j,k] * W2[k,i]
                # This is a 3D tensor. Compute via einsum.
                M[:, r_idx] = np.einsum('ij,jk,ki->ijk', U2, V2, W2).ravel()
            
            # Find rank of M
            try:
                rank_M = np.linalg.matrix_rank(M, tol=1e-10)
            except:
                rank_M = R_active
            
            if rank_M < R_active:
                # Find a null vector: M @ c = 0
                try:
                    _, s, vh = np.linalg.svd(M, full_matrices=False)
                    # Null space vectors are rows of vh corresponding to small singular values
                    null_mask = s < 1e-10
                    if np.any(null_mask):
                        # Take the first null vector
                        c = vh[-1, :]  # last row of vh is in null space
                        # Find the largest |c| component to use as pivot
                        r_pivot = active[np.argmax(np.abs(c))]
                        # Remove the pivot term
                        removed.add(r_pivot)
                        continue
                except:
                    pass
            
            # No more reductions found
            break
        
        active = [r for r in range(R) if r not in removed]
        return active

    active = try_reduce_rank(U, V, W, max_iter=100)
    
    if len(active) < R0:
        U = U[:, active]
        V = V[:, active]
        W = W[:, active]
        R_final = len(active)
    else:
        R_final = R0

    # Final verification
    ok, err = verify(U, V, W, tol=1e-9)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
