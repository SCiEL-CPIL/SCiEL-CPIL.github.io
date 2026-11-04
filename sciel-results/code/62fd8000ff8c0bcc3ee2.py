import numpy as np
import time
from fractions import Fraction

def _strassen_2x2x2():
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    return U2, V2, W2

def _build(rank=98):
    n, m, p = 4, 4, 8
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()
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

def _verify(U, V, W, n, m, p, tol=1e-6):
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_M(U, V, W, n, m, p):
    rank = U.shape[1]
    N = n*p*m
    M = np.zeros((N, rank))
    idx = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                M[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                idx += 1
    return M

def _is_half_integer(x, tol=1e-4):
    y = x * 2
    return abs(y - round(y)) < tol

def _is_rational_small(x, tol=1e-6):
    """Check if x is close to a rational with small denominator (up to 12)"""
    for d in range(1, 13):
        n_val = round(x * d)
        if abs(x - n_val / d) < tol:
            return True, n_val, d
    return False, 0, 1

def _try_reduce(U, V, W, n, m, p, deadline, max_rounds=100):
    U = U.astype(float).copy(); V = V.astype(float).copy(); W = W.astype(float).copy()
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for rnd in range(max_rounds):
        if time.time() > deadline - 15.0:
            break
        if R <= 1:
            break

        M = _build_M(U, V, W, n, m, p)
        N = M.shape[0]
        
        # Find columns that are in the span of others using SVD-based approach
        # Compute rank of M
        try:
            svals = np.linalg.svd(M, compute_uv=False)
        except np.linalg.LinAlgError:
            break
        
        # Find numerical rank
        rank_M = np.sum(svals > 1e-8 * max(1.0, svals[0]))
        
        if rank_M >= R:
            break  # No more dependencies to find
        
        # Find which columns are dependent
        # Use QR with column pivoting to identify independent columns
        try:
            Q, R_mat, P = np.linalg.qr(M, mode='complete', pivoting=True)
        except np.linalg.LinAlgError:
            break
        
        # P tells us the permutation; first rank_M columns of M[:, P] are independent
        # The remaining R - rank_M columns are dependent
        independent_cols = P[:rank_M]
        dependent_cols = P[rank_M:]
        
        if len(dependent_cols) == 0:
            break
        
        # For each dependent column, find its representation in terms of independent columns
        # M[:, dep] = M[:, indep] @ alpha
        M_indep = M[:, independent_cols]
        M_dep = M[:, dependent_cols]
        
        try:
            alpha, _, _, _ = np.linalg.lstsq(M_indep, M_dep, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        # For each dependent column d, we have M[:, d] = sum_i alpha[i,d] * M[:, indep[i]]
        # We want to eliminate column d by expressing it as a combination
        # The trick: if alpha[i,d] = s^3 for some half-integer s, we can scale
        # Actually, the proper way: if column d is a linear combination of others,
        # we can remove it if we can adjust the other columns.
        # 
        # Standard approach: if M[:, d] = sum_i c_i * M[:, indep_i], then we can
        # set new_U[:, indep_i] += c_i * new_U[:, d] (and similarly for V, W)
        # and then drop column d. But this doesn't work directly because the
        # decomposition requires U*V*W products.
        #
        # Correct approach: The tensor decomposition says T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r, v_r, w_r are the r-th columns of U, V, W.
        # If u_d ⊗ v_d ⊗ w_d is in the span of {u_i ⊗ v_i ⊗ w_i : i ≠ d},
        # we can eliminate it. But checking this is hard.
        #
        # Simpler approach that works: if column d of M is exactly zero, drop it.
        # Or if M[:, d] = c * M[:, e] for some e, we can combine.
        #
        # Let's try: for each dependent column, check if it's a scalar multiple of
        # a single independent column.
        
        eliminated = False
        for d in dependent_cols:
            if time.time() > deadline - 10.0:
                break
            
            # Check if M[:, d] is a scalar multiple of some M[:, indep_i]
            for i_idx, i_col in enumerate(independent_cols):
                if np.linalg.norm(M[:, d]) < 1e-10:
                    # Zero column, just drop
                    break
                
                # Check if M[:, d] ≈ c * M[:, i_col]
                if np.linalg.norm(M[:, i_col]) < 1e-10:
                    continue
                
                c = np.dot(M[:, d], M[:, i_col]) / (np.dot(M[:, i_col], M[:, i_col]))
                residual = M[:, d] - c * M[:, i_col]
                if np.linalg.norm(residual) < 1e-8 * max(1.0, np.linalg.norm(M[:, d])):
                    # M[:, d] = c * M[:, i_col]
                    # This means u_d ⊗ v_d ⊗ w_d = c * u_i ⊗ v_i ⊗ w_i
                    # We can absorb this into column i by:
                    # new term for i: (1 + something) but we need to be careful
                    # Actually, if the entire tensor term is proportional, we can
                    # just drop d and scale i appropriately. But that changes the
                    # sum. Instead, we should add the contribution of d to i.
                    #
                    # The decomposition is T = sum_r u_r v_r w_r
                    # If u_d v_d w_d = c * u_i v_i w_i, then
                    # T = ... + (1+c) u_i v_i w_i + ... (dropping d)
                    # But we can't just multiply u_i by (1+c) because that would
                    # scale all products involving i.
                    #
                    # This approach is flawed. Let's use a different strategy.
                    #
                    # Better: try to find if we can eliminate d by adding a
                    # multiple of d to other columns.
                    pass
                break
            
            # Alternative: try to express M[:, d] as sum of independent columns
            # and use the linear combination to modify U, V, W
            # This is complex. Let's try a simpler heuristic:
            # If the dependent column has small norm, just drop it and verify.
            
            if np.linalg.norm(M[:, d]) < 1e-10:
                # Drop this column
                keep = [q for q in range(R) if q != d]
                new_U = U[:, keep].copy()
                new_V = V[:, keep].copy()
                new_W = W[:, keep].copy()
                
                if _verify(new_U, new_V, new_W, n, m, p):
                    U, V, W = new_U, new_V, new_W
                    R -= 1
                    eliminated = True
                    if R < best_rank:
                        best_rank = R
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
        
        if not eliminated:
            # Try pairwise elimination: check if two columns can be combined
            # Actually, let's try a different approach: look for columns where
            # U[:, r] * V[:, r] * W[:, r] is a linear combination of others
            # This is equivalent to what we're doing with M.
            
            # Let's try: for each pair (r1, r2), check if we can eliminate one
            # by adjusting the other. This is O(R^2) which is 98^2 = 9604, manageable.
            # But it's complex. Let's just break and return what we have.
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 140.0

    U, V, W = _build(98)
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W, rank = _try_reduce(U, V, W, n, m, p, deadline)

    if not _verify(U, V, W, n, m, p, tol=1e-5):
        U, V, W = _build(98)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
