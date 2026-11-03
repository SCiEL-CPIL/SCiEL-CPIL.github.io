"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Compute the implied tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that non-diagonal entries are 0
    # Actually, for matrix multiplication tensor, T[i,j,k] should be 1 if the (i,j,k) triple
    # corresponds to a valid multiplication term. Let me reconsider.
    # The matrix multiplication tensor T[i,j,k] = 1 if i,j,k are all in range (which they always are for 5x5x5)
    # Wait, no. The tensor for matrix multiplication C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    # has T[i,j,k] = 1 for all i,j,k in [0,n). So all entries should be 1.
    # Let me verify more carefully.
    return True

def _try_merge(U, V, W, c1, c2, n, m, p):
    """
    Try to merge columns c1 and c2 into a single column.
    Returns (new_U, new_V, new_W) as lists if successful, else None.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    u1 = U[:, c1].copy()
    u2 = U[:, c2].copy()
    v1 = V[:, c1].copy()
    v2 = V[:, c2].copy()
    w1 = W[:, c1].copy()
    w2 = W[:, c2].copy()
    
    # We want to find u, v, w such that:
    # u * v * w = u1 * v1 * w1 + u2 * v2 * w2  (element-wise product of vectors)
    # where u, v, w have entries in {0, 1/2, 1}
    
    # The support of u must be a superset of supports of u1 and u2
    # Similarly for v and w.
    
    # Let's think about this differently. The tensor equation is:
    # For all (i,j,k): u[i*m+j] * v[j*p+k] * w[k*n+i] = u1[i*m+j]*v1[j*p+k]*w1[k*n+i] + u2[i*m+j]*v2[j*p+k]*w2[k*n+i]
    
    # This is a system of n*m*p equations in the variables u, v, w.
    # But it's nonlinear (product of three variables).
    
    # Alternative approach: enumerate possible supports and solve.
    # The support of u is contained in the union of supports of u1 and u2.
    # Since u1 and u2 are unit vectors in the schoolbook, their supports are single points.
    
    # For schoolbook, u1 has a single 1 at position (i1*m+j1), u2 at (i2*m+j2).
    # So u must have entries at positions (i1*m+j1) and (i2*m+j1) potentially, but also could have others.
    # Actually, if we allow u to have a 1/2 at a new position, that would create spurious tensor entries.
    
    # Let me just try: u = a*u1 + b*u2 where a,b in {0, 1/2, 1}
    # Similarly v = c*v1 + d*v2, w = e*w1 + f*w2
    # Then u*v*w = (a*u1 + b*u2)(c*v1 + d*v2)(e*w1 + f*w2)
    # = ace*u1*v1*w1 + acf*u1*v1*w2 + ade*u1*v2*w1 + adf*u1*v2*w2
    #   + bce*u2*v1*w1 + bcf*u2*v1*w2 + bde*u2*v2*w1 + bdf*u2*v2*w2
    
    # We need this to equal u1*v1*w1 + u2*v2*w2.
    # So the coefficients of the 8 terms must satisfy:
    # ace = 1 (coefficient of u1*v1*w1)
    # bdf = 1 (coefficient of u2*v2*w2)
    # All other 6 coefficients = 0
    
    # If a=0 then ace=0, contradiction. So a != 0.
    # If b=0 then bdf=0, contradiction. So b != 0.
    # Similarly c,d,e,f all nonzero.
    
    # With a,b,c,d,e,f in {1/2, 1} (nonzero):
    # ace = 1 => a*c*e = 1 => since each is 1/2 or 1, all must be 1.
    # bdf = 1 => b*d*f = 1 => all must be 1.
    # Then all cross terms have coefficient 1, not 0. Contradiction.
    
    # So simple linear combination doesn't work. We need a different approach.
    
    # Let's try allowing u, v, w to have entries at positions not in the original supports.
    # This is getting complex. Let me try a direct numerical approach.
    
    # Actually, for the schoolbook case, each term has exactly one nonzero in U, V, W.
    # Let's identify the (i,j,k) indices for each column.
    
    # Find the nonzero positions
    nz_u1 = np.nonzero(u1)[0]
    nz_u2 = np.nonzero(u2)[0]
    nz_v1 = np.nonzero(v1)[0]
    nz_v2 = np.nonzero(v2)[0]
    nz_w1 = np.nonzero(w1)[0]
    nz_w2 = np.nonzero(w2)[0]
    
    if len(nz_u1) != 1 or len(nz_u2) != 1 or len(nz_v1) != 1 or len(nz_v2) != 1 or len(nz_w1) != 1 or len(nz_w2) != 1:
        return None
    
    i1, j1 = divmod(nz_u1[0], m)
    j1_v, k1 = divmod(nz_v1[0], p)
    k1_w, i1_w = divmod(nz_w1[0], n)
    
    i2, j2 = divmod(nz_u2[0], m)
    j2_v, k2 = divmod(nz_v2[0], p)
    k2_w, i2_w = divmod(nz_w2[0], n)
    
    # Verify consistency
    if i1 != i1_w or j1 != j1_v or k1 != k1_w:
        return None
    if i2 != i2_w or j2 != j2_v or k2 != k2_w:
        return None
    
    # Now we need to find u, v, w (with entries in {0, 1/2, 1}) such that
    # the tensor equation holds.
    # The support of u can include (i1*m+j1), (i2*m+j2), and possibly others.
    # But if u has a nonzero at position (i'*m+j') where i' != i1,i2 and j' != j1,j2,
    # then for k such that w[k*n+i'] != 0, we'd get a spurious contribution.
    
    # Let's restrict: u has nonzeros only at positions corresponding to (i1,j1) and (i2,j2).
    # Similarly for v: (j1,k1) and (j2,k2).
    # And w: (k1,i1) and (k2,i2).
    
    # So u = [a at (i1*m+j1), b at (i2*m+j2), 0 elsewhere]
    # v = [c at (j1*p+k1), d at (j2*p+k2), 0 elsewhere]
    # w = [e at (k1*n+i1), f at (k2*n+i2), 0 elsewhere]
    
    # Then u*v*w has nonzeros only at:
    # (i1,j1,k1): a*c*e
    # (i1,j1,k2): a*d*f  [if j1==j2 or if v has nonzero at j2... wait, v[j1*p+k2] = 0 unless j1==j2]
    
    # Hmm, this is where it gets tricky. Let me be more careful.
    # u[i*m+j] * v[j*p+k] * w[k*n+i]
    # For this to be nonzero, we need:
    # u[i*m+j] != 0, v[j*p+k] != 0, w[k*n+i] != 0
    
    # With our restricted supports:
    # u nonzero at (i1,j1) and (i2,j2)
    # v nonzero at (j1,k1) and (j2,k2)
    # w nonzero at (k1,i1) and (k2,i2)
    
    # Possible (i,j,k) triples with nonzero contribution:
    # (i1, j1, k1): u[i1*m+j1]*v[j1*p+k1]*w[k1*n+i1] = a*c*e
    # (i1, j1, k2): u[i1*m+j1]*v[j1*p+k2]*w[k2*n+i1] = a*0*f = 0 (since v[j1*p+k2]=0 if j1!=j2, or if j1==j2 then v[j1*p+k2]=d)
    # etc.
    
    # This is getting very case-dependent. Let me just try all combinations of a,b,c,d,e,f in {0, 1/2, 1}
    # and check if the tensor equation holds.
    
    vals = [0, 0.5, 1.0]
    best = None
    
    for a in vals:
        for b in vals:
            for c in vals:
                for d in vals:
                    for e in vals:
                        for f in vals:
                            # Check tensor equation
                            ok = True
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        rhs = 0.0
                                        # Contribution from term 1
                                        if i == i1 and j == j1 and k == k1:
                                            rhs += 1.0
                                        # Contribution from term 2
                                        if i == i2 and j == j2 and k == k2:
                                            rhs += 1.0
                                        
                                        lhs = 0.0
                                        u_val = 0.0
                                        if i == i1 and j == j1:
                                            u_val = a
                                        elif i == i2 and j == j2:
                                            u_val = b
                                        
                                        v_val = 0.0
                                        if j == j1 and k == k1:
                                            v_val = c
                                        elif j == j2 and k == k2:
                                            v_val = d
                                        
                                        w_val = 0.0
                                        if k == k1 and i == i1:
                                            w_val = e
                                        elif k == k2 and i == i2:
                                            w_val = f
                                        
                                        lhs = u_val * v_val * w_val
                                        
                                        if abs(lhs - rhs) > 1e-9:
                                            ok = False
                                            break
                                    if not ok:
                                        break
                                if not ok:
                                    break
                            if ok:
                                # Check that we actually reduced rank (not just kept same)
                                # The new column should be different from both old columns
                                new_u = np.zeros(n * m)
                                new_v = np.zeros(m * p)
                                new_w = np.zeros(p * n)
                                if a != 0:
                                    new_u[i1 * m + j1] = a
                                if b != 0:
                                    new_u[i2 * m + j2] = b
                                if c != 0:
                                    new_v[j1 * p + k1] = c
                                if d != 0:
                                    new_v[j2 * p + k2] = d
                                if e != 0:
                                    new_w[k1 * n + i1] = e
                                if f != 0:
                                    new_w[k2 * n + i2] = f
                                
                                # Verify it's not just one of the original terms
                                if np.allclose(new_u, u1) and np.allclose(new_v, v1) and np.allclose(new_w, w1):
                                    continue
                                if np.allclose(new_u, u2) and np.allclose(new_v, v2) and np.allclose(new_w, w2):
                                    continue
                                
                                best = (new_u, new_v, new_w)
                                break
                        if best:
                            break
                    if best:
                        break
                if best:
                    break
            if best:
                break
        if best:
            break
    
    if best is None:
        return None
    
    new_u, new_v, new_w = best
    
    # Build new U, V, W with columns c1 and c2 replaced by new column at c1, and c2 removed
    new_U = U.copy()
    new_V = V.copy()
    new_W = W.copy()
    
    new_U[:, c1] = new_u
    new_V[:, c1] = new_v
    new_W[:, c1] = new_w
    
    # Remove column c2
    new_U = np.delete(new_U, c2, axis=1)
    new_V = np.delete(new_V, c2, axis=1)
    new_W = np.delete(new_W, c2, axis=1)
    
    return (new_U.tolist(), new_V.tolist(), new_W.tolist())

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    # Start with schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to merge pairs of columns
    current_U, current_V, current_W = U, V, W
    current_rank = rank
    
    # Keep trying merges until no more can be made or time runs out
    progress = True
    while progress and time.time() < deadline:
        progress = False
        merged = False
        for c1 in range(current_rank):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, current_rank):
                if time.time() > deadline:
                    break
                result = _try_merge(current_U, current_V, current_W, c1, c2, n, m, p)
                if result is not None:
                    current_U, current_V, current_W = result
                    current_rank -= 1
                    progress = True
                    merged = True
                    if current_rank < best_rank:
                        best_U, best_V, best_W = current_U, current_V, current_W
                        best_rank = current_rank
                    # After a merge, restart the loop since indices changed
                    break
            if merged:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
