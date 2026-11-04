"""Tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _construct_schoolbook(n=4, m=4, p=5):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return the 7 rank-1 tensors for 2x2x2 matrix multiplication.
    """
    terms = []
    terms.append(([1,0, 0,1], [1,0, 0,1], [1,0, 0,1]))
    terms.append(([0,0, 1,1], [1,0, 0,0], [0,1, 0,0]))
    terms.append(([1,0, 0,0], [0,0, 1,-1], [0,0, 1,0]))
    terms.append(([0,0, 0,1], [1,-1, 0,0], [0,0, 0,1]))
    terms.append(([1,1, 0,0], [0,0, 0,1], [0,0, 1,0]))
    terms.append(([-1,0, 1,0], [1,1, 0,0], [0,1, 0,0]))
    terms.append(([0,1, 0,-1], [0,0, 1,1], [0,0, 0,1]))
    return terms

def _strassen_2x2x3():
    """
    Return rank-1 tensors for 2x2x3 matrix multiplication.
    We use a decomposition with rank 9.
    For multiplying A (2x2) by B (2x3), result C (2x3).
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    We can use Strassen-like ideas. Let's construct one:
    Terms for 2x2x3:
    """
    # For 2x2x3, we can use the following approach:
    # Split B into two 2x2 matrices and a 2x1 matrix
    # Use Strassen for each 2x2 part, and schoolbook for the 2x1 part
    # But that gives 7+7+2 = 16, too high.
    
    # Better: use a direct decomposition.
    # The tensor T[i,j,k] = delta_{i,k} for the matmul tensor... 
    # Actually for 2x2x3 matmul: C[i,k] = sum_j A[i,j] B[j,k]
    # The structure tensor has T[i,j,k] = 1 if i,k match the "output" index pattern
    
    # Let me think of it as: we need U (2*2=4 entries), V (2*3=6 entries), W (3*2=6 entries)
    # such that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,k,j
    
    # For 2x2x3: n=2, m=2, p=3
    # U has 4 rows, V has 6 rows, W has 6 rows
    
    # Let me try to find a rank-9 decomposition by construction.
    # Use 3 Strassen-like groups, one for each column of B.
    
    # Actually, let me use a known result: rank(2,2,3) <= 9
    # One way: use the fact that 2x2x3 can be embedded in 2x2x4 which has rank 14 (2*Strassen)
    # But we want exactly 2x2x3.
    
    # Let me try a different approach: use 3 copies of a 2x2x1 decomposition
    # 2x2x1: C[i,0] = sum_j A[i,j]*B[j,0], rank = 2 (schoolbook)
    # So 3 * 2 = 6? No, that's not right because the Strassen structure doesn't decompose that way.
    
    # Let me just use schoolbook for now: rank = 2*2*3 = 12
    # And later try to optimize.
    
    # For now, return schoolbook terms for 2x2x3
    terms = []
    n, m, p = 2, 2, 3
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0]*(n*m)
                v = [0]*(m*p)
                w = [0]*(p*n)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                terms.append((u, v, w))
    return terms

def _embed_block(n, m, p, i0, j0, k0, terms):
    """Embed a block decomposition into the full tensor at offset (i0, j0, k0)."""
    U = np.zeros((n*m, len(terms)))
    V = np.zeros((m*p, len(terms)))
    W = np.zeros((p*n, len(terms)))
    
    for r, (u, v, w) in enumerate(terms):
        # u is indexed by (i_local, j_local) where i_local in [0,2), j_local in [0,2)
        for ii in range(2):
            for jj in range(2):
                U[(i0+ii)*m + (j0+jj), r] = u[ii*2 + jj]
        for jj in range(2):
            for kk in range(3 if p > 2 else 2):
                if jj*p + kk < m*p:
                    V[(j0+jj)*p + (k0+kk), r] = v[jj*2 + kk] if kk < 2 else 0
        for kk in range(3 if p > 2 else 2):
            for ii in range(2):
                if kk*n + ii < p*n:
                    W[(k0+kk)*n + (i0+ii), r] = w[kk*2 + ii] if kk < 2 else 0
    
    return U, V, W

def _construct_strassen_embedded():
    """
    Construct a decomposition using Strassen for the top-left 2x2x2 part
    and schoolbook for the remaining parts.
    """
    n, m, p = 4, 4, 5
    rank = 79
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_str, V_str, W_str = _embed_strassen_2x2x2()
    U[:, :7] = U_str
    V[:, :7] = V_str
    W[:, :7] = W_str
    
    col = 7
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if i < 2 and k < 2 and j < 2:
                    continue
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _embed_strassen_2x2x2():
    """Embed the 2x2x2 Strassen decomposition into the top-left 2x2x2 part of 4x4x5."""
    n, m, p = 4, 4, 5
    terms = _strassen_2x2x2()
    
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(2):
            for j in range(2):
                U[i*m + j, r] = u[i*2 + j]
        for j in range(2):
            for k in range(2):
                V[j*p + k, r] = v[j*2 + k]
        for k in range(2):
            for i in range(2):
                W[k*n + i, r] = w[k*2 + i]
    
    return U, V, W

def _try_merge_columns(U, V, W, n, m, p):
    """
    Try to reduce rank by checking if any pair of columns can be merged.
    Returns (U_new, V_new, W_new, merged) if a merge was found.
    """
    rank = U.shape[1]
    
    for a in range(rank):
        for b in range(a+1, rank):
            # Compute target tensor for merging columns a and b
            target = np.zeros((n, p, m))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        target[i,k,j] = (U[i*m+j, a]*V[j*p+k, a]*W[k*n+i, a] + 
                                        U[i*m+j, b]*V[j*p+k, b]*W[k*n+i, b])
            
            # Check if target is rank-1 using a more efficient method
            # A 3D tensor T[i,k,j] is rank-1 iff all its matrix slices are rank-1
            # and consistent
            is_rank1 = True
            
            # Check using the determinant criterion for 2x2 minors
            # For a rank-1 tensor, T[i1,k1,j1]*T[i2,k2,j2] = T[i1,k2,j1]*T[i2,k1,j2]
            # We only need to check a subset due to symmetry
            
            # Find a non-zero anchor
            anchor_found = False
            i0 = k0 = j0 = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        if abs(target[i,k,j]) > 1e-12:
                            i0, k0, j0 = i, k, j
                            anchor_found = True
                            break
                    if anchor_found:
                        break
                if anchor_found:
                    break
            
            if not anchor_found:
                # Target is all zeros, we can remove both columns
                U_new = np.delete(U, [a, b], axis=1)
                V_new = np.delete(V, [a, b], axis=1)
                W_new = np.delete(W, [a, b], axis=1)
                return U_new, V_new, W_new, True
            
            # Check rank-1 condition using the anchor
            # T[i,k,j] = T[i0,k0,j0] * (T[i,k0,j] / T[i0,k0,j]) * (T[i0,k,j] / T[i0,k0,j]) / ... 
            # More precisely: T[i,k,j] / T[i0,k0,j0] = (T[i,k0,j0] * T[i0,k,j0] * T[i0,k0,j]) / (T[i0,k0,j0]^2) ... 
            # Let's use: T[i,k,j] * T[i0,k0,j0] = T[i,k0,j0] * T[i0,k,j0] * T[i0,k0,j] / T[i0,k0,j0]
            # This is getting complicated. Let's use a simpler check.
            
            # For rank-1: T[i,k,j] = u[i*m+j] * v[j*p+k] * w[k*n+i]
            # So T[i,k,j] / T[i0,k0,j0] = (u[i*m+j]/u[i0*m+j0]) * (v[j*p+k]/v[j0*p+k0]) * (w[k*n+i]/w[k0*n+i0])
            
            # Let's define:
            # alpha[i,j] = T[i,k0,j] / T[i0,k0,j0]  (for fixed k0)
            # beta[j,k] = T[i0,k,j] / T[i0,k0,j0]  (for fixed i0)  
            # gamma[k,i] = T[i0,k,j0] / T[i0,k0,j0]  (for fixed j0)
            
            # Then T[i,k,j] should equal T[i0,k0,j0] * alpha[i,j] * beta[j,k] * gamma[k,i] / (alpha[i0,j0]*beta[j0,k0]*gamma[k0,i0])
            # But alpha[i0,j0] = T[i0,k0,j0]/T[i0,k0,j0] = 1, similarly for others.
            # So T[i,k,j] should equal T[i0,k0,j0] * alpha[i,j] * beta[j,k] * gamma[k,i]
            
            # Wait, that's not right either. Let me think again.
            # If T[i,k,j] = u[i,j] * v[j,k] * w[k,i], then:
            # T[i,k0,j] = u[i,j] * v[j,k0] * w[k0,i]
            # T[i0,k,j] = u[i0,j] * v[j,k] * w[k,i0]
            # T[i0,k0,j] = u[i0,j] * v[j,k0] * w[k0,i0]
            
            # T[i,k,j] * T[i0,k0,j0] = u[i,j]*v[j,k]*w[k,i] * u[i0,j0]*v[j0,k0]*w[k0,i0]
            # T[i,k0,j] * T[i0,k,j0] = u[i,j]*v[j,k0]*w[k0,i] * u[i0,j0]*v[j0,k]*w[k,i0]
            
            # For rank-1: T[i,k,j]*T[i0,k0,j0] = T[i,k0,j]*T[i0,k,j0] * (w[k,i]*v[j,k]*w[k0,i0]*v[j0,k0]) / (w[k0,i]*v[j,k0]*w[k,i0]*v[j0,k])
            # This is still messy.
            
            # Simpler: just check the 2x2 minors for a few key combinations
            # Check: for all i1,i2,k1,k2,j1,j2:
            # T[i1,k1,j1]*T[i2,k2,j2] == T[i1,k2,j1]*T[i2,k1,j2]
            
            # To save time, check a random sample
            rng = random.Random(42)
            samples = 50
            for _ in range(samples):
                i1 = rng.randint(0, n-1)
                k1 = rng.randint(0, p-1)
                j1 = rng.randint(0, m-1)
                i2 = rng.randint(0, n-1)
                k2 = rng.randint(0, p-1)
                j2 = rng.randint(0, m-1)
                lhs = target[i1,k1,j1] * target[i2,k2,j2]
                rhs = target[i1,k2,j1] * target[i2,k1,j2]
                if abs(lhs - rhs) > 1e-9 * max(1.0, abs(lhs), abs(rhs)):
                    is_rank1 = False
                    break
            
            if not is_rank1:
                continue
            
            # Try to extract the rank-1 vectors
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            # Use the anchor to normalize
            T_anchor = target[i0, k0, j0]
            if abs(T_anchor) < 1e-15:
                continue
            
            # Set w_vec[k0*n+i0] = 1 (gauge choice)
            w_vec[k0*n+i0] = 1.0
            
            # Then: u_vec[i0*m+j0] * v_vec[j0*p+k0] * 1 = T_anchor
            # And: u_vec[i*m+j0] * v_vec[j0*p+k] * w_vec[k*n+i0] = target[i,k,j0]
            # And: u_vec[i0*m+j] * v_vec[j*p+k0] * w_vec[k0*n+i0] = target[i0,k0,j]
            
            # From the last equation: u_vec[i0*m+j] = target[i0,k0,j] / (v_vec[j*p+k0] * 1)
            # From the second: u_vec[i*m+j0] = target[i,k,j0] / (v_vec[j0*p+k] * w_vec[k*n+i0])
            
            # This is a system of equations. Let's solve it iteratively.
            # Initialize with schoolbook-like values
            for i in range(n):
                for j in range(m):
                    u_vec[i*m+j] = 1.0
            for j in range(m):
                for k in range(p):
                    v_vec[j*p+k] = 1.0
            for k in range(p):
                for i in range(n):
                    w_vec[k*n+i] = 1.0
            
            # Iterate to converge
            converged = False
            for iter in range(100):
                # Update u_vec
                for i in range(n):
                    for j in range(m):
                        # u_vec[i*m+j] = target[i,k0,j] / (v_vec[j*p+k0] * w_vec[k0*n+i])
                        denom = v_vec[j*p+k0] * w_vec[k0*n+i]
                        if abs(denom) > 1e-15:
                            u_vec[i*m+j] = target[i,k0,j] / denom
                        # Also: u_vec[i*m+j] = target[i0,k,j] / (v_vec[j*p+k] * w_vec[k*n+i0])
                        denom2 = v_vec[j*p+k] * w_vec[k*n+i0]
                        if abs(denom2) > 1e-15:
                            u_vec[i*m+j] = target[i0,k,j] / denom2
                
                # Update v_vec
                for j in range(m):
                    for k in range(p):
                        # v_vec[j*p+k] = target[i0,k,j] / (u_vec[i0*m+j] * w_vec[k*n+i0])
                        denom = u_vec[i0*m+j] * w_vec[k*n+i0]
                        if abs(denom) > 1e-15:
                            v_vec[j*p+k] = target[i0,k,j] / denom
                        # Also: v_vec[j*p+k] = target[i,k0,j] / (u_vec[i*m+j] * w_vec[k0*n+i])
                        denom2 = u_vec[i*m+j] * w_vec[k0*n+i]
                        if abs(denom2) > 1e-15:
                            v_vec[j*p+k] = target[i,k0,j] / denom2
                
                # Update w_vec
                for k in range(p):
                    for i in range(n):
                        # w_vec[k*n+i] = target[i0,k,j0] / (u_vec[i0*m+j0] * v_vec[j0*p+k])
                        denom = u_vec[i0*m+j0] * v_vec[j0*p+k]
                        if abs(denom) > 1e-15:
                            w_vec[k*n+i] = target[i0,k,j0] / denom
                        # Also: w_vec[k*n+i] = target[i,k0,j0] / (u_vec[i*m+j0] * v_vec[j0*p+k0])
                        denom2 = u_vec[i*m+j0] * v_vec[j0*p+k0]
                        if abs(denom2) > 1e-15:
                            w_vec[k*n+i] = target[i,k0,j0] / denom2
                
                # Check convergence
                err = 0
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            pred = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
                            err = max(err, abs(pred - target[i,k,j]))
                if err < 1e-10:
                    converged = True
                    break
            
            if not converged:
                continue
            
            # Verify the extraction
            ok = True
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        pred = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
                        if abs(pred - target[i,k,j]) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if not ok:
                continue
            
            # Create new matrices with columns a,b replaced by the merged column
            U_new = np.delete(U, [a, b], axis=1)
            V_new = np.delete(V, [a, b], axis=1)
            W_new = np.delete(W, [a, b], axis=1)
            
            # Insert the new column at position min(a,b)
            pos = min(a, b)
            U_new = np.insert(U_new, pos, u_vec, axis=1)
            V_new = np.insert(V_new, pos, v_vec, axis=1)
            W_new = np.insert(W_new, pos, w_vec, axis=1)
            
            return U_new, V_new, W_new, True
    
    return U, V, W, False

def _greedy_merge(U, V, W, n=4, m=4, p=5, max_iter=100):
    """
    Try to reduce rank by finding pairs of rank-1 terms that can be merged.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for _ in range(max_iter):
        U, V, W, merged = _try_merge_columns(U, V, W, n, m, p)
        if not merged:
            break
    
    return U, V, W

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 110 second limit to be safe
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 80
    
    # Start with schoolbook
    best_result = _construct_schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try the Strassen-embedded approach
    try:
        strassen_result = _construct_strassen_embedded()
        rank_s = len(strassen_result["U"][0]) if strassen_result["U"] else 0
        if rank_s < best_rank:
            if _validate(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                best_result = strassen_result
                best_rank = rank_s
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_result
    
    # Try greedy merging on the Strassen-embedded result
    try:
        U = np.array(best_result["U"], dtype=float)
        V = np.array(best_result["V"], dtype=float)
        W = np.array(best_result["W"], dtype=float)
        
        U_merged, V_merged, W_merged = _greedy_merge(U, V, W, n, m, p, max_iter=50)
        new_rank = U_merged.shape[1]
        
        if new_rank < best_rank:
            result = {"U": U_merged.tolist(), "V": V_merged.tolist(), "W": W_merged.tolist()}
            if _validate(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = new_rank
    except Exception:
        pass
    
    # Check time again
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
