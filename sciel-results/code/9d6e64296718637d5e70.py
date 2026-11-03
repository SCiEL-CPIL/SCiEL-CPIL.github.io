"""
Matrix multiplication 2x4 by 4x5 - rank-minimal exact tensor decomposition.

Strategy:
  1. Build a rank-36 Strassen-based decomposition (4 Strassen 2x2x2 blocks + schoolbook 2x4x1).
  2. Attempt rank reduction via structured perturbation:
     - Pick a pair of columns to merge (linear combination).
     - Solve for the third factor to maintain exactness.
     - If the system is consistent, we have a valid rank-35 decomposition.
     - If not, try another pair.
  3. If rank reduction fails within time budget, return the rank-36 decomposition.

The tensor T for (n,m,p) = (2,4,5) has entries T[i,j,k] = 1 if the multiplication
A[i,j]*B[j,k] contributes to C[i,k], i.e., T is indexed by (i,j,k) with:
  T[i*m+j, j*p+k, k*n+i] = 1  for all valid i,j,k.

A rank-r decomposition gives U (nm x r), V (mp x r), W (pn x r) such that:
  T_{a,b,c} = sum_l U[a,l] * V[b,l] * W[c,l]

where a = i*m+j, b = j*p+k, c = k*n+i.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 135  # 15s safety margin before 150s limit


def _build_strassen_rank36():
    """Build the rank-36 Strassen decomposition."""
    n, m, p = DIMENSIONS
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        # M1 = (A00 + A11) * (B00 + B11)
        c = col_start
        U[i0 * m + j0, c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[j0 * p + k0, c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[(i0 + 1) * m + j0, c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[j0 * p + k0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] -= 1
        col += 1

        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0 * m + j0, c] += 1
        V[j0 * p + (k0 + 1), c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] -= 1
        W[(k0 + 1) * n + i0, c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[(j0 + 1) * p + k0, c] += 1
        V[j0 * p + k0, c] -= 1
        W[k0 * n + i0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0 * m + j0, c] += 1
        U[i0 * m + (j0 + 1), c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] -= 1
        W[(k0 + 1) * n + i0, c] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[(i0 + 1) * m + j0, c] += 1
        U[i0 * m + j0, c] -= 1
        V[j0 * p + k0, c] += 1
        V[j0 * p + (k0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0 * m + (j0 + 1), c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] -= 1
        V[(j0 + 1) * p + k0, c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj), c] = 1
                V[(j0 + dj) * p + k0, c] = 1
                W[k0 * n + (i0 + di), c] = 1
                c += 1
        return c

    col = add_strassen_222(0, 0, 0, 0)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T as a flat vector of length (n*m) * (m*p) * (p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Check: T[a,b,c] = sum_l U[a,l]*V[b,l]*W[c,l]
    # Use einsum: result[a,b,c] = sum_l U[a,l] V[b,l] W[c,l]
    result = np.einsum('al,bl,cl->abc', U, V, W)
    return np.allclose(result, T, atol=1e-10)


def _try_rank_reduction(U, V, W, n, m, p, max_pairs=200):
    """
    Try to reduce rank by 1 by merging two columns.
    For a pair (i,j) of columns, we set:
      U_new[:, k] = U[:,i] + alpha * U[:,j]  for some alpha
      V_new[:, k] = V[:,i] + beta * V[:,j]   for some beta
      W_new[:, k] = W[:,i] + gamma * W[:,j]  for some gamma
    and remove one of the two columns.
    
    Actually, a cleaner approach: we want to find if there exists a non-trivial
    linear dependence. We try: merge columns i and j by setting
      U_merged = U[:,i] + U[:,j]
      V_merged = V[:,i] + V[:,j]  
      W_merged = W[:,i] + W[:,j]
    and dropping column j. This works if the tensor contribution of column j
    can be absorbed.
    
    More generally, for rank reduction from r to r-1, we need to find scalars
    such that one column's contribution is a linear combination of others.
    
    A practical approach: for each pair (i,j), try to solve for the new W column
    given U and V are merged.
    """
    rank = U.shape[1]
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # For each pair of columns to merge, try to find if rank-35 works
    # Strategy: merge columns i and j by taking their sum in U and V,
    # then solve for the new W column.
    # 
    # If we merge columns i and j into a single column:
    #   U_new[:, merged] = U[:, i] + U[:, j]
    #   V_new[:, merged] = V[:, i] + V[:, j]
    #   W_new[:, merged] = ? (to be solved)
    #   All other columns unchanged.
    #
    # The constraint is: for all (a,b,c):
    #   T[a,b,c] = sum_{l != i,j} U[a,l]V[b,l]W[c,l] + (U[a,i]+U[a,j])*(V[b,i]+V[b,j])*W_new[c,merged]
    #
    # Let R[a,b,c] = T[a,b,c] - sum_{l != i,j} U[a,l]V[b,l]W[c,l]
    # Then we need: R[a,b,c] = (U[a,i]+U[a,j])*(V[b,i]+V[b,j])*W_new[c,merged]
    #
    # This means: R[a,b,c] / ((U[a,i]+U[a,j])*(V[b,i]+V[b,j])) must be independent of (a,b)
    # and equal to W_new[c,merged] for each c.
    
    for pair_count in range(max_pairs):
        if time.time() > _DEADLINE:
            return None
        
        # Pick a pair pseudo-randomly (deterministic based on pair_count)
        i = (pair_count * 7 + 3) % rank
        j = (pair_count * 13 + 11) % rank
        if i == j:
            j = (j + 1) % rank
        if i > j:
            i, j = j, i
        
        # Compute U_sum and V_sum
        U_sum = U[:, i] + U[:, j]
        V_sum = V[:, i] + V[:, j]
        
        # Compute the residual R after removing columns i and j
        # R[a,b,c] = T[a,b,c] - sum_{l != i,j} U[a,l]*V[b,l]*W[c,l]
        # = T[a,b,c] - (U[:,i]V[:,i]W[:,i] + U[:,j]V[:,j]W[:,j]) contracted
        # 
        # More efficiently: compute the full tensor from all columns, subtract
        # contributions of i and j.
        
        # Full tensor from all columns:
        # T_full[a,b,c] = sum_l U[a,l]V[b,l]W[c,l]
        # We know T_full = T (it's a valid decomposition)
        # T_no_ij[a,b,c] = T[a,b,c] - U[a,i]V[b,i]W[c,i] - U[a,j]V[b,j]W[c,j]
        
        # We need: T_no_ij[a,b,c] = U_sum[a] * V_sum[b] * W_new[c]
        # So: W_new[c] = T_no_ij[a,b,c] / (U_sum[a] * V_sum[b])  for any (a,b) where U_sum[a]*V_sum[b] != 0
        
        # Check consistency: for all (a,b) with U_sum[a]*V_sum[b] != 0:
        #   T_no_ij[a,b,c] / (U_sum[a]*V_sum[b]) should be the same for all (a,b) and equal W_new[c]
        
        # Compute T_no_ij for a few sample (a,b) pairs to check
        # This is expensive for large tensors, so sample
        nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
        
        # Compute T_no_ij[a,b,c] = T[a,b,c] - U[a,i]*V[b,i]*W[c,i] - U[a,j]*V[b,j]*W[c,j]
        # We need to check if T_no_ij is a rank-1 tensor in (a,b,c) with factors U_sum, V_sum, W_new.
        
        # For efficiency, check on a subset of (a,b) pairs
        sample_ab = [(a, b) for a in range(min(nm, 4)) for b in range(min(mp, 4))]
        
        W_new_candidates = []
        valid = True
        for (a, b) in sample_ab:
            us = U_sum[a]
            vs = V_sum[b]
            if abs(us) < 1e-12 or abs(vs) < 1e-12:
                # Can't determine W_new[c] from this (a,b)
                # Check that T_no_ij[a,b,c] = 0 for all c
                T_no_ij_ab = T[a, b, :] - U[a, i] * V[b, i] * W[:, i] - U[a, j] * V[b, j] * W[:, j]
                if np.any(np.abs(T_no_ij_ab) > 1e-10):
                    valid = False
                    break
                continue
            
            denom = us * vs
            T_no_ij_ab = T[a, b, :] - U[a, i] * V[b, i] * W[:, i] - U[a, j] * V[b, j] * W[:, j]
            W_candidate = T_no_ij_ab / denom
            W_new_candidates.append(W_candidate)
        
        if not valid:
            continue
        
        if not W_new_candidates:
            # All (a,b) have zero U_sum*V_sum, so T_no_ij must be zero everywhere
            # Check this
            T_no_ij_full = T - np.einsum('a,b,c->abc', U[:, i], V[:, i], W[:, i]) - np.einsum('a,b,c->abc', U[:, j], V[:, j], W[:, j])
            if np.any(np.abs(T_no_ij_full) > 1e-10):
                continue
            W_new = np.zeros(pn)
        else:
            W_new = W_new_candidates[0]
            # Check consistency with other candidates
            for wc in W_new_candidates[1:]:
                if not np.allclose(wc, W_new, atol=1e-10):
                    valid = False
                    break
            if not valid:
                continue
        
        # Verify the full decomposition
        U_new = np.delete(U, j, axis=1)
        U_new[:, i] = U_sum  # Replace column i with the sum
        V_new = np.delete(V, j, axis=1)
        V_new[:, i] = V_sum
        W_new_full = np.delete(W, j, axis=1)
        W_new_full[:, i] = W_new
        
        if _verify_decomposition(U_new, V_new, W_new_full, n, m, p):
            return U_new, V_new, W_new_full
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x5 matrix.
    """
    n, m, p = DIMENSIONS
    
    # Step 1: Build rank-36 Strassen decomposition
    U, V, W = _build_strassen_rank36()
    
    # Verify it's correct
    assert _verify_decomposition(U, V, W, n, m, p), "Strassen decomposition verification failed"
    
    # Step 2: Try rank reduction
    if time.time() < _DEADLINE - 10:
        result = _try_rank_reduction(U, V, W, n, m, p, max_pairs=500)
        if result is not None:
            U, V, W = result
            assert _verify_decomposition(U, V, W, n, m, p), "Rank-reduced decomposition verification failed"
    
    # Deadline guard
    if time.time() > _DEADLINE:
        raise TimeoutError("Deadline exceeded")
    
    # Convert to lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
