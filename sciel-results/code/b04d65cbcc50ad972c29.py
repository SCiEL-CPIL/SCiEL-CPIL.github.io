"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product as iproduct

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _tensor_slice(U, V, W, r):
    """
    Compute the tensor slice contributed by term r.
    Returns a 3D array of shape (n*m, m*p, p*n) but we flatten to 1D for efficiency.
    Actually, let's use a smarter approach: the contribution to entry (i,j,k) is
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We'll represent each term's contribution as a vector in R^(n*m*p) indexed by (i,j,k).
    """
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the contribution vector for term r
    # Index: (i, j, k) -> flat index i*m*p + j*p + k
    contrib = np.zeros(n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                contrib[i * m * p + j * p + k] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    return contrib

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Fast vectorized verification.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Build the full tensor contribution
    # For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can do this with broadcasting
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Reshape to 3D: U[i,j,r], V[j,k,r], W[k,i,r]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check valid entries: T[i,j,k] should be 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=tol):
        return False
    
    # Check invalid entries: for each (x, y, z) where x=k*n+i, y=i2*m+j, z=j2*p+k2
    # The product U[y,r]*V[z,r]*W[x,r] should be 0 unless i=i2, j=j2, k=k2
    # This is harder to vectorize directly, but we can check the "off-diagonal" contributions
    
    # Actually, let's verify by checking that the tensor equals the identity-like tensor
    # The matrix multiplication tensor has T[i,j,k] = 1 iff the indices are consistent
    # For n=2, m=5, p=6: T[i,j,k] = 1 for all i in [0,2), j in [0,5), k in [0,6)
    # Wait, that's not right. The tensor is:
    # T[i,j,k] = 1 if this is a valid (i,j,k) triple, else 0
    # But actually ALL (i,j,k) triples are valid! The tensor for matrix multiplication
    # M[i,k] = sum_j A[i,j] B[j,k] has T[i,j,k] = 1 for all i,j,k.
    # The "invalid" entries are when we try to use inconsistent indices.
    
    # Let me reconsider. The decomposition condition is:
    # For the matrix multiplication tensor, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # where the LHS is evaluated at (i,j,k) and the RHS is 1 when (i,j,k) is a valid triple.
    # But actually, in the standard formulation, the tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    # The "invalid" check is about ensuring that when we plug in inconsistent indices, we get 0.
    
    # Let me just do the full check properly
    # The tensor for 2x5x6 matrix multiplication:
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    # So all 2*5*6 = 60 entries should be 1.
    # There are no "invalid" entries in the sense that all (i,j,k) with i<n, j<m, k<p are valid.
    
    # Wait, I think I'm confusing myself. Let me re-read the verification.
    # The original code checks:
    # 1. For all (i,j,k) with i<n, j<m, k<p: sum_r ... = 1
    # 2. For all (x,y,z) with x<p*n, y<n*m, z<m*p: if the indices don't match, sum_r ... = 0
    
    # The second check is about the "extended" tensor where we allow any (x,y,z) and check consistency.
    # x = k*n + i, y = i2*m + j, z = j2*p + k2
    # The product is U[y,r]*V[z,r]*W[x,r] = U[i2*m+j, r]*V[j2*p+k2, r]*W[k*n+i, r]
    # This should be 1 if i=i2, j=j2, k=k2, else 0.
    
    # So the full check is:
    # For all (i, i2, j, j2, k, k2):
    #   sum_r U[i2*m+j, r] * V[j2*p+k2, r] * W[k*n+i, r] = 1 if i=i2 and j=j2 and k=k2, else 0
    
    # This is a much larger check. Let's do it efficiently.
    # Number of terms: n*n*m*m*p*p = 4*25*36 = 3600
    # For each, we sum over rank terms.
    
    # Vectorized approach:
    # Create arrays for all combinations
    i_vals = np.arange(n)
    i2_vals = np.arange(n)
    j_vals = np.arange(m)
    j2_vals = np.arange(m)
    k_vals = np.arange(p)
    k2_vals = np.arange(p)
    
    # Create meshgrid
    I, I2, J, J2, K, K2 = np.meshgrid(i_vals, i2_vals, j_vals, j2_vals, k_vals, k2_vals, indexing='ij')
    I = I.ravel()
    I2 = I2.ravel()
    J = J.ravel()
    J2 = J2.ravel()
    K = K.ravel()
    K2 = K2.ravel()
    
    # Compute y = i2*m + j, z = j2*p + k2, x = k*n + i
    Y = I2 * m + J
    Z = J2 * p + K2
    X = K * n + I
    
    # Get U[Y, :], V[Z, :], W[X, :]
    # Shape: (num_combos, rank)
    U_sel = U[Y, :]  # (num_combos, rank)
    V_sel = V[Z, :]  # (num_combos, rank)
    W_sel = W[X, :]  # (num_combos, rank)
    
    # Compute sum_r U_sel[:,r] * V_sel[:,r] * W_sel[:,r]
    result = np.einsum('cr,cr,cr->c', U_sel, V_sel, W_sel)
    
    # Expected: 1 if I==I2 and J==J2 and K==K2, else 0
    expected = ((I == I2) & (J == J2) & (K == K2)).astype(float)
    
    if not np.allclose(result, expected, atol=tol):
        return False
    
    return True

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P2
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # P3
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P4
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # P5
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # P6
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P7
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
            
    for k in range(p):
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
        
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def _can_remove_term(U, V, W, r_to_remove, n, m, p, tol=1e-8):
    """
    Check if term r_to_remove can be removed (its contribution is in the span of others).
    We check if the vector of contributions from term r_to_remove is a linear combination
    of the contributions from other terms.
    
    Actually, a simpler check: just verify the decomposition without that term.
    But that's O(rank) work. Let's do a smarter check.
    
    The contribution of term r to the full tensor is a vector in R^(n*n*m*m*p*p) (the extended tensor).
    If this vector is in the span of the other terms' contribution vectors, we can remove it.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Build the extended tensor contribution for each term
    # For term r, the contribution to (i,i2,j,j2,k,k2) is:
    # U[i2*m+j, r] * V[j2*p+k2, r] * W[k*n+i, r]
    # This is a vector of length n*n*m*m*p*p = 4*25*36 = 3600
    
    # For efficiency, only check if the removal works by direct verification
    # But do it quickly with the fast verifier
    U_test = np.delete(U, r_to_remove, axis=1)
    V_test = np.delete(V, r_to_remove, axis=1)
    W_test = np.delete(W, r_to_remove, axis=1)
    
    return _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol=tol)

def _try_eliminate_terms(U, V, W, n, m, p, max_attempts=500):
    """
    Try to eliminate terms by checking if removing a term maintains the decomposition.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    attempts = 0
    while attempts < max_attempts and _time_left() > 20:
        attempts += 1
        cur_rank = U.shape[1]
        if cur_rank <= best_rank:
            break
        
        # Try removing each term and see if we can compensate
        found = False
        for r_to_remove in range(cur_rank):
            if _time_left() < 20:
                break
            
            if _can_remove_term(U, V, W, r_to_remove, n, m, p):
                U = np.delete(U, r_to_remove, axis=1)
                V = np.delete(V, r_to_remove, axis=1)
                W = np.delete(W, r_to_remove, axis=1)
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if not found:
            break
    
    return best_U, best_V, best_W, best_rank

def _try_merge_terms(U, V, W, n, m, p, max_pairs=100):
    """
    Try to merge two terms into one by finding a linear combination that cancels.
    If terms r1 and r2 have contributions c1 and c2, and c1 + c2 = 0 in some sense,
    we might be able to replace them with a single term.
    
    Actually, a better approach: try all pairs and see if their sum can be represented
    as a single rank-1 tensor (outer product of three vectors).
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    n, m, p = DIMENSIONS
    
    # For each pair (r1, r2), check if their combined contribution can be a single term
    # The combined contribution is:
    # (U[i2*m+j, r1] + U[i2*m+j, r2]) * (V[j2*p+k2, r1] + V[j2*p+k2, r2]) * (W[k*n+i, r1] + W[k*n+i, r2])
    # This is NOT the same as the sum of individual contributions in general.
    # The sum is:
    # U[i2*m+j, r1]*V[j2*p+k2, r1]*W[k*n+i, r1] + U[i2*m+j, r2]*V[j2*p+k2, r2]*W[k*n+i, r2]
    # For these to be equal for all (i,i2,j,j2,k,k2), we need a very specific condition.
    
    # A simpler approach: try to find if two terms can be replaced by one term
    # by solving a system of equations. This is complex, so let's try a heuristic:
    # Check if the two terms have proportional U, V, W vectors (up to scaling).
    
    for r1 in range(rank):
        if _time_left() < 20:
            break
        for r2 in range(r1 + 1, rank):
            if _time_left() < 20:
                break
            
            # Check if U[:,r1] is proportional to U[:,r2]
            # and V[:,r1] is proportional to V[:,r2]
            # and W[:,r1] is proportional to W[:,r2]
            U1 = U[:, r1]
            U2 = U[:, r2]
            V1 = V[:, r1]
            V2 = V[:, r2]
            W1 = W[:, r1]
            W2 = W[:, r2]
            
            # Check proportionality
            if np.allclose(U1, 0) and np.allclose(U2, 0):
                continue
            if np.allclose(V1, 0) and np.allclose(V2, 0):
                continue
            if np.allclose(W1, 0) and np.allclose(W2, 0):
                continue
            
            # Try to find scalars a, b such that a*U1 + b*U2 = 0 (component-wise)
            # This is only possible if U1 and U2 are proportional
            # Check if U1 is proportional to U2
            if not np.allclose(U1 * U2, 0, atol=1e-10):
                # Not both zero, check proportionality
                # Find non-zero entry in U1
                nz = np.nonzero(np.abs(U1) > 1e-10)[0]
                if len(nz) == 0:
                    continue
                ratio = U2[nz[0]] / U1[nz[0]]
                if not np.allclose(U1 * ratio, U2, atol=1e-8):
                    continue
            
            if not np.allclose(V1, 0, atol=1e-10) and not np.allclose(V2, 0, atol=1e-10):
                nz = np.nonzero(np.abs(V1) > 1e-10)[0]
                if len(nz) == 0:
                    continue
                ratio_v = V2[nz[0]] / V1[nz[0]]
                if not np.allclose(V1 * ratio_v, V2, atol=1e-8):
                    continue
            
            if not np.allclose(W1, 0, atol=1e-10) and not np.allclose(W2, 0, atol=1e-10):
                nz = np.nonzero(np.abs(W1) > 1e-10)[0]
                if len(nz) == 0:
                    continue
                ratio_w = W2[nz[0]] / W1[nz[0]]
                if not np.allclose(W1 * ratio_w, W2, atol=1e-8):
                    continue
            
            # If we get here, the terms might be mergeable
            # Try removing both and adding a new term
            # The new term would be U_new = U1 + U2, V_new = V1 + V2, W_new = W1 + W2
            # But this only works if the cross terms cancel, which is rare.
            # Let's just try and verify.
            U_test = np.delete(U, [r1, r2], axis=1)
            V_test = np.delete(V, [r1, r2], axis=1)
            W_test = np.delete(W, [r1, r2], axis=1)
            
            # Add new term
            U_new = np.column_stack([U_test, U1 + U2])
            V_new = np.column_stack([V_test, V1 + V2])
            W_new = np.column_stack([W_test, W1 + W2])
            
            if _verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        if _time_left() < 20:
            break
    
    return best_U, best_V, best_W, best_rank

def _try_different_partitions(n, m, p):
    """
    Try different block partitions to find lower-rank decompositions.
    """
    best_rank = 54
    best_U, best_V, best_W = None, None, None
    
    # Try different ways to partition j indices into groups of size 2 and 1
    # and k indices into groups of size 2
    j_indices = list(range(m))
    k_indices = list(range(p))
    
    # Generate all ways to partition j into pairs and singles
    def get_j_partitions(indices):
        """Generate all partitions of indices into pairs and singles."""
        if not indices:
            yield []
            return
        if len(indices) == 1:
            yield [(indices[0],)]
            return
        # Try making the first element a single
        for rest in get_j_partitions(indices[1:]):
            yield [(indices[0],)] + rest
        # Try pairing the first element with another
        for i in range(1, len(indices)):
            pair = (indices[0], indices[i])
            remaining = [x for x in indices[1:] if x != indices[i]]
            for rest in get_j_partitions(remaining):
                yield [pair] + rest
    
    def get_k_partitions(indices):
        """Generate all partitions of indices into pairs and singles."""
        if not indices:
            yield []
            return
        if len(indices) == 1:
            yield [(indices[0],)]
            return
        for i in range(1, len(indices)):
            pair = (indices[0], indices[i])
            remaining = [x for x in indices[1:] if x != indices[i]]
            for rest in get_k_partitions(remaining):
                yield [pair] + rest
    
    # For efficiency, only try a few promising partitions
    j_partitions = list(get_j_partitions(j_indices))
    k_partitions = list(get_k_partitions(k_indices))
    
    # Limit search
    max_j_parts = min(30, len(j_partitions))
    max_k_parts = min(30, len(k_partitions))
    
    for j_part in j_partitions[:max_j_parts]:
        for k_part in k_partitions[:max_k_parts]:
            if _time_left() < 30:
                break
            
            # Count rank for this partition
            rank = 0
            for jp in j_part:
                for kp in k_part:
                    if len(jp) == 2 and len(kp) == 2:
                        rank += 7  # Strassen
                    elif len(jp) == 1 and len(kp) == 1:
                        rank += 2  # Two separate products (one per row of A)
                    elif len(jp) == 2 and len(kp) == 1:
                        rank += 2  # Each j in pair needs its own product with single k
                    elif len(jp) == 1 and len(kp) == 2:
                        rank += 2  # Single j, two k's
                        rank += 2
                    
            if rank >= best_rank:
                continue
            
            # Build decomposition for this partition
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            
            valid = True
            for jp in j_part:
                for kp in k_part:
                    if len(jp) == 2 and len(kp) == 2:
                        j0, j1 = jp
                        k0, k1 = kp
                        # Standard Strassen 2x2x2
                        # P1
                        U[0*m + j0, col] += 1
                        U[1*m + j1, col] += 1
                        V[j0*p + k0, col] += 1
                        V[j1*p + k1, col] += 1
                        W[k0*n + 0, col] += 1
                        W[k1*n + 1, col] += 1
                        col += 1
                        # P2
                        U[1*m + j0, col] += 1
                        U[1*m + j1, col] += 1
                        V[j0*p + k0, col] += 1
                        W[k0*n + 1, col] += 1
                        W[k1*n + 1, col] -= 1
                        col += 1
                        # P3
                        U[0*m + j0, col] += 1
                        V[j0*p + k1, col] += 1
                        V[j1*p + k1, col] -= 1
                        W[k1*n + 0, col] += 1
                        W[k1*n + 1, col] += 1
                        col += 1
                        # P4
                        U[1*m + j1, col] += 1
                        V[j1*p + k0, col] += 1
                        V[j0*p + k0, col] -= 1
                        W[k0*n + 0, col] += 1
                        W[k0*n + 1, col] += 1
                        col += 1
                        # P5
                        U[0*m + j0, col] += 1
                        U[0*m + j1, col] += 1
                        V[j1*p + k1, col] += 1
                        W[k0*n + 0, col] -= 1
                        W[k1*n + 0, col] += 1
                        col += 1
                        # P6
                        U[1*m + j0, col] += 1
                        U[0*m + j0, col] -= 1
                        V[j0*p + k0, col] += 1
                        V[j0*p + k1, col] += 1
                        W[k1*n + 1, col] += 1
                        col += 1
                        # P7
                        U[0*m + j1, col] += 1
                        U[1*m + j1, col] -= 1
                        V[j1*p + k0, col] += 1
                        V[j1*p + k1, col] += 1
                        W[k0*n + 0, col] += 1
                        col += 1
                    elif len(jp) == 1 and len(kp) == 1:
                        j = jp[0]
                        k = kp[0]
                        for i in range(n):
                            U[i*m + j, col] += 1
                            V[j*p + k, col] += 1
                            W[k*n + i, col] += 1
                            col += 1
                    elif len(jp) == 2 and len(kp) == 1:
                        j0, j1 = jp
                        k = kp[0]
                        for j in [j0, j1]:
                            for i in range(n):
                                U[i*m + j, col] += 1
                                V[j*p + k, col] += 1
                                W[k*n + i, col] += 1
                                col += 1
                    elif len(jp) == 1 and len(kp) == 2:
                        j = jp[0]
                        k0, k1 = kp
                        for k in [k0, k1]:
                            for i in range(n):
                                U[i*m + j, col] += 1
                                V[j*p + k, col] += 1
                                W[k*n + i, col] += 1
                                col += 1
            
            if col != rank:
                continue
            
            if _verify_decomposition_fast(U, V, W, n, m, p):
                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known 54-rank decomposition
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    # Verify the initial decomposition
    if not _verify_decomposition_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    else:
        # Try to eliminate terms
        if _time_left() > 40:
            U2, V2, W2, rank2 = _try_eliminate_terms(U, V, W, n, m, p, max_attempts=300)
            if rank2 < best_rank:
                best_U, best_V, best_W = U2, V2, W2
                best_rank = rank2
        
        # Try to merge terms
        if _time_left() > 30:
            U4, V4, W4, rank4 = _try_merge_terms(best_U, best_V, best_W, n, m, p)
            if rank4 < best_rank:
                best_U, best_V, best_W = U4, V4, W4
                best_rank = rank4
        
        # Try different partitions
        if _time_left() > 25:
            U3, V3, W3, rank3 = _try_different_partitions(n, m, p)
            if U3 is not None and rank3 < best_rank:
                best_U, best_V, best_W = U3, V3, W3
                best_rank = rank3
    
    # Return the best result found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
