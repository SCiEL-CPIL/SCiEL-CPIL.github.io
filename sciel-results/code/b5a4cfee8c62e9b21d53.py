"""Rewritten program for matrix multiplication tensor decomposition 3x4x6."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_schoolbook(n, m, p):
    """Build the schoolbook (trivial) decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition.
    
    The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k.
    A rank-r decomposition satisfies:
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T_computed = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    T_expected = np.ones((n, m, p))
    return np.allclose(T_computed, T_expected, atol=atol)

def _apply_strassen_2x2x2(U, V, W, n, m, p, i0, j0, k0):
    """Try to replace 8 schoolbook terms in a 2x2x2 block with 7 Strassen terms.
    
    Returns (success, new_U, new_V, new_W) or (False, None, None, None).
    """
    rank = U.shape[1]
    block_cols = []
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i, j, k = i0 + di, j0 + dj, k0 + dk
                # Find column c where U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
                # and all other entries in that column are 0 (or close to 0)
                found = False
                for c in range(rank):
                    if abs(U[i*m+j, c] - 1.0) < 1e-12 and \
                       abs(V[j*p+k, c] - 1.0) < 1e-12 and \
                       abs(W[k*n+i, c] - 1.0) < 1e-12:
                        # Check it's a pure schoolbook term
                        u_col = U[:, c].copy()
                        v_col = V[:, c].copy()
                        w_col = W[:, c].copy()
                        u_col[i*m+j] = 0
                        v_col[j*p+k] = 0
                        w_col[k*n+i] = 0
                        if np.max(np.abs(u_col)) < 1e-12 and \
                           np.max(np.abs(v_col)) < 1e-12 and \
                           np.max(np.abs(w_col)) < 1e-12:
                            block_cols.append(c)
                            found = True
                            break
                if not found:
                    return False, None, None, None
    
    if len(block_cols) != 8:
        return False, None, None, None
    
    # Now build the new U, V, W with 7 Strassen terms replacing these 8.
    new_U = U.copy()
    new_V = V.copy()
    new_W = W.copy()
    
    # Zero out the 8 schoolbook columns
    for c in block_cols:
        new_U[:, c] = 0
        new_V[:, c] = 0
        new_W[:, c] = 0
    
    # Define the 7 Strassen terms
    strassen_terms = []
    
    # M1
    u1 = {(i0, j0): 1, (i0+1, j0): 1}
    v1 = {(j0, k0): 1, (j0+1, k0): 1}
    w1 = {(k0, i0): 1, (k0+1, i0+1): 1}
    strassen_terms.append((u1, v1, w1))
    
    # M2
    u2 = {(i0, j0+1): 1, (i0+1, j0+1): 1}
    v2 = {(j0, k0): 1}
    w2 = {(k0, i0+1): 1}
    strassen_terms.append((u2, v2, w2))
    
    # M3
    u3 = {(i0, j0): 1}
    v3 = {(j0+1, k0): 1, (j0, k0): -1}
    w3 = {(k0+1, i0): 1, (k0, i0+1): -1}
    strassen_terms.append((u3, v3, w3))
    
    # M4
    u4 = {(i0+1, j0): 1}
    v4 = {(j0+1, k0+1): 1, (j0, k0): -1}
    w4 = {(k0+1, i0): -1, (k0+1, i0+1): 1}
    strassen_terms.append((u4, v4, w4))
    
    # M5
    u5 = {(i0, j0): 1, (i0, j0+1): 1}
    v5 = {(j0+1, k0+1): 1}
    w5 = {(k0+1, i0): 1, (k0+1, i0+1): 1}
    strassen_terms.append((u5, v5, w5))
    
    # M6
    u6 = {(i0+1, j0+1): 1, (i0, j0): -1}
    v6 = {(j0, k0): 1, (j0, k0+1): 1}
    w6 = {(k0, i0): 1, (k0, i0+1): -1}
    strassen_terms.append((u6, v6, w6))
    
    # M7
    u7 = {(i0, j0+1): 1, (i0+1, j0+1): -1}
    v7 = {(j0+1, k0+1): 1, (j0+1, k0): -1}
    w7 = {(k0, i0): -1}
    strassen_terms.append((u7, v7, w7))
    
    # Add the 7 Strassen terms as new columns
    for idx, (u_dict, v_dict, w_dict) in enumerate(strassen_terms):
        c = block_cols[idx]
        for (i, j), coeff in u_dict.items():
            if 0 <= i < n and 0 <= j < m:
                new_U[i * m + j, c] = coeff
        for (j, k), coeff in v_dict.items():
            if 0 <= j < m and 0 <= k < p:
                new_V[j * p + k, c] = coeff
        for (k, i), coeff in w_dict.items():
            if 0 <= k < p and 0 <= i < n:
                new_W[k * n + i, c] = coeff
    
    # Zero out the 8th column (we only use 7)
    c8 = block_cols[7]
    new_U[:, c8] = 0
    new_V[:, c8] = 0
    new_W[:, c8] = 0
    
    # Verify
    if _verify_decomposition(new_U, new_V, new_W, n, m, p):
        return True, new_U, new_V, new_W
    else:
        return False, None, None, None

def _apply_strassen_2x2x3(U, V, W, n, m, p, i0, j0, k0):
    """Try to replace 12 schoolbook terms in a 2x2x3 block with fewer Strassen terms.
    
    This is more complex; for now, we'll try to apply two overlapping 2x2x2 Strassen blocks
    if they don't conflict.
    
    Returns (success, new_U, new_V, new_W) or (False, None, None, None).
    """
    # For a 2x2x3 block, we can potentially apply two 2x2x2 Strassen blocks:
    # One at (i0, j0, k0) covering k in {k0, k0+1}
    # One at (i0, j0, k0+1) covering k in {k0+1, k0+2}
    # But they share the column at k=k0+1, so this is tricky.
    
    # Instead, let's try a different approach: apply Strassen to (i0, j0, k0) 
    # and then try to apply Strassen to (i0, j0, k0+2) if it's a 2x2x2 block
    # that doesn't overlap with the first one in a conflicting way.
    
    # Actually, for simplicity and correctness, let's just try to apply
    # non-overlapping 2x2x2 Strassen blocks in sequence.
    
    # For a 2x2x3 block at (i0, j0, k0), the 12 schoolbook terms are at:
    # (i0+di, j0+dj, k0+dk) for di in {0,1}, dj in {0,1}, dk in {0,1,2}
    
    # We can potentially apply:
    # 1. Strassen at (i0, j0, k0) for k in {k0, k0+1}
    # 2. Then we have remaining terms at k=k0+2 for i in {i0, i0+1}, j in {j0, j0+1}
    #    That's 4 terms, which is just a 2x2x1 block - no Strassen benefit.
    
    # Alternatively, we could try a custom 2x2x3 decomposition.
    # For now, return False to indicate this block type isn't handled.
    return False, None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook decomposition
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = U.shape[1]
    
    # Try to apply Strassen to all possible 2x2x2 blocks
    # A 2x2x2 block at (i0, j0, k0) covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # Valid positions: i0 in {0,1}, j0 in {0,1,2}, k0 in {0,1,2,3,4}
    
    # We need to be careful about overlapping blocks. Let's try a greedy approach:
    # Process blocks in a specific order, and only apply Strassen if all 8
    # corresponding schoolbook terms are still present (not already modified).
    
    # Try all possible blocks, but track which have been applied
    applied_blocks = set()
    
    # Sort blocks to try non-overlapping ones first
    # For simplicity, try all and let the verification handle conflicts
    
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                if time.time() > deadline - 5:
                    break
                
                # Check if this block overlaps with an already applied block
                # A block at (i0, j0, k0) overlaps with (i1, j1, k1) if
                # they share any (i,j,k) indices
                overlaps = False
                for (ai0, aj0, ak0) in applied_blocks:
                    # Check if the two 2x2x2 blocks share any (i,j,k)
                    i_set1 = {i0, i0+1}
                    i_set2 = {ai0, ai0+1}
                    j_set1 = {j0, j0+1}
                    j_set2 = {aj0, aj0+1}
                    k_set1 = {k0, k0+1}
                    k_set2 = {ak0, ak0+1}
                    
                    if i_set1 & i_set2 and j_set1 & j_set2 and k_set1 & k_set2:
                        overlaps = True
                        break
                
                if overlaps:
                    continue
                
                success, new_U, new_V, new_W = _apply_strassen_2x2x2(U, V, W, n, m, p, i0, j0, k0)
                if success:
                    U, V, W = new_U, new_V, new_W
                    applied_blocks.add((i0, j0, k0))
                    best_rank = U.shape[1]
    
    # Return the best result found
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
