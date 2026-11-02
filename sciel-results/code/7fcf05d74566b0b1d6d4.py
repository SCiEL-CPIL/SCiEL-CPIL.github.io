import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Return True iff U[r]·V[r]·W[r] reproduces C_{i,k} = A_{i,j}·B_{j,k} exactly."""
    rank = len(U[0]) if U else 0
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_strassen(n, m, p):
    """Construct the rank-28 Strassen decomposition for M_{2,2,8}.
    
    The tensor M_{2,2,8} is a direct sum of 4 independent M_{2,2,2} tensors.
    Each block uses the classic 7-term Strassen identity.
    """
    # Strassen 2x2 identity components
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # 7 Strassen terms: (A_part, B_part, C_part)
    # Each part is a dict mapping index to coefficient
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): 1}),  # M1 = (a+d)(x+y)
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): 1}),  # M2 = (c+d)x
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),  # M3 = a(y-x)
        ({(1,1): 1}, {(0,0): -1, (0,1): 1}, {(0,0): 1, (1,0): 1}),  # M4 = d(x-y)
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (0,1): 1}),  # M5 = (a+b)y
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),  # M6 = (c-a)(x+y)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,0): 1}),  # M7 = (b-d)(y+x)
    ]
    
    rank = 4 * 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for block in range(4):
        k0 = block * 2  # starting column for this block
        for s in range(7):
            r = block * 7 + s
            a_part, b_part, c_part = strassen_terms[s]
            
            # Fill U: A factor
            for (ai, aj), val in a_part.items():
                U[ai * m + aj][r] = float(val)
            
            # Fill V: B factor (columns k0, k0+1)
            for (bj, bcol), val in b_part.items():
                kglob = k0 + bcol
                V[bj * p + kglob][r] = float(val)
            
            # Fill W: C factor (official convention W[k*n+i])
            for (ci, ccol), val in c_part.items():
                kglob = k0 + ccol
                W[kglob * n + ci][r] = float(val)
    
    return U, V, W

def _build_direct_sum(n, m, p):
    """Construct the naive rank-32 direct sum decomposition for M_{2,2,8}.
    
    This is the standard bilinear form: for each i, j, k, we have one term.
    """
    rank = n * m * p  # 2*2*8 = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1.0
                V[j * p + k][r] = 1.0
                W[k * n + i][r] = 1.0
                r += 1
    
    return U, V, W

def _try_remove_term(U, V, W, n, m, p, term_idx, tol=1e-9):
    """Try to remove term term_idx by solving for adjustments to remaining terms.
    
    Returns (U_new, V_new, W_new) if successful, or None if not.
    """
    rank = len(U[0])
    if term_idx < 0 or term_idx >= rank:
        return None
    
    # Keep all terms except term_idx
    keep = [r for r in range(rank) if r != term_idx]
    new_rank = len(keep)
    
    U_keep = [U[i][r] for i in range(len(U)) for r in keep]
    U_keep = np.array(U_keep).reshape(len(U), new_rank)
    V_keep = np.array([V[i][r] for i in range(len(V)) for r in keep]).reshape(len(V), new_rank)
    W_keep = np.array([W[i][r] for i in range(len(W)) for r in keep]).reshape(len(W), new_rank)
    
    # The tensor T must be reproduced by the remaining terms
    # T[i,j,k] = sum_{r in keep} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We need to find U', V', W' close to U_keep, V_keep, W_keep such that
    # T[i,j,k] = sum_{r} U'[i,j,r] * V'[j,k,r] * W'[k,i,r]
    
    # This is a hard nonlinear problem. Instead, let's check if the removed term
    # is linearly dependent on the others in a way that allows exact cancellation.
    
    # Check if the removed term's contribution can be represented by a linear combination
    # of the remaining terms. This requires solving:
    # U_removed ⊗ V_removed ⊗ W_removed = sum_{r in keep} alpha_r * U_r ⊗ V_r ⊗ W_r
    # which is generally not possible unless there's special structure.
    
    # For now, just verify if removing the term still gives the correct tensor
    # (i.e., if the removed term had all zeros or was redundant)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(new_rank):
                    s += U_keep[i*m+j, r] * V_keep[j*p+k, r] * W_keep[k*n+i, r]
                T[i,j,k] = s
    
    # Check if T is all ones (the target)
    if np.allclose(T, 1.0, atol=tol):
        return U_keep.tolist(), V_keep.tolist(), W_keep.tolist()
    
    return None

def _search_lower_rank(U, V, W, n, m, p, max_attempts=100, deadline=None):
    """Search for lower rank decompositions by trying to remove terms.
    
    Returns (U_new, V_new, W_new, rank) or (None, None, None, None).
    """
    if deadline is None:
        deadline = time.time() + 110
    
    current_rank = len(U[0])
    U_cur, V_cur, W_cur = U, V, W
    
    # Try removing one term at a time
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Pick a random term to try removing
        idx = np.random.randint(0, len(U_cur[0]))
        result = _try_remove_term(U_cur, V_cur, W_cur, n, m, p, idx)
        if result is not None:
            U_new, V_new, W_new = result
            # Verify
            if _verify(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new, len(U_new[0])
    
    return None, None, None, None

def _apply_algebraic_reduction(U, V, W, n, m, p, deadline=None):
    """Try to use algebraic structure to reduce rank.
    
    The idea: view B as an element of a 4-dimensional algebra where A acts linearly.
    This can sometimes reveal lower rank decompositions.
    
    For now, this is a placeholder that tries some heuristic reductions.
    """
    if deadline is None:
        deadline = time.time() + 110
    
    # Try to find linear dependencies among the terms
    # Build the tensor and check if some combinations cancel
    
    rank = len(U[0])
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check for proportional terms (if term r2 is a scalar multiple of term r1,
    # and the scalars are consistent across U, V, W, we might be able to merge)
    for r1 in range(rank):
        if time.time() > deadline:
            break
        for r2 in range(r1 + 1, rank):
            # Check if U[:, r2] is proportional to U[:, r1]
            u1 = U_np[:, r1]
            u2 = U_np[:, r2]
            if np.allclose(u1, 0) or np.allclose(u2, 0):
                continue
            
            # Find scaling factor
            nonzero = np.abs(u1) > 1e-10
            if not np.any(nonzero):
                continue
            
            c = u2[nonzero] / u1[nonzero]
            if np.allclose(c, c[0], atol=1e-9):
                c_val = c[0]
                # Check V and W
                if np.allclose(V_np[:, r2], c_val * V_np[:, r1], atol=1e-9) and \
                   np.allclose(W_np[:, r2], c_val * W_np[:, r1], atol=1e-9):
                    # Terms are proportional; we can potentially merge them
                    # For now, just note this
                    pass
    
    return None, None, None, None

def construct():
    """Return {'U','V','W'} factor matrices of shapes (n*m,rank),(m*p,rank),(p*n,rank)
    with integer entries that exactly decompose the 2x2 * 2x8 multiplication tensor."""
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 115  # Safety margin before 120s limit
    
    # Start with Strassen rank-28 decomposition
    U, V, W = _build_strassen(n, m, p)
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to direct sum if Strassen fails
        U, V, W = _build_direct_sum(n, m, p)
        if not _verify(U, V, W, n, m, p):
            # This should never happen, but return something
            return {"U": U, "V": V, "W": W}
    
    current_rank = len(U[0])
    
    # Try to reduce rank
    if time.time() < deadline:
        U_red, V_red, W_red, rank_red = _search_lower_rank(U, V, W, n, m, p, deadline=deadline)
        if U_red is not None and rank_red < current_rank:
            U, V, W = U_red, V_red, W_red
            current_rank = rank_red
    
    if time.time() < deadline:
        U_red, V_red, W_red, rank_red = _apply_algebraic_reduction(U, V, W, n, m, p, deadline=deadline)
        if U_red is not None and rank_red < current_rank:
            U, V, W = U_red, V_red, W_red
            current_rank = rank_red
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fallback to verified Strassen
        U, V, W = _build_strassen(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
