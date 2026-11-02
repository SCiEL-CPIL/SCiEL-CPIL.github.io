"""Structural change: greedy coordinate descent over a rich Strassen-style basis
with exact rational validation for 4x5 by 5x6 matrix multiplication."""

import time
from fractions import Fraction
import random
import sys

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_basis(n, m, p):
    """
    Build a rich basis of candidate rank-1 tensors for the 4x5x6 problem.
    Each candidate is (U_vec, V_vec, W_vec) where:
      U_vec has length n*m, V_vec has length m*p, W_vec has length p*n.
    
    The basis is constructed from products of small "pattern" vectors that
    include Strassen-style combinations (sums/differences of coordinates).
    """
    candidates = []
    
    # Define atomic patterns for U (length n*m = 20), V (length m*p = 30), W (length p*n = 24)
    # We'll build patterns by combining row/column selections with signs.
    
    # For U: index i*m + j, i in 0..n-1, j in 0..m-1
    # For V: index j*p + k, j in 0..m-1, k in 0..p-1
    # For W: index k*n + i, k in 0..p-1, i in 0..n-1
    
    # Strassen-style: for 2x2, we have patterns like (1,1), (1,-1) etc.
    # Generalize: for each "block" of rows/cols, try sum and difference.
    
    def make_u_vec(i_set, sign_map):
        """Create U vector from a set of (i,j) pairs with signs."""
        vec = [0] * (n * m)
        for (i, j), s in sign_map.items():
            vec[i * m + j] = s
        return vec
    
    def make_v_vec(j_set, sign_map):
        """Create V vector from a set of (j,k) pairs with signs."""
        vec = [0] * (m * p)
        for (j, k), s in sign_map.items():
            vec[j * p + k] = s
        return vec
    
    def make_w_vec(k_set, sign_map):
        """Create W vector from a set of (k,i) pairs with signs."""
        vec = [0] * (p * n)
        for (k, i), s in sign_map.items():
            vec[k * n + i] = s
        return vec
    
    # Generate candidates using structured combinations.
    # Strategy: for each "scale" of sub-problem, generate Strassen-like patterns.
    
    # Level 0: single elements (schoolbook)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                candidates.append((u, v, w))
    
    # Level 1: Strassen-style 2x2 blocks along each dimension.
    # For U: pair of rows (i1, i2) x single col j: (1,1) and (1,-1)
    # For V: single row j x pair of cols (k1, k2): (1,1) and (1,-1)
    # For W: pair of cols k x single row i: (1,1) and (1,-1)
    
    # We'll generate combinations where one dimension has a 2-element Strassen pattern
    # and the others are single.
    
    def add_strassen_candidate(u_pattern, v_pattern, w_pattern):
        """Add a candidate from given pattern dicts."""
        u = [0] * (n * m)
        for idx, s in u_pattern.items():
            u[idx] = s
        v = [0] * (m * p)
        for idx, s in v_pattern.items():
            v[idx] = s
        w = [0] * (p * n)
        for idx, s in w_pattern.items():
            w[idx] = s
        # Skip if all zero
        if any(u) and any(v) and any(w):
            candidates.append((u, v, w))
    
    # Strassen along i (rows of A / cols of W)
    for i1 in range(0, n - 1, 2):
        i2 = i1 + 1
        for j in range(m):
            for k in range(p):
                # Sum pattern
                add_strassen_candidate(
                    {i1 * m + j: 1, i2 * m + j: 1},
                    {j * p + k: 1},
                    {k * n + i1: 1, k * n + i2: 1}
                )
                # Diff pattern
                add_strassen_candidate(
                    {i1 * m + j: 1, i2 * m + j: -1},
                    {j * p + k: 1},
                    {k * n + i1: 1, k * n + i2: -1}
                )
    
    # Strassen along j (cols of A / rows of B)
    for i in range(n):
        for j1 in range(0, m - 1, 2):
            j2 = j1 + 1
            for k in range(p):
                # Sum pattern
                add_strassen_candidate(
                    {i * m + j1: 1, i * m + j2: 1},
                    {j1 * p + k: 1, j2 * p + k: 1},
                    {k * n + i: 1}
                )
                # Diff pattern
                add_strassen_candidate(
                    {i * m + j1: 1, i * m + j2: -1},
                    {j1 * p + k: 1, j2 * p + k: -1},
                    {k * n + i: 1}
                )
    
    # Strassen along k (cols of B / rows of W)
    for i in range(n):
        for j in range(m):
            for k1 in range(0, p - 1, 2):
                k2 = k1 + 1
                # Sum pattern
                add_strassen_candidate(
                    {i * m + j: 1},
                    {j * p + k1: 1, j * p + k2: 1},
                    {k1 * n + i: 1, k2 * n + i: 1}
                )
                # Diff pattern
                add_strassen_candidate(
                    {i * m + j: 1},
                    {j * p + k1: 1, j * p + k2: -1},
                    {k1 * n + i: 1, k2 * n + i: -1}
                )
    
    # Level 2: 2D Strassen patterns (e.g., 2x2 blocks in two dimensions)
    # This gets expensive, so we'll be selective.
    # For 4x4x4 Strassen, we have 7 multiplications. Let's generalize.
    
    # For now, add some 2x2x2 Strassen patterns at key positions.
    # The classic 2x2 Strassen has 7 patterns. Let's add them for each 2x2x2 sub-block.
    
    def add_2x2x2_strassen(i_base, j_base, k_base):
        """Add 2x2x2 Strassen patterns for sub-blocks starting at (i_base, j_base, k_base)."""
        i1, i2 = i_base, i_base + 1
        j1, j2 = j_base, j_base + 1
        k1, k2 = k_base, k_base + 1
        
        if i2 >= n or j2 >= m or k2 >= p:
            return
        
        # The 7 Strassen products for 2x2x2:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        u_idx = lambda i, j: i * m + j
        v_idx = lambda j, k: j * p + k
        w_idx = lambda k, i: k * n + i
        
        patterns = [
            # M1
            ({u_idx(i1,j1):1, u_idx(i2,j2):1}, {v_idx(j1,k1):1, v_idx(j2,k2):1}, {w_idx(k1,i1):1, w_idx(k2,i2):1, w_idx(k1,i2):-1, w_idx(k2,i1):-1}),
            # M2
            ({u_idx(i2,j1):1, u_idx(i2,j2):1}, {v_idx(j1,k1):1}, {w_idx(k1,i2):1}),
            # M3
            ({u_idx(i1,j1):1}, {v_idx(j1,k2):1, v_idx(j2,k2):-1}, {w_idx(k2,i1):1}),
            # M4
            ({u_idx(i2,j2):1}, {v_idx(j2,k1):1, v_idx(j1,k1):-1}, {w_idx(k1,i2):1}),
            # M5
            ({u_idx(i1,j1):1, u_idx(i1,j2):1}, {v_idx(j2,k2):1}, {w_idx(k2,i1):1}),
            # M6
            ({u_idx(i2,j1):1, u_idx(i1,j1):-1}, {v_idx(j1,k1):1, v_idx(j1,k2):1}, {w_idx(k1,i2):1, w_idx(k2,i2):1}),
            # M7
            ({u_idx(i1,j2):1, u_idx(i2,j2):-1}, {v_idx(j2,k1):1, v_idx(j2,k2):1}, {w_idx(k1,i1):1, w_idx(k2,i1):1}),
        ]
        
        for u_pat, v_pat, w_pat in patterns:
            add_strassen_candidate(u_pat, v_pat, w_pat)
    
    # Add 2x2x2 Strassen patterns for all valid positions
    for i_base in range(0, n - 1, 2):
        for j_base in range(0, m - 1, 2):
            for k_base in range(0, p - 1, 2):
                add_2x2x2_strassen(i_base, j_base, k_base)
    
    # Also add shifted versions to cover odd positions
    for i_base in range(1, n - 1, 2):
        for j_base in range(0, m - 1, 2):
            for k_base in range(0, p - 1, 2):
                if i_base + 1 < n:
                    add_2x2x2_strassen(i_base, j_base, k_base)
    
    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for u, v, w in candidates:
        key = (tuple(u), tuple(v), tuple(w))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((u, v, w))
    
    return unique_candidates


def _validate_decomposition(U, V, W, n, m, p, deadline):
    """
    Validate that the decomposition exactly reproduces matrix multiplication.
    For all i, k: C[i,k] = sum_j A[i,j]*B[j,k]
    The decomposition gives: C[i,k] = sum_r U[i*m + ?, r] * V[?, r] * W[k*n+i, r]
    
    Actually, the tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    This must equal 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k],
    which is always the case for the bilinear map. Wait, no.
    
    The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k because
    C[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) term has coefficient 1.
    
    So we need: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    return False
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if s != 1:
                    return False
    return True


def _greedy_search(candidates, n, m, p, max_rank, deadline):
    """
    Greedy coordinate descent: start with empty decomposition, iteratively add
    the candidate that best reduces the residual.
    
    The residual at (i,j,k) is: 1 - sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r]
    """
    best_result = None
    best_rank = n * m * p  # schoolbook rank as fallback
    
    # Initialize with schoolbook as the current best
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find a better decomposition
    for target_rank in range(30, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Greedy: build decomposition one tensor at a time
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        residual = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual[(i,j,k)] = Fraction(1)
        
        used = []
        
        for r in range(target_rank):
            if time.time() > deadline:
                break
            
            best_cand = None
            best_score = None
            
            # Try a subset of candidates for speed
            subset_size = min(len(candidates), 500)
            subset = random.sample(candidates, subset_size)
            
            for cand_idx, (u_pat, v_pat, w_pat) in enumerate(subset):
                if time.time() > deadline:
                    break
                
                # Compute the "coverage" of this candidate: how many residual
                # entries it can help with.
                # The candidate contributes u_pat[i*m+j] * v_pat[j*p+k] * w_pat[k*n+i]
                # to the (i,j,k) entry.
                
                # Estimate score: sum of |residual[(i,j,k)] * contribution|
                # over all (i,j,k) where contribution is nonzero.
                
                score = Fraction(0)
                for i in range(n):
                    for j in range(m):
                        u_val = u_pat[i*m+j]
                        if u_val == 0:
                            continue
                        for k in range(p):
                            v_val = v_pat[j*p+k]
                            if v_val == 0:
                                continue
                            w_val = w_pat[k*n+i]
                            if w_val == 0:
                                continue
                            contrib = u_val * v_val * w_val
                            if contrib != 0:
                                score += abs(residual[(i,j,k)] * contrib)
                
                if best_score is None or score > best_score:
                    best_score = score
                    best_cand = cand_idx
            
            if best_cand is None or best_score is None or best_score == 0:
                break
            
            u_pat, v_pat, w_pat = subset[best_cand]
            
            # Add this candidate to the decomposition
            for i in range(n):
                for j in range(m):
                    U[i*m+j][r] = u_pat[i*m+j]
                for k in range(p):
                    V[j*p+k][r] = v_pat[j*p+k]
                for i in range(n):
                    W[k*n+i][r] = w_pat[k*n+i]
            
            # Update residual
            for i in range(n):
                for j in range(m):
                    u_val = u_pat[i*m+j]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = v_pat[j*p+k]
                        if v_val == 0:
                            continue
                        w_val = w_pat[k*n+i]
                        if w_val == 0:
                            continue
                        contrib = u_val * v_val * w_val
                        residual[(i,j,k)] -= contrib
            
            used.append(best_cand)
        
        # Check if we achieved exact decomposition
        if _validate_decomposition(U, V, W, n, m, p, deadline):
            # Convert to lists of ints
            U_list = [[int(x) for x in row] for row in U]
            V_list = [[int(x) for x in row] for row in V]
            W_list = [[int(x) for x in row] for row in W]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = target_rank
            break
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_result, best_rank


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 by 5x6 matrix multiplication.
    Uses greedy coordinate descent over a Strassen-style basis with exact validation.
    Falls back to schoolbook (rank 120) if no better solution is found in time.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Build candidate basis
    candidates = _build_basis(n, m, p)
    
    # Try to find a better decomposition
    result, rank = _greedy_search(candidates, n, m, p, max_rank=110, deadline=deadline)
    
    # If greedy didn't find anything better, use schoolbook
    if rank >= n * m * p:
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
    
    return result
# EVOLVE-BLOCK-END
