"""
Strategy:
1. Base: 44-term decomposition (4 Strassen blocks for 2x2x8 + schoolbook for 2x1x8).
2. Try to reduce rank by finding a better decomposition.
3. Use known results: rank(2x3x8) is at most 44.
4. Try a different partitioning: 3x2 blocks instead of 2x2 blocks.
   For 2x3x8, we can view it as:
   - Row 0 of A (2 elements) times columns of B
   - Row 1 of A (2 elements) times columns of B
   But a better approach: partition the 3 columns of A into 2+1.
   
   Actually, let's think about this differently.
   A is 2x3, B is 3x8.
   
   We can partition the middle dimension (size 3) into 2+1.
   For the 2x2x8 part: use Strassen on 2x2 blocks, giving 7 terms per 2x2x2 block,
   so 7*4 = 28 terms for 2x2x8.
   For the 2x1x8 part: schoolbook, giving 16 terms.
   Total: 44.
   
   Alternatively, partition as 1+2:
   For the 1x2x8 part: schoolbook, giving 16 terms.
   For the 2x2x8 part: Strassen, giving 28 terms.
   Total: 44.
   
   What if we use a different algorithm for the 2x2x8 part?
   The rank of 2x2x8 matrix multiplication is at most 7*4 = 28 (using Strassen on each 2x2x2 block).
   But maybe there's a better decomposition for 2x2x8?
   
   Known: rank(2x2x4) <= 14 (using Strassen on two 2x2x2 blocks).
   Actually, the rank of 2x2x4 is at most 14, but it might be lower.
   
   Let me think about whether we can share terms between the Strassen blocks.
   In the current construction, each 2x2x2 Strassen block uses 7 independent terms.
   Since the blocks operate on different output columns (k=0,1 for block 0; k=2,3 for block 1; etc.),
   the W vectors have disjoint support, so the terms are independent.
   
   Therefore, 44 is likely optimal for this construction family.
   
   Let's try a completely different approach: use the fact that 3 = 2+1 and try
   to find a decomposition that shares terms between the 2x2x8 and 2x1x8 parts.
   
   For the 2x1x8 part, the terms are:
   t_{i,k}^{(1)}: U[i*3+2, r] = 1, V[2*8+k, r] = 1, W[k*2+i, r] = 1
   
   For the Strassen terms in block 0 (k=0,1):
   They have support on U[i*3+a] for a in {0,1}, V[a*8+k] for a in {0,1}, k in {0,1},
   and W[k*2+i] for k in {0,1}.
   
   The key question: can we modify the Strassen terms to also contribute to the a=2 part?
   If we add a small component to U[i*3+2, r] and V[2*8+k, r] for some Strassen terms,
   we might be able to reduce the number of dedicated 2x1x8 terms.
   
   But this would change the Strassen decomposition and likely break it.
   
   Given the time constraints, I'll return the 44-term decomposition as the best known.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Build 44-term decomposition
    rank_44 = 44
    U_44 = [[0] * rank_44 for _ in range(nm)]
    V_44 = [[0] * rank_44 for _ in range(mp)]
    W_44 = [[0] * rank_44 for _ in range(pn)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_44[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_44[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_44[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(n):
        for k in range(p):
            U_44[i * m + 2][col] = 1
            V_44[2 * p + k][col] = 1
            W_44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Verify 44-term
    candidate_44 = {"U": U_44, "V": V_44, "W": W_44}
    if not _validate(candidate_44):
        return _schoolbook()

    # Try to find a 43-term decomposition by checking if any term is redundant.
    # Use exact rational arithmetic for the check.
    
    # For each term r, check if it can be expressed as a linear combination of others.
    # We test this by checking if the rank-1 tensor t_r is in the span of {t_s : s != r}.
    
    # To do this efficiently, we pick a set of (i,a,k) indices and check if the
    # vector of t_r values at those indices is in the span of the corresponding
    # vectors for other terms.
    
    # Use all 48 entries (2*3*8) for an exact check.
    all_indices = [(i, a, k) for i in range(n) for a in range(m) for k in range(p)]
    
    # Build the tensor matrix: T[idx, r] = t_r[idx]
    T = np.zeros((len(all_indices), rank_44), dtype=object)
    for idx, (i, a, k) in enumerate(all_indices):
        for r in range(rank_44):
            T[idx, r] = Fraction(U_44[i * m + a][r]) * Fraction(V_44[a * p + k][r]) * Fraction(W_44[k * n + i][r])

    # For each term r, check if column r is in the span of other columns using exact Gaussian elimination.
    # This is expensive, so we only do it for a subset of terms.
    
    # Instead, let's try a simpler approach: check if removing any single term
    # still allows us to represent the tensor.
    
    # For each r, try to solve T_without_r * c = T[:, r] exactly.
    # If a solution exists, then term r is redundant.
    
    # Use numpy with float64 for speed, then verify with exact arithmetic.
    T_float = np.array(T, dtype=float)
    
    redundant_terms = []
    for r in range(rank_44):
        if not _elapsed_ok():
            break
        mask = [s for s in range(rank_44) if s != r]
        M_reduced = T_float[:, mask]
        b = T_float[:, r]
        
        try:
            c, residuals, rank_M, sv = np.linalg.lstsq(M_reduced, b, rcond=None)
        except:
            continue
        
        residual = np.linalg.norm(b - M_reduced @ c)
        norm_b = np.linalg.norm(b)
        if norm_b > 1e-10 and residual / norm_b < 1e-8:
            # Verify with exact arithmetic
            # c is approximate; round to nearest fraction
            c_exact = []
            valid = True
            for ci in c:
                frac = Fraction(ci).limit_denominator(10000)
                c_exact.append(frac)
            
            # Check if M_reduced * c_exact == b exactly
            ok = True
            for idx in range(len(all_indices)):
                acc = Fraction(0)
                for j, s in enumerate(mask):
                    acc += T[idx, s] * c_exact[j]
                if acc != T[idx, r]:
                    ok = False
                    break
            if ok:
                redundant_terms.append(r)

    if redundant_terms:
        # Remove redundant terms
        keep = [r for r in range(rank_44) if r not in redundant_terms]
        new_rank = len(keep)
        U_new = [[U_44[i][r] for r in keep] for i in range(nm)]
        V_new = [[V_44[i][r] for r in keep] for i in range(mp)]
        W_new = [[W_44[i][r] for r in keep] for i in range(pn)]
        
        candidate_new = {"U": U_new, "V": V_new, "W": W_new}
        if _validate(candidate_new):
            return candidate_new

    return candidate_44

# EVOLVE-BLOCK-END
