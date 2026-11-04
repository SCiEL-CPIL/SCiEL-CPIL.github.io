"""
Strategy: Combine Strassen 2x2x2 blocks with polynomial evaluation/interpolation
for the 8-column dimension.

Approach:
1. Partition middle dimension (size 3) into 2+1.
2. For 2x2x8 part: Use Strassen for 2x2 core, and polynomial methods for the 8 columns.
   - Use Toom-5 (5 points) + Toom-3 (3 points) for the 8-column dimension.
   - This gives rank 5*7 + 3*7 = 56... that's worse.
   
   Alternative: Use a single polynomial of degree 7 evaluated at 8 points.
   The rank for multiplying a 2x2 matrix by a degree-7 polynomial is at most 7*8 = 56
   using schoolbook, but can be reduced.
   
   Actually, let's think differently. The 2x2x8 tensor can be seen as:
   For each of the 4 output positions (i,k), we need sum_a A[i,a]*B[a,k].
   
   Using Strassen on the 2x2 part gives 7 multiplications per pair of k values.
   For 8 k values, we have 4 pairs, giving 28 terms.
   
   For the 2x1x8 part: 16 schoolbook terms.
   Total: 44.
   
   Can we do better? Let's try to share terms between the 2x2x8 and 2x1x8 parts.
   
   The 2x1x8 terms are: for each i,k, term with U[i*3+2]=1, V[2*8+k]=1, W[k*2+i]=1.
   
   If we modify Strassen terms to also contribute to a=2, we might reduce total rank.
   
   Let's try: For each Strassen term r, add alpha_r to U[i*3+2, r] and beta_r to V[2*8+k, r].
   The constraint is that the total contribution to each (i,a,k) must still be 1.
   
   For a=0,1: The Strassen terms already give the correct value. Adding a=2 components
   doesn't affect them (since V[a*8+k] for a=0,1 is unchanged, and U[i*3+a] for a=0,1
   is unchanged).
   
   For a=2: We need sum_r U[i*3+2, r]*V[2*8+k, r]*W[k*2+i, r] = 1 for all i,k.
   
   Currently W[k*2+i, r] is only nonzero for the Strassen terms that involve output (k,i).
   Each output (k,i) is involved in exactly one Strassen block (the one containing k).
   Within that block, W[k*2+i, r] takes specific values.
   
   This is complex. Let me just implement the 44-term solution robustly and try
   a simple optimization: check if any term can be eliminated.
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

    # Strassen 2x2x2
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

    candidate_44 = {"U": U_44, "V": V_44, "W": W_44}
    if not _validate(candidate_44):
        return _schoolbook()

    # Try to find a 43-term decomposition by checking if any term is redundant.
    all_indices = [(i, a, k) for i in range(n) for a in range(m) for k in range(p)]
    
    # Build tensor matrix T[idx, r] = t_r[idx]
    T = np.zeros((len(all_indices), rank_44), dtype=object)
    for idx, (i, a, k) in enumerate(all_indices):
        for r in range(rank_44):
            T[idx, r] = Fraction(U_44[i * m + a][r]) * Fraction(V_44[a * p + k][r]) * Fraction(W_44[k * n + i][r])

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
            c_exact = []
            valid = True
            for ci in c:
                frac = Fraction(ci).limit_denominator(10000)
                c_exact.append(frac)
            
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
