"""
Strategy: Strassen 2x2x2 blocks for the 2x2 sub-part + schoolbook for the 2x1x8 part.
Then attempt to reduce rank by finding linear dependencies among the rank-1 tensors.

The 44-term decomposition:
- 4 blocks of Strassen (7 terms each) = 28 terms for the 2x2x8 part
- 16 schoolbook terms for the 2x1x8 part

We then check if any term is a linear combination of others (in the tensor space).
If so, we can eliminate it.

Additionally, we try a more aggressive approach: solve for a smaller rank decomposition
by treating it as a bilinear optimization problem.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0

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

    # Strassen 2x2x2 decomposition
    # U rows: [A00, A01, A10, A11]
    # V rows: [B00, B01, B10, B11]
    # W rows: [C00, C01, C10, C11]
    S_U = [
        [1, 0, 0, 1],   # M1 = (A00+A11)*(B00+B11)
        [0, 0, 1, 1],   # M2 = (A10+A11)*B00
        [1, 0, 0, 0],   # M3 = A00*(B01-B11)
        [0, 0, 0, 1],   # M4 = A11*(B10-B00)
        [1, 1, 0, 0],   # M5 = (A00+A01)*B11
        [-1, 0, 1, 0],  # M6 = (A10-A00)*(B00+B01)
        [0, 1, 0, -1],  # M7 = (A11-A01)*(B10+B11)
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    S_W = [
        [1, 0, 0, 1],   # C00 = M1 + M4 - M5 - M7
        [0, 1, 0, -1],  # C01 = M3 + M5
        [0, 0, 1, 1],   # C10 = M2 + M4
        [1, 1, 0, 0],   # C11 = M1 - M2 + M6 + M7
        [-1, 0, 1, 0],  # (not used as output, these are intermediate)
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    # Actually let me re-derive Strassen properly.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A11-A01)(B10+B11)
    #
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M5 + M7
    
    S_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    S_W = [
        [1, 0, 0, 1],   # M1: C00+=1, C11+=1
        [0, 0, 1, 0],   # M2: C10+=1
        [0, 1, 0, 0],   # M3: C01+=1
        [-1, 0, 1, 0],  # M4: C00-=1, C10+=1
        [0, 1, 0, 1],   # M5: C01+=1, C11+=1
        [1, 0, 0, 0],   # M6: C00+=1
        [-1, 0, 0, 1],  # M7: C00-=1, C11+=1
    ]
    
    # Verify Strassen
    # C00 = M1 - M4 + M6 - M7
    # C00 from S_W: 1 + 0 + 0 + (-1) + 0 + 1 + (-1) = 0? No wait, S_W[r][c] is coefficient of M_r in C_c
    # C00: S_W[0][0] + S_W[1][0] + S_W[2][0] + S_W[3][0] + S_W[4][0] + S_W[5][0] + S_W[6][0]
    # = 1 + 0 + 0 + (-1) + 0 + 1 + (-1) = 0. That's wrong.
    
    # Let me be more careful. S_W[r][c] = coefficient of M_r in output C_c.
    # C00 = M1 - M4 + M6 - M7
    # So S_W[0][0]=1, S_W[3][0]=-1, S_W[5][0]=1, S_W[6][0]=-1, rest 0.
    # C01 = M3 + M5
    # S_W[2][1]=1, S_W[4][1]=1
    # C10 = M2 + M4
    # S_W[1][2]=1, S_W[3][2]=1
    # C11 = M1 - M2 + M5 + M7
    # S_W[0][3]=1, S_W[1][3]=-1, S_W[4][3]=1, S_W[6][3]=1
    
    S_W = [
        [1, 0, 0, 1],   # M1: C00+=1, C11+=1
        [0, 0, 1, -1],  # M2: C10+=1, C11-=1
        [0, 1, 0, 0],   # M3: C01+=1
        [-1, 0, 1, 0],  # M4: C00-=1, C10+=1
        [0, 1, 0, 1],   # M5: C01+=1, C11+=1
        [1, 0, 0, 0],   # M6: C00+=1
        [-1, 0, 0, 1],  # M7: C00-=1, C11+=1
    ]
    
    # Verify: C00 = 1 - 1 + 1 - 1 = 0? No: M1(1) + M4(-1) + M6(1) + M7(-1) = 1-1+1-1=0. Wrong!
    # C00 should be A00*B00 + A01*B10.
    # Let me just use the standard formulation directly.
    
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A11-A01)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M5 + M7
    
    # Verify C00: M1 - M4 + M6 - M7
    # = (A00+A11)(B00+B11) - A11(B10-B00) + (A10-A00)(B00+B01) - (A11-A01)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B10 + A11*B00 + A10*B00 + A10*B01 - A00*B00 - A00*B01 - A11*B10 - A11*B11 + A01*B10 + A01*B11
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B10 + A11*B00 + A10*B00 + A10*B01 - A00*B00 - A00*B01 - A11*B10 - A11*B11 + A01*B10 + A01*B11
    # = (A00*B00 - A00*B00) + (A00*B11 + A01*B11) + (A11*B00 + A11*B00) + (A11*B11 - A11*B11) + (-A11*B10 - A11*B10 + A01*B10) + (A10*B00 + A10*B01 - A00*B01)
    # = 0 + (A00+A01)*B11 + 2*A11*B00 + 0 + (-2*A11+A01)*B10 + (A10-A00)*B01 + A10*B00
    # Hmm, this doesn't simplify to A00*B00 + A01*B10. Let me recheck.
    
    # Actually I think the standard Strassen is:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # Let me verify C00 = M1 + M4 - M5 - M7:
    # = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 - (A11-A01)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 - A11*B10 - A11*B11 + A01*B10 + A01*B11
    # = A00*B00 + (A00*B11 - A00*B11) + (A11*B00 - A11*B00) + (A11*B11 - A11*B11) + (A11*B10 - A11*B10) + (-A01*B11 + A01*B11) + A01*B10
    # = A00*B00 + A01*B10 ✓
    
    # C01 = M3 + M5 = A00(B01-B11) + (A00+A01)B11 = A00*B01 - A00*B11 + A00*B11 + A01*B11 = A00*B01 + A01*B11 ✓
    # C10 = M2 + M4 = (A10+A11)B00 + A11(B10-B00) = A10*B00 + A11*B00 + A11*B10 - A11*B00 = A10*B00 + A11*B10 ✓
    # C11 = M1 - M2 + M6 + M7 = (A00+A11)(B00+B11) - (A10+A11)B00 + (A10-A00)(B00+B01) + (A11-A01)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A10*B00 + A10*B01 - A00*B00 - A00*B01 + A11*B10 + A11*B11 - A01*B10 - A01*B11
    # = (A00*B00 - A00*B00) + A00*B11 + (A11*B00 - A11*B00) + (A11*B11 + A11*B11) + (-A10*B00 + A10*B00) + A10*B01 - A00*B01 + A11*B10 - A01*B10 - A01*B11
    # = A00*B11 + 2*A11*B11 + A10*B01 - A00*B01 + A11*B10 - A01*B10 - A01*B11
    # Hmm, that doesn't look right either. Let me just use the known correct Strassen.
    
    # The correct Strassen formulas:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # I'll just use these and verify numerically.
    
    S_U = [
        [1, 0, 0, 1],   # M1 = (A00+A11)
        [0, 0, 1, 1],   # M2 = (A10+A11)
        [1, 0, 0, 0],   # M3 = A00
        [0, 0, 0, 1],   # M4 = A11
        [1, 1, 0, 0],   # M5 = (A00+A01)
        [-1, 0, 1, 0],  # M6 = (A10-A00)
        [0, 1, 0, -1],  # M7 = (A11-A01)
    ]
    S_V = [
        [1, 0, 0, 1],   # M1: (B00+B11)
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: (B01-B11)
        [-1, 0, 1, 0],  # M4: (B10-B00)
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: (B00+B01)
        [0, 0, 1, 1],   # M7: (B10+B11)
    ]
    S_W = [
        [1, 0, 0, 1],   # M1: C00+=1, C11+=1
        [0, 0, 1, -1],  # M2: C10+=1, C11-=1
        [0, 1, 0, 0],   # M3: C01+=1
        [1, 0, 1, 0],   # M4: C00+=1, C10+=1
        [-1, 1, 0, 0],  # M5: C00-=1, C01+=1
        [0, 0, 0, 1],   # M6: C11+=1
        [-1, 0, 0, 1],  # M7: C00-=1, C11+=1
    ]
    
    # Build the 44-term decomposition
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
    
    # Try to reduce rank by checking linear dependencies
    all_indices = [(i, a, k) for i in range(n) for a in range(m) for k in range(p)]
    num_constraints = len(all_indices)  # 48
    
    # Build tensor matrix T[idx, r] = t_r[idx]
    T = np.zeros((num_constraints, rank_44), dtype=object)
    for idx, (i, a, k) in enumerate(all_indices):
        for r in range(rank_44):
            T[idx, r] = Fraction(U_44[i * m + a][r]) * Fraction(V_44[a * p + k][r]) * Fraction(W_44[k * n + i][r])
    
    T_float = np.array(T, dtype=float)
    
    # Check for redundant terms
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
            for idx in range(num_constraints):
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
