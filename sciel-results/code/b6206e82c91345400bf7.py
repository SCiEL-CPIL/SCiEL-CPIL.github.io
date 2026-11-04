import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]
DEADLINE = 130.0  # safety margin before 150s kill

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    def elapsed():
        return time.time() - t0
    
    # The multiplication tensor T has T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # We want to write T = sum_{c=0}^{r-1} u_c * v_c * w_c where u_c is in R^{nm}, v_c in R^{mp}, w_c in R^{pn}
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Reconstruction: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    
    # Start with schoolbook as baseline
    r_school = n * m * p  # 144
    U_sb = [[0] * r_school for _ in range(nm)]
    V_sb = [[0] * r_school for _ in range(mp)]
    W_sb = [[0] * r_school for _ in range(pn)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][idx] = 1
                V_sb[j * p + k][idx] = 1
                W_sb[k * n + i][idx] = 1
                idx += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = r_school
    
    # Strategy: Try to find lower rank using structured decompositions
    # 1. Strassen-based for the 4x4x4 part + schoolbook for 4x4x5
    # 2. Winograd/Frequency domain approaches
    # 3. Random search with local optimization
    
    def validate(U, V, W, r):
        """Validate that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(r):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    def validate_frac(U, V, W, r):
        """Validate using exact fraction arithmetic"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0, 1)
                    for c in range(r):
                        uc = U[i * m + j][c]
                        vc = V[j * p + k][c]
                        wc = W[k * n + i][c]
                        if isinstance(uc, Fraction) or isinstance(vc, Fraction) or isinstance(wc, Fraction):
                            total += Fraction(uc, 1) * Fraction(vc, 1) * Fraction(wc, 1) if not isinstance(uc, Fraction) else uc * vc * wc
                        else:
                            total += Fraction(uc * vc * wc, 1)
                    if total != 1:
                        return False
        return True
    
    # --- Strassen 4x4x4 + schoolbook 4x4x5 ---
    # Strassen 2x2x2 terms
    STRASSEN_TERMS = [
        {'a': [1, 0, 0, 1], 'b': [1, 0, 0, 1]},  # M0 = (a00+a11)(b00+b11)
        {'a': [0, 0, 0, 1], 'b': [-1, 0, 1, 0]},  # M1 = a11(b10-b00)
        {'a': [0, 0, 1, 1], 'b': [0, 1, 0, 0]},   # M2 = (a10+a11)b01
        {'a': [1, 0, 0, 0], 'b': [0, 1, 0, -1]},  # M3 = a00(b01-b11)
        {'a': [1, 1, 0, 0], 'b': [1, 0, 0, 1]},   # M4 = (a00+a01)(b11+b00)
        {'a': [0, 1, 0, -1], 'b': [0, 0, 1, 1]},  # M5 = (a01-a11)(b10+b11)
        {'a': [-1, 0, 1, 0], 'b': [1, 1, 0, 0]},  # M6 = (a10-a00)(b00+b01)
    ]
    C_BLOCKS = [
        [1, 1, -1, 0, 0, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, -1, 0], # C11
    ]
    
    def gen_strassen_4x4x4():
        """Generate Strassen-of-Strassen terms for 4x4*4x4->4x4.
        Returns list of (i, j, k, coeff) for 4x4 matrices."""
        terms = {}
        for outer_s in range(7):
            a_coeff = STRASSEN_TERMS[outer_s]['a']
            b_coeff = STRASSEN_TERMS[outer_s]['b']
            for cbi in range(4):
                outer_sign = C_BLOCKS[cbi][outer_s]
                if outer_sign == 0:
                    continue
                ci_b, cj_b = divmod(cbi, 2)
                for inner_s in range(7):
                    ia_coeff = STRASSEN_TERMS[inner_s]['a']
                    ib_coeff = STRASSEN_TERMS[inner_s]['b']
                    for inner_cbi in range(4):
                        inner_sign = C_BLOCKS[inner_cbi][inner_s]
                        if inner_sign == 0:
                            continue
                        inner_ci, inner_cj = divmod(inner_cbi, 2)
                        c_row = ci_b * 2 + inner_ci
                        c_col = cj_b * 2 + inner_cj
                        for li in range(2):
                            for lj in range(2):
                                ia_c = ia_coeff[li * 2 + lj]
                                if ia_c == 0:
                                    continue
                                for ri in range(2):
                                    for rj in range(2):
                                        ib_c = ib_coeff[ri * 2 + rj]
                                        if ib_c == 0:
                                            continue
                                        for ab in range(4):
                                            ac = a_coeff[ab]
                                            if ac == 0:
                                                continue
                                            abr, abc = divmod(ab, 2)
                                            a_row = 2 * abr + li
                                            a_col = 2 * abc + lj
                                            for bb in range(4):
                                                bc = b_coeff[bb]
                                                if bc == 0:
                                                    continue
                                                bbr, bbc = divmod(bb, 2)
                                                b_row = 2 * bbr + ri
                                                b_col = 2 * bbc + rj
                                                coeff = outer_sign * inner_sign * ac * ia_c * bc * ib_c
                                                if coeff != 0:
                                                    key = (a_row, a_col, b_row, b_col, c_row, c_col)
                                                    terms[key] = terms.get(key, 0) + coeff
        return [(a, b, c, coeff) for (a, b, c, d, e, f), coeff in terms.items() if coeff != 0]
    
    strassen_terms = gen_strassen_4x4x4()
    # Convert to 4x4x9 format
    U_st = [[0] * len(strassen_terms) for _ in range(nm)]
    V_st = [[0] * len(strassen_terms) for _ in range(mp)]
    W_st = [[0] * len(strassen_terms) for _ in range(pn)]
    
    for idx, (a_row, a_col, b_row, b_col, c_row, c_col, coeff) in enumerate(strassen_terms):
        # a: i=a_row, j=a_col
        U_st[a_row * m + a_col][idx] = coeff
        # b: j=b_row, k=b_col (b_col in 0..3)
        V_st[b_row * p + b_col][idx] = 1
        # w: k=c_col, i=c_row
        W_st[c_col * n + c_row][idx] = 1
    
    # Add schoolbook for k=4..8
    school_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                school_terms.append((i, j, k, 1))
    
    total_rank = len(strassen_terms) + len(school_terms)
    U_full = [[0] * total_rank for _ in range(nm)]
    V_full = [[0] * total_rank for _ in range(mp)]
    W_full = [[0] * total_rank for _ in range(pn)]
    
    for idx in range(len(strassen_terms)):
        U_full[idx // 1]  # already set
        # Copy strassen part
        for i in range(nm):
            U_full[i][idx] = U_st[i][idx]
        for i in range(mp):
            V_full[i][idx] = V_st[i][idx]
        for i in range(pn):
            W_full[i][idx] = W_st[i][idx]
    
    for idx, (i, j, k, coeff) in enumerate(school_terms):
        base = len(strassen_terms) + idx
        U_full[i * m + j][base] = coeff
        V_full[j * p + k][base] = 1
        W_full[k * n + i][base] = 1
    
    if validate(U_full, V_full, total_rank):
        if total_rank < best_rank:
            best_rank = total_rank
            best_U, best_V, best_W = U_full, V_full, W_full
    
    # --- Try Winograd/Frequency domain for 4x4x4 part ---
    # Use 3x3 Winograd-style: split 4x4 into overlapping 3x3 blocks? 
    # Actually, let's try a different structural approach:
    # Use the fact that 4x4x4 can be done with rank 49 (Strassen) but maybe we can do better
    # with a different blocking. Try 2x2 Strassen on one dimension and schoolbook on another.
    
    # Alternative: 4x4x4 = (2x2 Strassen) on rows/cols of A and B, but treat columns of B differently
    # Let's try: split B into two 4x2 blocks. For each 4x2 block, use 4x4x2 decomposition.
    # 4x4x2 can be done with rank 4*4*2 = 32 schoolbook, or maybe less.
    
    # Actually, let's try a random search approach to find lower rank
    # The key insight: we can use iterative rank-1 extraction
    
    # For the 4x4x9 tensor, try to find a decomposition with rank < 129
    # using a greedy approach: repeatedly find the best rank-1 approximation and subtract
    
    if elapsed() < DEADLINE - 10:
        # Try random integer/half-integer searches
        random.seed(42)
        
        def random_search(max_rank, iterations=50):
            """Try to find a valid decomposition with given rank using random initialization + local search"""
            for _ in range(iterations):
                if elapsed() > DEADLINE - 5:
                    return None, None, None
                r = max_rank
                U = [[0] * r for _ in range(nm)]
                V = [[0] * r for _ in range(mp)]
                W = [[0] * r for _ in range(pn)]
                
                # Initialize with random values from {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}
                for i in range(nm):
                    for c in range(r):
                        U[i][c] = random.choice([-1, 0, 0, 1])
                for i in range(mp):
                    for c in range(r):
                        V[i][c] = random.choice([-1, 0, 0, 1])
                for i in range(pn):
                    for c in range(r):
                        W[i][c] = random.choice([-1, 0, 0, 1])
                
                # Local search: try to fix errors by modifying entries
                for step in range(200):
                    if elapsed() > DEADLINE - 5:
                        return None, None, None
                    # Check current error
                    errors = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0
                                for c in range(r):
                                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                                if abs(total - 1) > 1e-9:
                                    errors.append((i, j, k, total - 1))
                    
                    if not errors:
                        return U, V, W
                    
                    # Pick a random error and try to fix it
                    i, j, k, err = random.choice(errors)
                    # Try modifying a random entry in U, V, or W
                    mat = random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        row = i * m + j
                        col = random.randint(0, r - 1)
                        old = U[row][col]
                        for delta in [-1, 1, -0.5, 0.5]:
                            new = old + delta
                            if new in [-1, -0.5, 0, 0.5, 1]:
                                U[row][col] = new
                                # Quick check if this helps
                                total = 0
                                for c in range(r):
                                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                                if abs(total - 1) < abs(err):
                                    break
                                else:
                                    U[row][col] = old
                    elif mat == 'V':
                        row = j * p + k
                        col = random.randint(0, r - 1)
                        old = V[row][col]
                        for delta in [-1, 1, -0.5, 0.5]:
                            new = old + delta
                            if new in [-1, -0.5, 0, 0.5, 1]:
                                V[row][col] = new
                                total = 0
                                for c in range(r):
                                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                                if abs(total - 1) < abs(err):
                                    break
                                else:
                                    V[row][col] = old
                    else:
                        row = k * n + i
                        col = random.randint(0, r - 1)
                        old = W[row][col]
                        for delta in [-1, 1, -0.5, 0.5]:
                            new = old + delta
                            if new in [-1, -0.5, 0, 0.5, 1]:
                                W[row][col] = new
                                total = 0
                                for c in range(r):
                                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                                if abs(total - 1) < abs(err):
                                    break
                                else:
                                    W[row][col] = old
            return None, None, None
        
        # Try ranks from 128 down
        for r_try in range(128, max(100, best_rank - 20), -2):
            if elapsed() > DEADLINE - 10:
                break
            U_try, V_try, W_try = random_search(r_try, iterations=10)
            if U_try is not None and validate(U_try, V_try, W_try, r_try):
                if r_try < best_rank:
                    best_rank = r_try
                    best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # --- Try a structured 3x3 Winograd approach for the 4x4x4 core ---
    # Winograd F(2,3) or similar can reduce rank for small matrix multiplication
    # For 3x3x3, Winograd achieves rank 15 vs 27 schoolbook
    # For 4x4x4, we can pad to 6x6x6 and use Winograd, but that might not help
    
    # Let's try: use the known result that 4x4x4 has rank at most 49 (Strassen)
    # but the true rank is between 44 and 49. We'll stick with 49 for now.
    
    # Final validation
    if not validate(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        best_rank = r_school
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {"U": best_U, "V": best_V, "W": best_W}
