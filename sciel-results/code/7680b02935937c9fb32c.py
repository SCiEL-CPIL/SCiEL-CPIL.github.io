"""Matrix multiplication 2x5x5: schoolbook baseline + integer merge search for rank 49."""
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, rank):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    s += U[ui][c] * V[vj][c] * W[wk][c]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W, rank


def construct():
    deadline = time.time() + 130  # safety margin before the 150s kill
    n, m, p = DIMENSIONS

    U, V, W, rank = _schoolbook(n, m, p)
    if not _validate(U, V, W, n, m, p, rank):
        raise Exception("schoolbook invalid")

    def to_int(M):
        return [[int(x) for x in row] for row in M]

    def fallback():
        return {"U": to_int(U), "V": to_int(V), "W": to_int(W)}

    # ---- Attempt rank-49 by merging two schoolbook columns ----
    # For a merge of columns a=(i1,j1,k1) and b=(i2,j2,k2) with a != b,
    # we need integers x,y with x+y = 1 such that for every (i,j,k):
    #   x*u_a*v_a*w_a + y*u_b*v_b*w_b = 1
    # Since x+y=1, the condition is equivalent to (x-1)*u_a*v_a*w_a + (y-1)*u_b*v_b*w_b = 0,
    # i.e. x*u_a*v_a*w_a = y*u_b*v_b*w_b on the support where either is nonzero.
    #
    # With x = 1 - y:
    #   (1-y)*A = y*B   =>   A = y*(A+B)   =>   y = A/(A+B)
    # For y to be integer and x=1-y to be integer, we need A+B | A (and A+B | B).
    #
    # In schoolbook, A = u_a*v_a*w_a is 1 iff (i,j,k)==a else 0;
    # B = u_b*v_b*w_b is 1 iff (i,j,k)==b else 0.
    # So at (i,j,k)=a: A=1,B=0 -> y=1, x=0 (works only if b's pattern vanishes at a, which it does).
    # At (i,j,k)=b: A=0,B=1 -> y=0, x=1.
    # At (i,j,k)=other: A=B=0 -> any y works.
    # So y must be simultaneously 1 and 0 -> impossible UNLESS a and b are the same column.
    #
    # Conclusion: a pure 2-column merge of the schoolbook cannot reduce rank. We must
    # perturb the schoolbook first. We try a simple perturbation: shift one column's
    # W-entry to a different (i',k) pair, creating overlap that a merge can exploit.
    #
    # Given the difficulty, we run a small randomized perturbation+merge search.
    import random
    rng = random.Random(12345)

    found = False
    while not found and time.time() < deadline:
        # Try random pair (a,b) and a random perturbation of W on column a.
        a = rng.randrange(rank)
        b = rng.randrange(rank)
        if a == b:
            continue

        i1 = a // (m * p)
        r1 = a % (m * p)
        j1 = r1 // p
        k1 = r1 % p

        i2 = b // (m * p)
        r2 = b % (m * p)
        j2 = r2 // p
        k2 = r2 % p

        # Perturb: change W on column a from (k1*n+i1) to a different (k1*n+i1') with i1' != i1
        if n < 2:
            continue
        i1p = 1 - i1  # flip i
        # New W row for column a: k1*n + i1p
        new_w_row = k1 * n + i1p
        old_w_row = k1 * n + i1

        # Build candidate
        Uc = [row[:] for row in U]
        Vc = [row[:] for row in V]
        Wc = [row[:] for row in W]
        Wc[old_w_row][a] = Fraction(0)
        Wc[new_w_row][a] = Fraction(1)

        # Now try to find integers x,y with x+y=1 satisfying the merge condition.
        # A(i,j,k) = Uc[i*m+j][a] * Vc[j*p+k][a] * Wc[k*n+i][a]
        # B(i,j,k) = Uc[i*m+j][b] * Vc[j*p+k][b] * Wc[k*n+i][b]
        # We need (1-y)*A + y*B = 1 for all (i,j,k).
        # Equivalently: A + y*(B - A) = 1  =>  y*(B-A) = 1 - A.
        # For each (i,j,k) where B != A: y = (1 - A) / (B - A).
        # Collect constraints on y.
        y_val = None
        feasible = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A = Uc[i * m + j][a] * Vc[j * p + k][a] * Wc[k * n + i][a]
                    B = Uc[i * m + j][b] * Vc[j * p + k][b] * Wc[k * n + i][b]
                    if B == A:
                        # Need A == 1
                        if A != 1:
                            feasible = False
                            break
                    else:
                        num = 1 - A
                        den = B - A
                        if den == 0:
                            feasible = False
                            break
                        cand = num / den
                        if y_val is None:
                            y_val = cand
                        elif y_val != cand:
                            feasible = False
                            break
                if not feasible:
                    break
            if not feasible:
                break

        if not feasible or y_val is None:
            continue
        # y must be integer (and x = 1 - y integer)
        if y_val.denominator != 1:
            continue
        y = int(y_val)
        x = 1 - y

        # Apply merge: new column c_new = x*col_a + y*col_b (in U,V,W), drop columns a and b
        # Build new matrices of rank-1
        new_rank = rank - 1
        U_new = [[Fraction(0)] * new_rank for _ in range(n * m)]
        V_new = [[Fraction(0)] * new_rank for _ in range(m * p)]
        W_new = [[Fraction(0)] * new_rank for _ in range(p * n)]

        # Map old columns to new columns: all except a,b keep their index; a,b -> merged at position a (assume a<b, place at min)
        merged_pos = min(a, b)
        for old_c in range(rank):
            if old_c == a or old_c == b:
                continue
            new_c = old_c if old_c < merged_pos else old_c - 1
            for r in range(n * m):
                U_new[r][new_c] = Uc[r][old_c]
            for r in range(m * p):
                V_new[r][new_c] = Vc[r][old_c]
            for r in range(p * n):
                W_new[r][new_c] = Wc[r][old_c]
        # Merged column
        for r in range(n * m):
            U_new[r][merged_pos] = x * Uc[r][a] + y * Uc[r][b]
        for r in range(m * p):
            V_new[r][merged_pos] = x * Vc[r][a] + y * Vc[r][b]
        for r in range(p * n):
            W_new[r][merged_pos] = x * Wc[r][a] + y * Wc[r][b]

        if _validate(U_new, V_new, W_new, n, m, p, new_rank):
            return {"U": to_int(U_new), "V": to_int(V_new), "W": to_int(W_new)}

    # Time's up or no merge found: return schoolbook
    return fallback()
# EVOLVE-BLOCK-END
