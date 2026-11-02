import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Residual of a decomposition: T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r]
    def residual(U, V, W, n, m, p):
        rank = len(U[0])
        R = {}
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m; j_a = a % m
                    j_b = b // p; k_b = b % p
                    k_c = c // n; i_c = c % n
                    target = 1 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    d = target - s
                    if d != 0:
                        R[(a, b, c)] = d
        return R

    def verify(U, V, W, n, m, p):
        return len(residual(U, V, W, n, m, p)) == 0

    def is_halfint(x):
        return x.denominator in (1, 2)

    def check_valid(U, V, W, n, m, p):
        if not verify(U, V, W, n, m, p):
            return False
        for M in (U, V, W):
            for row in M:
                for x in row:
                    if not is_halfint(x):
                        return False
        return True

    def newton(U, V, W, n, m, p, max_iter=25):
        """Solve multilinear system U,V,W -> T via Gauss-Seidel on one factor at a time,
        then Newton on the whole (linear in each factor). Exact Fraction arithmetic."""
        rank = len(U[0])
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        for it in range(max_iter):
            # Residual after each single-factor update
            def upd_factor(fix):
                # fix in {'U','V','W'}: solve for that factor holding others fixed
                if fix == 'U':
                    # For each a: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                    # Let M[a,r] = sum_{b,c} V[b,r]*W[c,r]*T[a,b,c]; N[a,r,s]=sum V[b,r]W[c,r]V[b,s]W[c,s]
                    # Solve N U[a] = M[a]
                    for a in range(n * m):
                        # Build linear system for U[a,:]: unknowns x_r
                        # sum_r x_r * (sum_{b,c} T[a,b,c] V[b,r] W[c,r]) = 0? No.
                        # We want: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all b,c.
                        # This is linear in U[a,:]: for each (b,c): sum_r x_r (V[b,r]W[c,r]) = T[a,b,c]
                        # So A[(b,c), r] = V[b,r]W[c,r], rhs[(b,c)] = T[a,b,c]
                        A = {}
                        rhs = {}
                        for b in range(m * p):
                            for c in range(p * n):
                                i_a = a // m; j_a = a % m
                                j_b = b // p; k_b = b % p
                                k_c = c // n; i_c = c % n
                                t = 1 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0
                                row = []
                                for r in range(rank):
                                    row.append(V[b][r] * W[c][r])
                                A[(b, c)] = row
                                rhs[(b, c)] = t
                        U[a] = solve_linear(A, rhs, rank)
                elif fix == 'V':
                    for b in range(m * p):
                        A = {}
                        rhs = {}
                        for a in range(n * m):
                            for c in range(p * n):
                                i_a = a // m; j_a = a % m
                                j_b = b // p; k_b = b % p
                                k_c = c // n; i_c = c % n
                                t = 1 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0
                                row = []
                                for r in range(rank):
                                    row.append(U[a][r] * W[c][r])
                                A[(a, c)] = row
                                rhs[(a, c)] = t
                        V[b] = solve_linear(A, rhs, rank)
                else:
                    for c in range(p * n):
                        A = {}
                        rhs = {}
                        for a in range(n * m):
                            for b in range(m * p):
                                i_a = a // m; j_a = a % m
                                j_b = b // p; k_b = b % p
                                k_c = c // n; i_c = c % n
                                t = 1 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0
                                row = []
                                for r in range(rank):
                                    row.append(U[a][r] * V[b][r])
                                A[(a, b)] = row
                                rhs[(a, b)] = t
                        W[c] = solve_linear(A, rhs, rank)
            upd_factor('U')
            upd_factor('V')
            upd_factor('W')
            if len(residual(U, V, W, n, m, p)) == 0:
                break
        return U, V, W

    def solve_linear(A, rhs, rank):
        """Solve A x = rhs over Fractions. A: dict key->list of length rank, rhs: dict key->frac.
        Uses Gaussian elimination on the stacked system. Returns list length rank (least-squares
        fallback: return zeros if underdetermined/consistent)."""
        # Stack rows
        rows = []
        for key in A:
            rows.append((A[key], rhs[key]))
        # Build augmented matrix
        aug = []
        for row, b in rows:
            aug.append([Fraction(x) for x in row] + [Fraction(b)])
        # Gaussian elimination with partial pivoting
        N = len(aug)
        if N == 0:
            return [Fraction(0)] * rank
        # Determine pivot columns
        r = 0
        pivots = []
        for col in range(rank):
            if r >= N:
                break
            # find pivot
            piv = None
            for i in range(r, N):
                if aug[i][col] != 0:
                    piv = i
                    break
            if piv is None:
                continue
            aug[r], aug[piv] = aug[piv], aug[r]
            pivots.append(col)
            # eliminate
            pv = aug[r][col]
            for j in range(col, rank + 1):
                aug[r][j] = aug[r][j] / pv
            for i in range(N):
                if i == r:
                    continue
                f = aug[i][col]
                if f == 0:
                    continue
                for j in range(col, rank + 1):
                    aug[i][j] -= f * aug[r][j]
            r += 1
        # Back substitution
        x = [Fraction(0)] * rank
        for i in range(r - 1, -1, -1):
            col = pivots[i]
            s = aug[i][rank]
            for j in range(col + 1, rank):
                s -= aug[i][j] * x[j]
            x[col] = s
        return x

    best = schoolbook()
    best_rank = n * m * p

    # Try to reduce rank by merging schoolbook columns and solving.
    # Start from rank 42, merge pairs of columns into one (reducing rank by 1),
    # then Newton-solve the resulting multilinear system.
    def try_rank(target_rank, max_attempts=40):
        nonlocal best, best_rank
        if target_rank >= best_rank:
            return False
        for attempt in range(max_attempts):
            if time.time() > deadline - 6.0:
                return False
            random.seed(1000000 + target_rank * 1000 + attempt)
            # Start from schoolbook and randomly merge (target_rank) column groups.
            # We keep all 42 terms but assign each to one of target_rank groups.
            all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            groups = [[] for _ in range(target_rank)]
            shuffled = all_triples[:]
            random.shuffle(shuffled)
            for idx, triple in enumerate(shuffled):
                groups[idx % target_rank].append(triple)
            # Build initial U,V,W (each group column is sum of its terms)
            U = [[Fraction(0)] * target_rank for _ in range(n * m)]
            V = [[Fraction(0)] * target_rank for _ in range(m * p)]
            W = [[Fraction(0)] * target_rank for _ in range(p * n)]
            for col, triples in enumerate(groups):
                for (i, j, k) in triples:
                    U[i * m + j][col] += 1
                    V[j * p + k][col] += 1
                    W[k * n + i][col] += 1
            # Newton solve
            U, V, W = newton(U, V, W, n, m, p, max_iter=30)
            if check_valid(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                return True
        return False

    # Descend from 41 to 25, stop when a lower rank is found.
    for target_rank in range(41, 24, -1):
        if time.time() > deadline - 8.0:
            break
        if try_rank(target_rank, max_attempts=20):
            break

    # Convert to plain ints/floats? Keep as Fraction (valid entries). Return as-is.
    return best
