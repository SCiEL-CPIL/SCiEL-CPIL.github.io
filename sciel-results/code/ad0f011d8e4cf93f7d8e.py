"""Fresh design: structured random search for a low-rank exact decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def _matmul_tensor_coeffs():
    """Return dict mapping (i,j,k) -> 1 for the matmul tensor T[i,j,k] = delta_{i,k?}...
    Actually tensor for C = A*B: T[a,b,c] such that C[a,c] = sum_b A[a,b] B[b,c].
    The tensor T has entries T[a,b,c] = 1 if (a,b,c) corresponds to product A[a,b]*B[b,c].
    We represent the tensor as a 3D array of size (n, m, p) where T[i,j,k]=1 for all i,j,k? No.
    The multiplication tensor is sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} (with W index k*n+i).
    So T is nonzero at (i,j,k) for every i,j,k — but each term e_{i,j}⊗e_{j,k}⊗e_{k,i} is a rank-1.
    The tensor as a multilinear map: <T, u⊗v⊗w> = sum_{i,j,k} u_{i,j} v_{j,k} w_{k,i}.
    So the tensor's "coefficients" in the standard basis e_{i,j}⊗e_{j,k}⊗e_{k,i} are all 1.
    """
    n, m, p = DIMENSIONS
    # tensor entries: T[i][j][k] = 1 for all i in n, j in m, k in p
    # because the term e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} has coefficient 1.
    return n, m, p

def _verify(U, V, W):
    """Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # U shape (n*m, rank), V (m*p, rank), W (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    s += U[ui][c] * V[vj][c] * W[wk][c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W):
    """Vectorized verification using numpy."""
    import numpy as np
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # For each c, we have vectors u_c (n*m), v_c (m*p), w_c (p*n)
    # We need sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] == 1
    # Build for each c: reshape u_c to (n,m), v_c to (m,p), w_c to (p,n)
    # term_c = einsum('ijc,jkc,kic->ijk', Uc, Vc, Wc) where Uc[i,j,c]=U[i*m+j,c]
    Uc = Ua.reshape(n, m, rank)
    Vc = Va.reshape(m, p, rank)
    Wc = Wa.reshape(p, n, rank)
    # sum over c of Uc[i,j,c]*Vc[j,k,c]*Wc[k,i,c]
    # einsum: i j c, j k c, k i c -> i j k
    result = np.einsum('ijc,jkc,kic->ijk', Uc, Vc, Wc)
    # result[i,j,k] should be 1
    return np.allclose(result, 1.0, atol=1e-9)

def _verify_exact(U, V, W):
    """Exact verification using Fractions to avoid floating point issues."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    from fractions import Fraction
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    s += Uf[ui][c] * Vf[vj][c] * Wf[wk][c]
                if s != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    best = _schoolbook()
    best_rank = n * m * p

    # Try to find a lower rank decomposition via random structured search.
    # Each rank-1 term c has factors u_c (length nm), v_c (length mp), w_c (length pn).
    # We restrict entries to {0, ±1/2, ±1} for simplicity (half-integers allowed).
    # Strategy: random greedy — build up terms, at each step pick a random (i,j,k) that is
    # not yet satisfied, and try to add a term that helps. But exact satisfaction of all
    # 75 equations is a strong constraint.
    #
    # Alternative: use the known structure. The tensor is sum_{i,j,k} e_{ij}⊗e_{jk}⊗e_{ki}.
    # A rank-1 term with support on a single (i,j,k) gives exactly that term. So 75 terms
    # with unit entries work. To reduce rank, we need terms that cover multiple (i,j,k)
    # simultaneously, which requires the factors to have multiple nonzero entries.
    #
    # Let's try a randomized local search: start from schoolbook, randomly merge or
    # replace terms, and check if we can reduce rank while maintaining exactness.
    #
    # Simpler: try random decompositions of small rank and verify.
    # For rank r, pick random u_c, v_c, w_c with entries in {0, ±1/2, ±1},
    # and check if sum_c u_c[i*m+j]*v_c[j*p+k]*w_c[k*n+i] == 1 for all i,j,k.
    # This is a system of 75 equations. Random search in this space is very unlikely
    # to hit exact integer solutions for low rank.
    #
    # Better: use the structure. Note that for fixed j, the terms involve u_{i,j}, v_{j,k}, w_{k,i}.
    # We can try to factor each "slice" separately.
    #
    # Given the difficulty, let's try a modest random search for rank < 75 and return
    # the best found, falling back to schoolbook.

    random.seed(42)

    def random_factor(length):
        # entries in {0, 0.5, 1, -0.5, -1}
        vals = [0, 0.5, 1, -0.5, -1]
        return [random.choice(vals) for _ in range(length)]

    # Try a few ranks
    for target_rank in range(40, 75):
        if time.time() > deadline:
            break
        for attempt in range(200):
            if time.time() > deadline:
                break
            U = [random_factor(nm) for _ in range(target_rank)]
            V = [random_factor(mp) for _ in range(target_rank)]
            W = [random_factor(pn) for _ in range(target_rank)]
            # Transpose to row-major: U[i*m+j][c]
            U_rows = [[U[c][i * m + j] for c in range(target_rank)] for i in range(n) for j in range(m)]
            V_rows = [[V[c][j * p + k] for c in range(target_rank)] for j in range(m) for k in range(p)]
            W_rows = [[W[c][k * n + i] for c in range(target_rank)] for k in range(p) for i in range(n)]
            if _verify_fast(U_rows, V_rows, W_rows):
                # Exact check
                if _verify_exact(U_rows, V_rows, W_rows):
                    best = {"U": U_rows, "V": V_rows, "W": W_rows}
                    best_rank = target_rank
                    break
        if best_rank <= target_rank:
            break

    return best
