"""Matrix multiplication 3x4x4: search for a rank-minimal exact tensor decomposition."""
import time
import random
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
N, M, P = DIMENSIONS
NM, MP, PN = N * M, M * P, P * N
SCHOOLBOOK = NM * MP * PN  # 48
DEADLINE = 150.0
SAFE_MARGIN = 20.0


def _tensor_terms():
    """Yield (u, v, w) indices where the multiplication tensor has a 1."""
    terms = []
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = i * M + j
                v = j * P + k
                w = k * N + i
                terms.append((u, v, w))
    return terms


TERMS = _tensor_terms()


def _verify_exact(U, V, W, rank):
    """Check sum_c U[u,c]*V[v,c]*W[w,c] == 1 for each (u,v,w) term,
    and == 0 for all other triples. Uses integer arithmetic on scaled values."""
    # We work with the raw entries; they may be Fraction or int.
    # Represent each rank-c column as a triple of vectors and check.
    # Precompute for each c: Ucol (NM), Vcol (MP), Wcol (PN).
    # For efficiency, build the tensor T[u,v,w] = sum_c U[u,c]V[v,c]W[w,c]
    # and compare to the target.
    # Target tensor is sparse (48 ones). We only need to check:
    #  (a) each term triple == 1
    #  (b) a sufficient set of non-term triples == 0 to be confident.
    # For a *proof*, we must check all NM*MP*PN triples. That is 12*16*12=2304.
    # Doing it exactly with Fractions for up to ~40 rank columns is feasible.
    from fractions import Fraction as F
    # Convert to Fraction lists
    Uf = [[F(x) for x in row] for row in U]
    Vf = [[F(x) for x in row] for row in V]
    Wf = [[F(x) for x in row] for row in W]
    # Check all triples
    for u in range(NM):
        for v in range(MP):
            for w in range(PN):
                s = F(0)
                for c in range(rank):
                    s += Uf[u][c] * Vf[v][c] * Wf[w][c]
                # Determine expected value
                # expected is 1 iff (u,v,w) is a term, else 0
                i = u // M
                j = u % M
                jj = v // P
                k = v % P
                kk = w // N
                ii = w % N
                if i == ii and j == jj and k == kk:
                    if s != 1:
                        return False
                else:
                    if s != 0:
                        return False
    return True


def _verify_mod(U, V, W, rank, mod):
    """Fast modular check over Z/modZ. Returns True if consistent with target."""
    # Target: 1 at term triples, 0 elsewhere (mod mod)
    for u in range(NM):
        for v in range(MP):
            for w in range(PN):
                s = 0
                for c in range(rank):
                    s = (s + U[u][c] * V[v][c] * W[w][c]) % mod
                i = u // M
                j = u % M
                jj = v // P
                k = v % P
                kk = w // N
                ii = w % N
                expected = 1 if (i == ii and j == jj and k == kk) else 0
                if s % mod != expected % mod:
                    return False
    return True


def _schoolbook():
    rank = SCHOOLBOOK
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for (u, v, w) in TERMS:
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1
        col += 1
    return {"U": U, "V": V, "W": W}


def _search_rank(r, mod, max_iters, deadline, rng):
    """Randomized sparse search over GF(mod) for a rank-r decomposition.
    Returns (U,V,W) as lists of ints (mod-represented) or None."""
    best = None
    for it in range(max_iters):
        if time.time() > deadline:
            break
        U = [[0] * r for _ in range(NM)]
        V = [[0] * r for _ in range(MP)]
        W = [[0] * r for _ in range(PN)]
        # For each rank column, pick a random pattern.
        # Use a biased sparse generator: each entry nonzero with prob p,
        # values in {1, -1} mod mod (i.e., 1 and mod-1).
        p = 0.15
        for c in range(r):
            for u in range(NM):
                if rng.random() < p:
                    U[u][c] = rng.choice([1, mod - 1])
            for v in range(MP):
                if rng.random() < p:
                    V[v][c] = rng.choice([1, mod - 1])
            for w in range(PN):
                if rng.random() < p:
                    W[w][c] = rng.choice([1, mod - 1])
        if _verify_mod(U, V, W, r, mod):
            best = (U, V, W)
            break
    return best


def _search_rank_structured(r, mod, max_iters, deadline, rng):
    """Search with structured columns: each column picks a random
    (i,j,k)-type support that mimics the tensor's 1-fibers, then random signs.
    This biases toward the algebraic structure of matrix multiplication."""
    best = None
    # Precompute candidate support patterns: a column can be a "pure" fiber
    # (u fixed, v and w vary) etc. We'll generate random signed sparse columns
    # but with higher density on structured positions.
    for it in range(max_iters):
        if time.time() > deadline:
            break
        U = [[0] * r for _ in range(NM)]
        V = [[0] * r for _ in range(MP)]
        W = [[0] * r for _ in range(PN)]
        for c in range(r):
            # Choose a random anchor (i,j,k)
            i = rng.randrange(N)
            j = rng.randrange(M)
            k = rng.randrange(P)
            u0 = i * M + j
            v0 = j * P + k
            w0 = k * N + i
            # Put a sign on the anchor
            su = rng.choice([1, mod - 1])
            sv = rng.choice([1, mod - 1])
            sw = rng.choice([1, mod - 1])
            U[u0][c] = su
            V[v0][c] = sv
            W[w0][c] = sw
            # Add a few extra random entries (sparse perturbation)
            for _ in range(rng.randrange(0, 3)):
                u = rng.randrange(NM)
                if U[u][c] == 0:
                    U[u][c] = rng.choice([1, mod - 1])
            for _ in range(rng.randrange(0, 3)):
                v = rng.randrange(MP)
                if V[v][c] == 0:
                    V[v][c] = rng.choice([1, mod - 1])
            for _ in range(rng.randrange(0, 3)):
                w = rng.randrange(PN)
                if W[w][c] == 0:
                    W[w][c] = rng.choice([1, mod - 1])
        if _verify_mod(U, V, W, r, mod):
            best = (U, V, W)
            break
    return best


def _lift_to_ints(U, V, W, r, mod):
    """Map mod-represented {1, mod-1} to {1, -1} ints; 0 stays 0."""
    def conv(x):
        if x == 0:
            return 0
        if x == 1:
            return 1
        if x == mod - 1:
            return -1
        # general case: keep as is (shouldn't happen for our generators)
        return x
    U2 = [[conv(x) for x in row] for row in U]
    V2 = [[conv(x) for x in row] for row in V]
    W2 = [[conv(x) for x in row] for row in W]
    return U2, V2, W2


def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFE_MARGIN
    rng = random.Random(12345)

    # Try decreasing ranks from a sensible lower bound up to schoolbook-1.
    # The trivial lower bound for n x m x p matrix mult rank is at least
    # max(n*m, m*p, p*n) = max(12,16,12)=16, but realistically much higher.
    # We'll try ranks in a focused window below 48.
    candidate_ranks = list(range(40, 48))  # 40..47
    # Also try a couple of more aggressive ranks
    candidate_ranks = [44, 42, 40, 46, 45, 43, 41, 47] + [r for r in range(36, 40)]

    for r in candidate_ranks:
        if time.time() > deadline:
            break
        # Try several moduli for robustness
        for mod in (5, 7, 11):
            if time.time() > deadline:
                break
            iters = 400
            res = _search_rank_structured(r, mod, iters, deadline, rng)
            if res is None:
                res = _search_rank(r, mod, iters, deadline, rng)
            if res is not None:
                U, V, W = res
                U2, V2, W2 = _lift_to_ints(U, V, W, r, mod)
                # Exact rational verification
                if time.time() > deadline - 5:
                    break
                if _verify_exact(U2, V2, W2, r):
                    return {"U": U2, "V": V2, "W": W2}

    # Fallback: schoolbook (always valid, rank 48)
    return _schoolbook()


if __name__ == "__main__":
    res = construct()
    r = len(res["U"][0])
    print("rank:", r)
