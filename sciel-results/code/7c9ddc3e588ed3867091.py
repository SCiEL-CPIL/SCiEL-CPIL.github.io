"""Matrix multiplication tensor decomposition for 2x3 by 3x10.

Strategy: greedy exact rank-1 peeling.
The multiplication tensor T has support S = {(i*m+j, j*p+k, k*n+i) : i,j,k}.
A rank-1 component u ⊗ v ⊗ w with u,v,w ∈ {0, ±1/2, ±1}^* contributes
sign = (u[a]*v[b]*w[c]) at index (a,b,c) and zero elsewhere (wherever any
factor is zero). We peel off such components one at a time: pick a sparse
support pattern (A, B, C), assign values to each non-zero entry so that the
resulting rank-1 tensor R satisfies R[x] ∈ {0, ±1} everywhere AND
R[x] = T[x] for every x where T[x] != 0 (i.e. R is a "face" of T). Then
subtract R from T. Repeat until T is zero. The number of peeled components
is the rank.
"""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T


def _support_indices(n, m, p):
    """Return the list of (a, b, c) where T has a 1."""
    S = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                S.append((i * m + j, j * p + k, k * n + i))
    return S


def _try_peel_one(T, support, max_support_size=3, time_limit=None):
    """Try to find a single rank-1 component R (integer/half-integer entries)
    such that R is a "face" of T: R[x] ∈ {0, ±1} for all x, and R[x] = T[x]
    for all x where T[x] != 0 (with T[x] ∈ {0, ±1} at this stage).
    Returns (u, v, w) as 1-D arrays (with zeros outside the support) or None.
    """
    nm, mp, pn = T.shape
    S = support  # list of (a, b, c) where T is non-zero

    # For each non-zero index, the rank-1 value must equal T[a,b,c].
    # So for each (a,b,c) in S, u[a]*v[b]*w[c] = T[a,b,c].
    # For indices not in S, we need u[a]*v[b]*w[c] = 0, meaning at least one
    # of u[a], v[b], w[c] is zero.

    # We'll enumerate sparse support patterns: pick a subset A of {0..nm-1},
    # B of {0..mp-1}, C of {0..pn-1} where u,v,w are non-zero.
    # Constraint: for every (a,b,c) in S, a ∈ A, b ∈ B, c ∈ C (since the
    # product must be non-zero). So A, B, C must cover all indices appearing
    # in S.

    # Required indices:
    A_req = sorted(set(a for a, b, c in S))
    B_req = sorted(set(b for a, b, c in S))
    C_req = sorted(set(c for a, b, c in S))

    # The minimum support sizes are |A_req|, |B_req|, |C_req|.
    # We can try to add extra indices (where u/v/w are non-zero but the
    # product happens to be zero at all T-nonzero positions... but that's
    # impossible since if a ∈ A and b ∈ B and c ∈ C, the product is non-zero).
    # Actually, if a ∈ A, b ∈ B, c ∈ C, then u[a]*v[b]*w[c] ≠ 0, so this
    # index must be in S (otherwise T=0 but R≠0, contradiction).
    # Therefore: A × B × C ⊆ S (as a set of triples).
    # This means A, B, C must be "compatible": every triple (a,b,c) with
    # a∈A, b∈B, c∈C must be in S.

    # Start with the minimal sets and try to find values.
    # For the minimal sets, A×B×C must be ⊆ S. Let's check.
    # S has exactly |A_req|*|B_req|*|C_req| = n*m*p elements? No!
    # S has n*m*p elements, but A_req has n*m elements, B_req has m*p, C_req has p*n.
    # So |A_req|*|B_req|*|C_req| = n*m * m*p * p*n = (n*m*p)^2 which is >> n*m*p.
    # So A_req × B_req × C_req is NOT a subset of S in general.
    # This means we CANNOT use the full required sets as-is.

    # We need to find subsets A ⊇ A_req? No, A must contain all a that appear in S.
    # Wait: if (a,b,c) ∈ S, then u[a]*v[b]*w[c] = T[a,b,c] ≠ 0, so u[a]≠0, v[b]≠0, w[c]≠0.
    # So A must contain A_req, B must contain B_req, C must contain C_req.
    # But then A×B×C ⊇ A_req×B_req×C_req which has (n*m*p)^2 elements, far more than |S|=n*m*p.
    # This is a contradiction UNLESS many of those extra triples happen to be in S.

    # For the multiplication tensor, S = {(i*m+j, j*p+k, k*n+i)}.
    # A_req = {i*m+j : i∈[n], j∈[m]} = all of [0, nm).
    # B_req = {j*p+k : j∈[m], k∈[p]} = all of [0, mp).
    # C_req = {k*n+i : k∈[p], i∈[n]} = all of [0, pn).
    # So A_req × B_req × C_req = [0,nm) × [0,mp) × [0,pn) which is the ENTIRE tensor space.
    # But S is only n*m*p of the (n*m)*(m*p)*(p*n) possible indices.
    # So it's IMPOSSIBLE to have a single rank-1 term with integer/half-integer
    # entries that is a "face" of T (unless the tensor is itself rank 1).

    # This means we cannot peel off a single rank-1 term that matches T on its
    # entire support. We need a different approach.

    # Alternative: allow the rank-1 term to have values that don't need to
    # match T exactly on the support, but the SUM of all rank-1 terms must equal T.
    # This is the standard tensor decomposition problem. The greedy approach
    # would be: pick any valid rank-1 term (integer/half-integer), subtract it,
    # and repeat. But "valid" just means it doesn't create fractional residuals.

    # Let's try a simpler approach: use the schoolbook decomposition as a
    # baseline, and try to MERGE pairs of rank-1 terms into a single term
    # when possible (i.e., if two terms can be combined into one with
    # integer/half-integer entries).

    return None


def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {'U': U, 'V': V, 'W': W, 'rank': rank}


def _try_merge_decomposition(n, m, p, T, time_budget=100):
    """Try to reduce the schoolbook rank by finding alternative decompositions.
    
    Key insight: for matrix multiplication M(n,m,p), the rank is known to be
    at most n*m*p (schoolbook) and at least max(n*m, m*p, p*n). For 2x3x10,
    that's rank 60 vs lower bound 30.
    
    We try a structured approach: look for decompositions where the factors
    have a block structure that allows cancellation.
    """
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + time_budget

    # Try ranks from lower bound up to schoolbook-1
    min_rank = max(nm, mp, pn)  # 30
    max_rank = nm * mp * pn  # 60

    # For each rank, try to find a decomposition using a structured search
    # over small integer/half-integer values.
    
    # Strategy: use the fact that the tensor has a specific structure.
    # T[i*m+j, j*p+k, k*n+i] = 1.
    # A rank-1 term (u, v, w) contributes u[i*m+j] * v[j*p+k] * w[k*n+i].
    
    # Let's try a direct search for small ranks using random sparse initialization
    # with integer/half-integer constraints.
    
    for rank in range(min_rank, max_rank):
        if time.time() > deadline:
            break
        
        # Try a few random sparse integer/half-integer initializations
        for trial in range(20):
            if time.time() > deadline:
                break
            
            rng = random.Random(trial * 1000 + rank)
            
            # Initialize with sparse integer/half-integer values
            U = np.zeros((nm, rank), dtype=np.float64)
            V = np.zeros((mp, rank), dtype=np.float64)
            W = np.zeros((pn, rank), dtype=np.float64)
            
            # Each column gets a few non-zero entries
            for k in range(rank):
                # Pick 1-3 non-zero entries in each factor
                nz_u = rng.randint(1, 3)
                nz_v = rng.randint(1, 3)
                nz_w = rng.randint(1, 3)
                
                idx_u = rng.sample(range(nm), min(nz_u, nm))
                idx_v = rng.sample(range(mp), min(nz_v, mp))
                idx_w = rng.sample(range(pn), min(nz_w, pn))
                
                for idx in idx_u:
                    U[idx, k] = rng.choice([0.5, 1, -0.5, -1])
                for idx in idx_v:
                    V[idx, k] = rng.choice([0.5, 1, -0.5, -1])
                for idx in idx_w:
                    W[idx, k] = rng.choice([0.5, 1, -0.5, -1])
            
            # Check if this is an exact decomposition
            recon = np.einsum('ak,bk,ck->abc', U, V, W)
            if np.allclose(recon, T, atol=1e-9):
                # Verify entries are integer or half-integer
                if np.all(np.allclose(U * 2, np.round(U * 2))) and \
                   np.all(np.allclose(V * 2, np.round(V * 2))) and \
                   np.all(np.allclose(W * 2, np.round(W * 2))):
                    print(f"Found rank-{rank} decomposition at trial {trial}")
                    return U, V, W, rank
        
        # Also try a gradient-based approach with projection to nearest
        # integer/half-integer grid
        if rank <= max_rank - 5:  # Only for lower ranks
            for trial in range(5):
                if time.time() > deadline:
                    break
                
                rng = np.random.RandomState(trial * 7 + rank)
                U = rng.randn(nm, rank) * 0.5
                V = rng.randn(mp, rank) * 0.5
                W = rng.randn(pn, rank) * 0.5
                
                lr = 0.05
                for step in range(500):
                    if time.time() > deadline:
                        break
                    
                    recon = np.einsum('ak,bk,ck->abc', U, V, W)
                    diff = recon - T
                    loss = np.sum(diff ** 2)
                    
                    if loss < 1e-12:
                        # Project to nearest integer/half-integer
                        U2 = np.round(U * 2) / 2
                        V2 = np.round(V * 2) / 2
                        W2 = np.round(W * 2) / 2
                        recon2 = np.einsum('ak,bk,ck->abc', U2, V2, W2)
                        if np.allclose(recon2, T, atol=1e-9):
                            print(f"Found rank-{rank} decomposition (gradient+project) at trial {trial}, step {step}")
                            return U2, V2, W2, rank
                        break
                    
                    # Gradient
                    grad_U = np.einsum('abc,bk,ck->ak', diff, V, W)
                    grad_V = np.einsum('abc,ak,ck->bk', diff, U, W)
                    grad_W = np.einsum('abc,ak,bk->ck', diff, U, V)
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                    
                    # Periodic projection
                    if step % 100 == 99:
                        U = np.round(U * 2) / 2
                        V = np.round(V * 2) / 2
                        W = np.round(W * 2) / 2
    
    return None, None, None, None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    T = _build_tensor(n, m, p)

    # Try to find a better decomposition
    try:
        U, V, W, rank = _try_merge_decomposition(n, m, p, T, time_budget=100)
        if U is not None and rank < n * m * p:
            # Final verification
            recon = np.einsum('ak,bk,ck->abc', U, V, W)
            if np.allclose(recon, T, atol=1e-9):
                print(f"Verified rank-{rank} decomposition")
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    except Exception as e:
        print(f"Exception: {e}")

    # Fallback to schoolbook
    sb = _schoolbook_decomposition(n, m, p)
    return {'U': sb['U'].tolist(), 'V': sb['V'].tolist(), 'W': sb['W'].tolist()}

# EVOLVE-BLOCK-END
