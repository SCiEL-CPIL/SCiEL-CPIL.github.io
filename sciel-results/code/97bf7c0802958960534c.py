"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total = n * m * p  # 60
    deadline = time.time() + 140  # safety margin before 150s kill

    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k.
    # A rank-r decomposition is three matrices U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all (i,j,k).
    #
    # We search for rank-1 decompositions with entries in {0, ±1, ±1/2}.
    # A rank-1 term is (u, v, w) where u is a row vector of length m,
    # v is a column vector of length p, w is a column vector of length n.
    # Actually more precisely:
    #   U[i*m+j, c] = u_{i,j}, V[j*p+k, c] = v_{j,k}, W[k*n+i, c] = w_{k,i}
    # The contribution of term c to (i,j,k) is u_{i,j} * v_{j,k} * w_{k,i}.
    #
    # For a fixed term, define:
    #   a_{i,j} = u_{i,j}
    #   b_{j,k} = v_{j,k}
    #   c_{k,i} = w_{k,i}
    # The product at (i,j,k) is a_{i,j} * b_{j,k} * c_{k,i}.
    #
    # We need to find r terms such that for every (i,j,k):
    #   sum_{c=1}^{r} a^{(c)}_{i,j} * b^{(c)}_{j,k} * c^{(c)}_{k,i} = 1
    #
    # Strategy: use randomized search over small alphabets.
    # Each term is defined by (a, b, c) where a is n x m, b is m x p, c is p x n,
    # with entries from a small set.

    # For a term (a, b, c), define its "signature" as the vector of values at all (i,j,k).
    # sig_{i,j,k} = a_{i,j} * b_{j,k} * c_{k,i}
    #
    # We want to find r signatures that sum to the all-ones vector.
    #
    # Key insight: we can fix the structure. For n=2, the c matrix is 6x2.
    # Let's enumerate possible patterns.

    # Actually, let's think about this differently. The tensor is T[i,j,k]=1.
    # We can write T as a sum of rank-1 tensors.
    # A rank-1 tensor is alpha * (u ⊗ v ⊗ w) where u in R^{n*m}, v in R^{m*p}, w in R^{p*n}.
    # But due to the structure, the rank-1 terms have special form.

    # Let's use the fact that for n=2, we can try to decompose using the following:
    # For each pair (j,k), the "middle" index, we can have terms that contribute
    # to C[i,k] via A[i,j]*B[j,k].

    # Alternative approach: use known bounds and try to construct explicitly.
    # R(2,5,6) <= R(2,5,2)*3 + R(2,5,3)*2 ... this gets complex.

    # Let's try a direct randomized search for a decomposition with rank < 60.
    # We'll search for terms with entries in {0, 1, -1} and {0, 1/2, -1/2}.

    # For efficiency, precompute the "contribution" of a term to each (i,j,k).
    # A term is (a, b, c) where:
    #   a: list of n*m entries (row-major: a[i*m+j])
    #   b: list of m*p entries (row-major: b[j*p+k])
    #   c: list of p*n entries (row-major: c[k*n+i])
    # Contribution at (i,j,k) = a[i*m+j] * b[j*p+k] * c[k*n+i]

    # We need sum over terms of contributions = 1 at every (i,j,k).

    # Let's define the set of (i,j,k) indices.
    idx = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    total_idx = len(idx)  # 60

    def term_contribution(a, b, c):
        """Compute the contribution vector of a single rank-1 term."""
        contrib = [0] * total_idx
        for t, (i, j, k) in enumerate(idx):
            contrib[t] = a[i*m + j] * b[j*p + k] * c[k*n + i]
        return contrib

    # Try to find a set of terms whose contributions sum to all-ones.
    # Use randomized local search.

    def random_term(alphabet):
        """Generate a random term with entries from alphabet."""
        a = [random.choice(alphabet) for _ in range(n * m)]
        b = [random.choice(alphabet) for _ in range(m * p)]
        c = [random.choice(alphabet) for _ in range(p * n)]
        return a, b, c

    def try_decomposition(rank, max_iters, alphabet):
        """Try to find a rank-r decomposition using randomized search."""
        target = [1] * total_idx
        # Start with random terms
        terms = [random_term(alphabet) for _ in range(rank)]
        current_sum = [0] * total_idx
        for term in terms:
            contrib = term_contribution(*term)
            for t in range(total_idx):
                current_sum[t] += contrib[t]

        best_err = sum(abs(current_sum[t] - target[t]) for t in range(total_idx))

        for it in range(max_iters):
            if best_err == 0:
                return terms
            # Pick a random term to modify
            idx_term = random.randint(0, rank - 1)
            # Compute current sum without this term
            old_contrib = term_contribution(*terms[idx_term])
            new_sum = [current_sum[t] - old_contrib[t] for t in range(total_idx)]

            # Try a few random replacements
            for _ in range(20):
                new_term = random_term(alphabet)
                new_contrib = term_contribution(*new_term)
                trial_sum = [new_sum[t] + new_contrib[t] for t in range(total_idx)]
                err = sum(abs(trial_sum[t] - target[t]) for t in range(total_idx))
                if err < best_err:
                    best_err = err
                    terms[idx_term] = new_term
                    current_sum = trial_sum
                    break

        if best_err == 0:
            return terms
        return None

    # First, try to find a decomposition with rank less than 60.
    # Use alphabets with small entries.

    best_rank = n * m * p  # 60 (schoolbook)
    best_terms = None

    # Schoolbook terms
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = [0] * (n * m)
                b = [0] * (m * p)
                c = [0] * (p * n)
                a[i * m + j] = 1
                b[j * p + k] = 1
                c[k * n + i] = 1
                schoolbook_terms.append((a, b, c))
    best_terms = schoolbook_terms
    best_rank = len(schoolbook_terms)

    # Try smaller ranks
    for rank in range(50, best_rank):
        if time.time() > deadline:
            break
        # Use alphabet {0, 1, -1}
        result = try_decomposition(rank, 5000, [0, 1, -1])
        if result is not None:
            best_rank = rank
            best_terms = result
            break

    # Try with half-integers
    if best_rank > 50:
        for rank in range(50, best_rank):
            if time.time() > deadline:
                break
            result = try_decomposition(rank, 3000, [0, 1, -1, Fraction(1,2), Fraction(-1,2)])
            if result is not None:
                best_rank = rank
                best_terms = result
                break

    # Build U, V, W from the best terms
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]

    for c, (a, b, w) in enumerate(best_terms):
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = a[i * m + j]
        for j in range(m):
            for k in range(p):
                V[j * p + k][c] = b[j * p + k]
        for k in range(p):
            for i in range(n):
                W[k * n + i][c] = w[k * n + i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
