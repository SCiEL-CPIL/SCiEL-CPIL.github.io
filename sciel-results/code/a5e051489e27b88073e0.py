"""
Kissing number construction in dimension 11 using D22 lattice structure.

The key insight from the 1.58 scoring solution is to use the D22 lattice (or a
related high-density lattice) projected down to 11 dimensions, or equivalently,
to construct a set of integer vectors in R^11 that achieves a higher density than
the simple E8 ⊕ A3 construction.

For dimension 11, the D22 lattice structure can be leveraged as follows:
- D22 is defined as vectors in R^22 with integer coordinates whose sum is even.
- We can take the 11-dimensional subspace where coordinates come in pairs, or use
  a checkerboard construction.

Actually, a more direct approach: use the fact that in R^11, we can construct
vectors from the D11 lattice (integer vectors with even coordinate sum) which has
220 minimal vectors of norm sqrt(2). But we need min pairwise distance >= max norm.

If all vectors have norm sqrt(2), then we need min pairwise distance >= sqrt(2).
For vectors v, w with v·v = w·w = 2, |v-w|^2 = 4 - 2v·w.
So |v-w| >= sqrt(2) means 4 - 2v·w >= 2, i.e., v·w <= 1.

The D11 lattice has 2^11 * ... minimal vectors. Let me count:
D_n has 2^n * (n-1) * ... minimal vectors? No.
D_n minimal vectors (norm sqrt(2)): all vectors with two ±1's and rest 0.
Count: C(n,2) * 2^2 = C(n,2)*4 for n=11: 55*4 = 220.

So D11 gives 220 vectors. That's what the seed already has.

To get more, we need to include vectors of larger norm. If we allow vectors of norm
sqrt(3) (three ±1's), then max norm = sqrt(3), and we need min dist >= sqrt(3).
|v-w|^2 = |v|^2 + |w|^2 - 2v·w >= 3.

This becomes more complex. The 1.58 score corresponds to 348 vectors.

Let me think about the Leech lattice or other constructions. Actually, for d=11,
a known good construction uses the D22 lattice folded: take vectors in D22 and
project or restrict.

Alternative: Use the fact that we can mix vectors of different norms as long as
the constraint holds. Let me implement a greedy algorithm that:
1. Starts with D11 minimal vectors (220 vectors of norm sqrt(2)).
2. Tries to add vectors of norm sqrt(3) (three ±1's) if they satisfy the distance constraint.
3. Also tries vectors of norm 2 (one coordinate ±2, or four ±1's, etc.)

With a time deadline guard.
"""

import numpy as np
import itertools
import time

D = 11
TIME_LIMIT = 100  # seconds, leave margin before 150s kill


def _norm_sq(v):
    return int(np.dot(v, v))


def _check_add(v, vectors, max_norm_sq):
    """Check if vector v can be added to the set."""
    vn_sq = _norm_sq(v)
    # New max norm sq is max(max_norm_sq, vn_sq)
    new_max = max(max_norm_sq, vn_sq)
    for w in vectors:
        diff = v - w
        dist_sq = int(np.dot(diff, diff))
        if dist_sq < new_max:
            return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    # Start with D11 minimal vectors: all vectors with exactly two ±1 coordinates
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vectors.append(v)

    vectors = np.array(vectors, dtype=np.int64)
    max_norm_sq = 2  # all have norm^2 = 2

    # Now try to add vectors with norm^2 = 3 (exactly three ±1 coordinates)
    # There are C(11,3) * 2^3 = 165 * 8 = 1320 such vectors
    # Check each against all current vectors

    candidates_3 = []
    for combo in itertools.combinations(range(D), 3):
        for signs in itertools.product((1, -1), repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates_3.append(v)

    # Greedily add candidates of norm^2 = 3
    added_count = 0
    for c in candidates_3:
        if time.time() > deadline:
            break
        c = c.astype(np.int64)
        # Check against all existing vectors
        ok = True
        for w in vectors:
            diff = c - w
            dist_sq = int(np.dot(diff, diff))
            # Need dist_sq >= max(3, 2) = 3
            if dist_sq < 3:
                ok = False
                break
        if ok:
            vectors = np.vstack([vectors, c[None, :]])
            added_count += 1

    # Now try to add vectors with norm^2 = 4
    # Types: one coordinate ±2; or four coordinates ±1
    candidates_4 = []
    # Type 1: one coordinate ±2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates_4.append(v)
    # Type 2: four coordinates ±1
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates_4.append(v)

    # After adding norm^2=3 vectors, max_norm_sq might be 3
    current_max_norm_sq = max(max_norm_sq, 3 if added_count > 0 else 0)

    for c in candidates_4:
        if time.time() > deadline:
            break
        c = c.astype(np.int64)
        vn_sq = int(np.dot(c, c))
        new_max = max(current_max_norm_sq, vn_sq)
        ok = True
        for w in vectors:
            diff = c - w
            dist_sq = int(np.dot(diff, diff))
            if dist_sq < new_max:
                ok = False
                break
        if ok:
            vectors = np.vstack([vectors, c[None, :]])
            current_max_norm_sq = new_max

    # Try norm^2 = 5: five ±1's, or one ±2 and one ±1
    if time.time() < deadline - 10:
        candidates_5 = []
        # Type: one ±2, one ±1
        for i in range(D):
            for j in range(D):
                if i == j:
                    continue
                for s1 in (1, -1):
                    for s2 in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = 2 * s1
                        v[j] = s2
                        candidates_5.append(v)
        # Type: five ±1's
        for combo in itertools.combinations(range(D), 5):
            for signs in itertools.product((1, -1), repeat=5):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(combo, signs):
                    v[idx] = s
                candidates_5.append(v)

        for c in candidates_5:
            if time.time() > deadline:
                break
            c = c.astype(np.int64)
            vn_sq = int(np.dot(c, c))
            new_max = max(current_max_norm_sq, vn_sq)
            ok = True
            for w in vectors:
                diff = c - w
                dist_sq = int(np.dot(diff, diff))
                if dist_sq < new_max:
                    ok = False
                    break
            if ok:
                vectors = np.vstack([vectors, c[None, :]])
                current_max_norm_sq = new_max

    return vectors
