"""Rewritten program: algebraic Diophantine feasibility with prime-mask pruning.

We must return two lists A, B of 3 distinct integers in [1,128] such that all
DISTINCT values in the sumset A+B are pairwise coprime, maximising |A+B|.

Key structural insight (parity):
  Pairwise-coprime => at most ONE even number in the whole sumset.
  For a 3x3 additive grid a_i+b_j the number of even sums is
      k*m + (3-k)*(3-m)   (k=#even in A, m=#even in B)
  This equals 1 for NO (k,m) in [0,3]^2, so we must have ZERO even sums.
  That forces (A all odd & B all even) or (A all even & B all odd).
  Hence every sum is ODD and factor 2 never appears; we only need to manage
  odd prime factors.

Strategy:
  1. Precompute prime-factor masks for all integers up to 256.
  2. Enumerate A (3 distinct values) and B (3 distinct values) with the parity
     constraint, using prime-mask bit operations to quickly reject candidates
     where any two distinct sums share a common prime factor.
  3. Track the best (max distinct sumset size) found.
  4. Use a time deadline to return the best result before 85s.
"""

import time
from itertools import combinations

# Deadline: 85 seconds (5s safety margin before 90s kill)
_DEADLINE = 85.0


def _build_prime_masks(max_val=256):
    """For each n in [2, max_val], compute a bitmask of its prime factors.
    Use the index of the prime in a sorted list as the bit position."""
    # Sieve to get smallest prime factor
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val ** 0.5) + 1):
        if spf[i] == i:  # prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Collect all primes
    primes = [p for p in range(2, max_val + 1) if spf[p] == p]
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    masks = [0] * (max_val + 1)
    for n in range(2, max_val + 1):
        m = n
        mask = 0
        while m > 1:
            p = spf[m]
            mask |= (1 << prime_to_idx[p])
            while m % p == 0:
                m //= p
        masks[n] = mask
    return masks, primes


def _valid_sumset(sums_set, masks):
    """Check if all distinct sums in sums_set are pairwise coprime.
    sums_set is a set of distinct integers >= 2."""
    # Get the masks for each sum
    items = list(sums_set)
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            if masks[items[i]] & masks[items[j]]:
                return False
    return True


def _count_distinct(sums_list):
    """Return the number of distinct values in a list of 9 sums."""
    return len(set(sums_list))


def construct():
    start_time = time.time()
    masks, primes = _build_prime_masks(256)

    # Valid ranges with parity constraint:
    # Case 1: A all odd, B all even
    # Case 2: A all even, B all odd
    # Both give all-odd sums.

    odd_vals = [x for x in range(1, 129) if x % 2 == 1]   # 1,3,5,...,127
    even_vals = [x for x in range(2, 129) if x % 2 == 0]  # 2,4,6,...,128

    best_count = 0
    best_A = None
    best_B = None

    # Precompute: for each pair of values (one from set1, one from set2),
    # the sum and its mask. We'll use this to speed up the inner loop.

    # For efficiency, we enumerate A first (smaller effective space with pruning),
    # then B. But since both are C(64,3) ~ 41664, let's be smart.

    # Actually, let's enumerate all triples for A and B and use early termination.
    # C(64,3) = 41664 for each set. Total pairs: 41664^2 ~ 1.7e9 -- too many.

    # Better approach: enumerate A, then for each A, enumerate B with pruning.
    # For a fixed A, the 9 sums are a_i + b_j. We can prune B candidates
    # as we build them up.

    # Let's try a different approach: enumerate all 9-sum configurations via
    # the 6 variables, but use the algebraic structure.

    # Actually, the most practical approach given the time budget:
    # Enumerate A (41664 options) and for each, enumerate B (41664 options),
    # but with aggressive pruning using bitmasks.

    # Even better: since we need pairwise coprime sums, and there are at most 9
    # distinct sums, let's just do the double enumeration but with fast rejection.

    # For each A triple, precompute the 3 values. For each B triple, compute
    # 9 sums, get distinct ones, check pairwise coprime via masks.

    # To speed up: for a fixed A, as we iterate B triples, we can check the
    # sums incrementally. But the simplest fast check is:
    #   sums = [a_i + b_j for i in range(3) for j in range(3)]
    #   distinct = set(sums)
    #   if len(distinct) <= best_count: skip
    #   check pairwise coprime on distinct

    # Let's estimate: 41664 * 41664 = 1.7e9 iterations. Each iteration is ~20 ops.
    # That's ~3.4e10 ops, way too slow in Python.

    # We need a smarter approach. Let's use the fact that sums must be pairwise
    # coprime. For a fixed A = (a0, a1, a2), the sums are:
    #   a0+b0, a0+b1, a0+b2, a1+b0, a1+b1, a1+b2, a2+b0, a2+b1, a2+b2
    #
    # Key: if b_j is such that a0+b_j shares a factor with a0+b_k for k!=j,
    # we can reject early.

    # Alternative: iterate over B values one at a time for a fixed A.
    # For fixed A and fixed b0, b1, check if {a_i+b0, a_i+b1} are pairwise coprime
    # (among themselves and across). Then add b2.

    # Let me restructure: for fixed A, iterate b0, b1, b2 with pruning.

    # Precompute for each A: nothing special, just the 3 values.

    # For fixed A=(a0,a1,a2):
    #   For b0 in even_vals (or odd):
    #     sums0 = [a0+b0, a1+b0, a2+b0]
    #     Check these 3 are pairwise coprime (with each other)
    #     For b1 in even_vals, b1 != b0:
    #       sums1 = [a0+b1, a1+b1, a2+b1]
    #       Check all 6 sums are pairwise coprime
    #       For b2 in even_vals, b2 not in {b0,b1}:
    #         sums2 = [a0+b2, a1+b2, a2+b2]
    #         Check all 9 sums (distinct) are pairwise coprime

    # This is 64 * 63 * 62 = 249984 per A, times 41664 A's = ~1e10. Still too slow.

    # We need more pruning. Let's precompute compatibility.

    # For a fixed A, define for each pair (b, b') whether the 6 sums
    # {a_i+b, a_i+b'} are pairwise coprime. This is a 64x64 boolean matrix.
    # Then for each valid pair (b0,b1), check b2 against both.

    # Actually, let's just try the naive approach with NumPy vectorization
    # or accept that we need to be very clever.

    # Given the 85s budget, let's try a randomized + systematic hybrid:
    # 1. Try all A triples (41664) with a fast B search.
    # 2. For each A, use the pairwise compatibility matrix to quickly find valid B.

    # Let me implement the pairwise compatibility approach.

    def search_case(A_vals, B_vals):
        """A_vals: list of 3 distinct values (same parity), B_vals: list of 3 distinct values (opposite parity).
        Returns best (count, A, B) found."""
        nonlocal best_count, best_A, best_B
        a0, a1, a2 = A_vals
        nB = len(B_vals)
        # Precompute sums for each b in B_vals
        # sums_for_b[b_idx] = (a0+b, a1+b, a2+b)
        sums_for_b = [(a0 + b, a1 + b, a2 + b) for b in B_vals]

        # Precompute mask for each sum
        # mask_for_b[b_idx] = (mask of a0+b, mask of a1+b, mask of a2+b)
        mask_for_b = [(masks[s0], masks[s1], masks[s2]) for (s0, s1, s2) in sums_for_b]

        # Precompute: for each pair (i,j) of B indices, whether the 6 sums are pairwise coprime
        # compat[i][j] = True if all 6 sums {a_k+b_i, a_k+b_j} are pairwise coprime
        # Also need: the 3 sums for b_i are pairwise coprime among themselves,
        # and the 3 sums for b_j are pairwise coprime among themselves.

        # First, check which individual b's have internally coprime sums
        valid_b = []
        for idx in range(nB):
            m0, m1, m2 = mask_for_b[idx]
            if (m0 & m1) == 0 and (m0 & m2) == 0 and (m1 & m2) == 0:
                valid_b.append(idx)

        if len(valid_b) < 3:
            return

        # Now for pairs of valid b's, check cross-coprimality
        # compat_pair[i][j] for i,j in valid_b
        vb = valid_b
        nv = len(vb)
        # Build a set of valid pairs
        valid_pairs = []
        for ii in range(nv):
            for jj in range(ii + 1, nv):
                i, j = vb[ii], vb[jj]
                # Check cross: sums from b_i vs sums from b_j
                mi = mask_for_b[i]
                mj = mask_for_b[j]
                ok = True
                for x in range(3):
                    for y in range(3):
                        if mi[x] & mj[y]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    valid_pairs.append((i, j))

        if len(valid_pairs) < 1:
            return

        # For each valid pair, find a third b that is compatible with both
        # Precompute for each b, which other b's it's compatible with
        # Actually, let's just iterate: for each pair (i,j), try each k != i,j
        # and check if k is compatible with both i and j.

        # To speed up, precompute compat_with[i] = set of b indices compatible with i
        compat_with = [set() for _ in range(nB)]
        for (i, j) in valid_pairs:
            compat_with[i].add(j)
            compat_with[j].add(i)

        for (i, j) in valid_pairs:
            # Candidate k must be in compat_with[i] & compat_with[j], and k != i, k != j
            candidates = compat_with[i] & compat_with[j]
            candidates.discard(i)
            candidates.discard(j)
            for k in candidates:
                # Check distinctness: B_vals[i], B_vals[j], B_vals[k] all distinct
                # They are distinct by construction (different indices in B_vals which has distinct values)
                # Compute the 9 sums and count distinct
                s_i = sums_for_b[i]
                s_j = sums_for_b[j]
                s_k = sums_for_b[k]
                all_sums = s_i + s_j + s_k
                distinct = set(all_sums)
                cnt = len(distinct)
                if cnt > best_count:
                    # Verify pairwise coprime on distinct sums (should be true by construction)
                    if _valid_sumset(distinct, masks):
                        best_count = cnt
                        best_A = list(A_vals)
                        best_B = [B_vals[i], B_vals[j], B_vals[k]]

    # Case 1: A odd, B even
    for A_combo in combinations(odd_vals, 3):
        if time.time() - start_time > _DEADLINE:
            break
        search_case(A_combo, even_vals)

    if time.time() - start_time > _DEADLINE:
        if best_A is not None:
            return [best_A, best_B]
        return [[1, 7, 13], [4, 10, 16]]

    # Case 2: A even, B odd
    for A_combo in combinations(even_vals, 3):
        if time.time() - start_time > _DEADLINE:
            break
        search_case(A_combo, odd_vals)

    if best_A is not None:
        return [best_A, best_B]
    return [[1, 7, 13], [4, 10, 16]]
