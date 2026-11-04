import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    # Rank-17 fallback solution
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)

    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)

    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)

    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)

    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)

    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)

    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)

    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)

    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)

    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)

    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)

    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)

    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)

    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)

    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)

    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)

    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    def is_exact(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Verify rank-17 fallback
    if not is_exact(U17, V17, W17, 17):
        # Should not happen, but return zeros as last resort
        return {"U": [[0]*17 for _ in range(n*m)],
                "V": [[0]*17 for _ in range(m*p)],
                "W": [[0]*17 for _ in range(p*n)]}

    # Try to find rank-16 using structured search
    # Strategy: For rank 16, we have 16 terms. The tensor has 18 entries.
    # Each term is a rank-1 tensor. We need sum of 16 rank-1 tensors = all-ones tensor.
    #
    # Key insight: A rank-1 tensor U_r V_r W_r has support on
    # |supp(U_r)| * |supp(V_r)| * |supp(W_r)| entries.
    # For the all-ones tensor, we need the sum to equal 1 at each of the 18 positions.
    #
    # Let's try a different approach: use the fact that we can have terms with
    # multiple non-zero entries in U, V, W.
    #
    # For rank 16, let's try:
    # - 6 terms that are "Strassen-like" (2 nonzeros in U, 2 in V, 2 in W)
    # - 10 terms that are "simple" (1 nonzero each)
    # Total: 6 + 10 = 16
    #
    # But wait, the 6 Strassen terms would need to cover the "complex" part
    # and the 10 simple terms cover the remaining 10 entries.
    # That's the same as rank 17 but with 6 instead of 7 complex terms.
    #
    # Alternative: Can we use terms with 2 nonzeros in one factor?
    # E.g., a term where U has 2 nonzeros, V has 1, W has 1 covers 2 entries.
    # Or U has 1, V has 2, W has 1 covers 2 entries.
    # Or U has 1, V has 1, W has 2 covers 2 entries.
    #
    # For rank 16: we need to cover 18 entries with 16 terms.
    # If some terms cover 2 entries, we can save terms.
    # 18 entries / 16 terms means we need at least 2 terms to cover 2 entries each,
    # or 1 term to cover 3 entries, etc.
    #
    # Let's try: 14 simple terms (cover 14 entries) + 2 double terms (cover 4 entries) = 18
    # Or: 15 simple terms (cover 15 entries) + 1 triple term (cover 3 entries) = 18
    # Or: 12 simple terms (cover 12 entries) + 3 double terms (cover 6 entries) = 18
    #
    # A "double term" with U having 2 nonzeros at (i1,j1) and (i2,j2),
    # V having 1 nonzero at (j,k), W having 1 nonzero at (k,i):
    # This covers (i1,j1,k) and (i2,j2,k) if j1=j2=j and i1,i2 are the W indices.
    # Wait, W[k*n+i] must be nonzero. If W has one nonzero at (k,i), then i is fixed.
    # So U[i1,j] and U[i2,j] with same j, and W[k,i]... but i is fixed by W.
    # So i1=i2=i. That means U has nonzeros at (i,j1) and (i,j2).
    # Then the term covers (i,j1,k) and (i,j2,k).
    #
    # Similarly, V with 2 nonzeros at (j1,k) and (j2,k), U at (i,j), W at (k,i):
    # Covers (i,j1,k) and (i,j2,k). Same j needed for U.
    #
    # W with 2 nonzeros at (k1,i) and (k2,i), U at (i,j), V at (j,k):
    # Covers (i,j,k1) and (i,j,k2). Same k needed for V.
    #
    # So double terms can merge entries that share two indices.
    #
    # Let's enumerate the 18 entries and find pairs that share two indices:
    # Entries are (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    #
    # Pairs sharing (i,j): (i,j,0) and (i,j,1), (i,j,0) and (i,j,2), (i,j,1) and (i,j,2)
    # Pairs sharing (i,k): (i,0,k) and (i,1,k), (i,0,k) and (i,2,k), (i,1,k) and (i,2,k)
    # Pairs sharing (j,k): (0,j,k) and (1,j,k)
    #
    # For a double term covering (i,j1,k) and (i,j2,k):
    # U has nonzeros at (i,j1) and (i,j2), V has nonzero at (j,k) [but j varies!]
    # Wait, V[j,k] - if j1 != j2, V needs nonzeros at (j1,k) and (j2,k).
    # So V also has 2 nonzeros. And W has 1 nonzero at (k,i).
    # This is a "double-double" term: 2 nonzeros in U, 2 in V, 1 in W.
    # It covers (i,j1,k) and (i,j2,k) with values U[i,j1]*V[j1,k]*W[k,i] and U[i,j2]*V[j2,k]*W[k,i].
    # For both to equal 1: U[i,j1]*V[j1,k]*W[k,i] = 1 and U[i,j2]*V[j2,k]*W[k,i] = 1.
    #
    # Similarly for other pairings.
    #
    # Let's try a greedy approach:
    # 1. Start with all 18 entries needing coverage.
    # 2. Find pairs that can be covered by a single term.
    # 3. Use as many double terms as possible, then fill with simple terms.
    #
    # For rank 16, we need to cover 18 entries with 16 terms.
    # Let d = number of double terms, s = number of simple terms.
    # d + s = 16, 2d + s = 18 => d = 2, s = 14.
    # So we need exactly 2 double terms and 14 simple terms.
    #
    # Let's find 2 disjoint pairs of entries that can each be covered by one term.

    # All 18 entries
    all_entries = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    # Find all possible double-term pairings
    # Type A: (i,j1,k) and (i,j2,k) - share i and k, differ in j
    #   U: nonzeros at (i,j1) and (i,j2)
    #   V: nonzeros at (j1,k) and (j2,k)
    #   W: nonzero at (k,i)
    #   Values: U[i,j1]*V[j1,k]*W[k,i] = 1, U[i,j2]*V[j2,k]*W[k,i] = 1
    #   With entries in {-1,0,1}: U[i,j1]=a, U[i,j2]=b, V[j1,k]=c, V[j2,k]=d, W[k,i]=e
    #   a*c*e = 1, b*d*e = 1 => a*c = b*d (since e=±1)
    #   So a=c=b=d=±1 works, or a=c=-1, b=d=-1, etc.

    # Type B: (i1,j,k) and (i2,j,k) - share j and k, differ in i
    #   U: nonzeros at (i1,j) and (i2,j)
    #   V: nonzero at (j,k)
    #   W: nonzeros at (k,i1) and (k,i2)
    #   Values: U[i1,j]*V[j,k]*W[k,i1] = 1, U[i2,j]*V[j,k]*W[k,i2] = 1

    # Type C: (i,j,k1) and (i,j,k2) - share i and j, differ in k
    #   U: nonzero at (i,j)
    #   V: nonzeros at (j,k1) and (j,k2)
    #   W: nonzeros at (k1,i) and (k2,i)
    #   Values: U[i,j]*V[j,k1]*W[k1,i] = 1, U[i,j]*V[j,k2]*W[k2,i] = 1

    def find_double_pairs():
        """Find all valid double-term pairings."""
        pairs = []
        for i in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    if j1 >= j2:
                        continue
                    for k in range(p):
                        # Type A: (i,j1,k) and (i,j2,k)
                        e1 = (i, j1, k)
                        e2 = (i, j2, k)
                        pairs.append(('A', i, j1, j2, k, e1, e2))

        for i1 in range(n):
            for i2 in range(n):
                if i1 >= i2:
                    continue
                for j in range(m):
                    for k in range(p):
                        # Type B: (i1,j,k) and (i2,j,k)
                        e1 = (i1, j, k)
                        e2 = (i2, j, k)
                        pairs.append(('B', i1, i2, j, k, e1, e2))

        for i in range(n):
            for j in range(m):
                for k1 in range(p):
                    for k2 in range(p):
                        if k1 >= k2:
                            continue
                        # Type C: (i,j,k1) and (i,j,k2)
                        e1 = (i, j, k1)
                        e2 = (i, j, k2)
                        pairs.append(('C', i, j, k1, k2, e1, e2))

        return pairs

    all_pairs = find_double_pairs()

    # We need to select 2 disjoint pairs from all_pairs
    # Then the remaining 14 entries get simple terms.
    # We also need to check that the double terms don't interfere with
    # the simple terms (i.e., the double term's other entries are 0).

    # For a double term of Type A covering (i,j1,k) and (i,j2,k):
    # U has nonzeros at (i,j1) and (i,j2). All other U entries for this term are 0.
    # V has nonzeros at (j1,k) and (j2,k). All other V entries are 0.
    # W has nonzero at (k,i). All other W entries are 0.
    #
    # The contribution to entry (i',j',k') from this term is:
    # U[i',j'] * V[j',k'] * W[k',i']
    # This is nonzero only if:
    # (i',j') in {(i,j1),(i,j2)} AND (j',k') in {(j1,k),(j2,k)} AND (k',i') = (k,i)
    # => i'=i, j' in {j1,j2}, k'=k, and (j',k') in {(j1,k),(j2,k)}
    # => i'=i, k'=k, j' in {j1,j2}
    # So it only contributes to (i,j1,k) and (i,j2,k). Good.

    # Similarly for other types.

    # Now, for the simple terms, each covers exactly one entry.
    # A simple term for entry (i,j,k) has:
    # U nonzero at (i,j), V nonzero at (j,k), W nonzero at (k,i).
    # Values: U[i,j]*V[j,k]*W[k,i] = 1.
    # With entries in {-1,0,1}: U[i,j]=a, V[j,k]=b, W[k,i]=c, a*b*c=1.
    # So a=b=c=1 or a=b=c=-1 or a=1,b=-1,c=-1, etc.

    # The issue: if a double term uses U[i,j1] and U[i,j2], and a simple term
    # also needs U[i,j1], they can coexist (different r indices).
    # Each term is independent (different r), so there's no interference.
    #
    # Wait, that's the key insight! Each term r is independent.
    # So we just need:
    # - 2 double terms covering 4 entries
    # - 14 simple terms covering 14 entries
    # - Total 16 terms covering all 18 entries
    #
    # The only constraint is that the 2 double pairs must be disjoint
    # (no entry appears in both).

    # Let's find 2 disjoint pairs
    def find_two_disjoint_pairs(pairs):
        for idx1 in range(len(pairs)):
            for idx2 in range(idx1 + 1, len(pairs)):
                p1 = pairs[idx1]
                p2 = pairs[idx2]
                e1_1, e1_2 = p1[5], p1[6]
                e2_1, e2_2 = p2[5], p2[6]
                if e1_1 != e2_1 and e1_1 != e2_2 and e1_2 != e2_1 and e1_2 != e2_2:
                    return (p1, p2)
        return None

    result = find_two_disjoint_pairs(all_pairs)

    if result is not None:
        p1, p2 = result
        # Build rank-16 solution
        U16 = np.zeros((n * m, 16))
        V16 = np.zeros((m * p, 16))
        W16 = np.zeros((p * n, 16))

        # Term 0: double term for p1
        # Term 1: double term for p2
        # Terms 2-15: simple terms for remaining 14 entries

        def add_double_term_A(term_idx, i, j1, j2, k):
            """Type A: covers (i,j1,k) and (i,j2,k)"""
            U16[i * m + j1, term_idx] = 1
            U16[i * m + j2, term_idx] = 1
            V16[j1 * p + k, term_idx] = 1
            V16[j2 * p + k, term_idx] = 1
            W16[k * n + i, term_idx] = 1

        def add_double_term_B(term_idx, i1, i2, j, k):
            """Type B: covers (i1,j,k) and (i2,j,k)"""
            U16[i1 * m + j, term_idx] = 1
            U16[i2 * m + j, term_idx] = 1
            V16[j * p + k, term_idx] = 1
            W16[k * n + i1, term_idx] = 1
            W16[k * n + i2, term_idx] = 1

        def add_double_term_C(term_idx, i, j, k1, k2):
            """Type C: covers (i,j,k1) and (i,j,k2)"""
            U16[i * m + j, term_idx] = 1
            V16[j * p + k1, term_idx] = 1
            V16[j * p + k2, term_idx] = 1
            W16[k1 * n + i, term_idx] = 1
            W16[k2 * n + i, term_idx] = 1

        def add_simple_term(term_idx, i, j, k):
            """Simple term: covers (i,j,k)"""
            U16[i * m + j, term_idx] = 1
            V16[j * p + k, term_idx] = 1
            W16[k * n + i, term_idx] = 1

        # Add double term 1
        if p1[0] == 'A':
            add_double_term_A(0, p1[1], p1[2], p1[3], p1[4])
        elif p1[0] == 'B':
            add_double_term_B(0, p1[1], p1[2], p1[3], p1[4])
        elif p1[0] == 'C':
            add_double_term_C(0, p1[1], p1[2], p1[3], p1[4])

        # Add double term 2
        if p2[0] == 'A':
            add_double_term_A(1, p2[1], p2[2], p2[3], p2[4])
        elif p2[0] == 'B':
            add_double_term_B(1, p2[1], p2[2], p2[3], p2[4])
        elif p2[0] == 'C':
            add_double_term_C(1, p2[1], p2[2], p2[3], p2[4])

        # Mark covered entries
        covered = set()
        covered.add(p1[5])
        covered.add(p1[6])
        covered.add(p2[5])
        covered.add(p2[6])

        # Add simple terms for remaining entries
        term_idx = 2
        for entry in all_entries:
            if entry not in covered:
                i, j, k = entry
                add_simple_term(term_idx, i, j, k)
                term_idx += 1

        # Verify
        if is_exact(U16, V16, W16, 16):
            return {
                "U": U16.tolist(),
                "V": V16.tolist(),
                "W": W16.tolist()
            }

    # Fallback to rank 17
    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }
