import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Base cycle edges
    base_edges = [(i, (i + 1) % N) for i in range(N)]

    # Chord (0,s) for s in 2..39 creates cycles:
    #   length s+1: vertices [0,1,...,s]
    #   length N-s+1: vertices [0, N-1, N-2, ..., s]
    # Pair of chords (0,a), (0,b) with a<b creates:
    #   length b-a+2: vertices [0, b, b-1, ..., a]

    required = set(range(3, N + 1))  # lengths 3..39 (40 is base cycle)

    def lengths_from_chords(S):
        """Return set of lengths covered by chord set S (excluding N)."""
        cov = set()
        S_sorted = sorted(S)
        for s in S_sorted:
            l1 = s + 1
            if 3 <= l1 < N:
                cov.add(l1)
            l2 = N - s + 1
            if 3 <= l2 < N:
                cov.add(l2)
        for i in range(len(S_sorted)):
            for j in range(i + 1, len(S_sorted)):
                a = S_sorted[i]
                b = S_sorted[j]
                l = b - a + 2
                if 3 <= l < N:
                    cov.add(l)
        return cov

    # Try increasing chord counts from theoretical minimum
    # With k chords: max lengths = 2k + k*(k-1)//2 = k*(k+3)//2
    # Need 37 lengths (3..39), so k*(k+3)//2 >= 37 => k >= 8
    # But constraints may require more. Start from 8 and go up.

    candidates = list(range(2, N))  # 2..39

    # Precompute for each candidate s: which lengths it covers alone
    single_lengths = {}
    for s in candidates:
        ls = set()
        l1 = s + 1
        if 3 <= l1 < N:
            ls.add(l1)
        l2 = N - s + 1
        if 3 <= l2 < N:
            ls.add(l2)
        single_lengths[s] = ls

    # Greedy with pruning: find minimal S
    best_S = None
    for max_k in range(8, 20):  # try 8, 9, ..., 19 chords
        if time.time() > deadline - 10.0:
            break

        # Use greedy to find a set of size <= max_k
        S = set()
        covered = set()
        remaining = required.copy()

        for _ in range(max_k):
            if not remaining:
                break
            if time.time() > deadline - 5.0:
                break

            best_s = None
            best_gain = -1
            for s in candidates:
                if s in S:
                    continue
                # Compute what this chord adds
                new_lens = set(single_lengths[s])
                # Also add pair lengths with existing chords
                for a in S:
                    if a < s:
                        l = s - a + 2
                        if 3 <= l < N:
                            new_lens.add(l)
                    else:
                        l = a - s + 2
                        if 3 <= l < N:
                            new_lens.add(l)
                gain = len(new_lens & remaining)
                if gain > best_gain:
                    best_gain = gain
                    best_s = s

            if best_s is None or best_gain == 0:
                break

            S.add(best_s)
            # Update covered
            new_cov = lengths_from_chords(S)
            covered = new_cov
            remaining -= covered

        if not remaining:
            best_S = S
            break

    if best_S is None:
        # Fallback: use the known 19-chord solution
        best_S = set(range(2, 21))

    # Build edges
    edges = list(base_edges)
    for s in sorted(best_S):
        edges.append((0, s))

    # Remove duplicates
    unique_edges = []
    seen = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            unique_edges.append((u, v))
    edges = unique_edges

    # Build cycles
    S_sorted = sorted(best_S)
    cycles_by_len = {N: list(range(N))}

    for s in S_sorted:
        l1 = s + 1
        if 3 <= l1 < N and l1 not in cycles_by_len:
            cycles_by_len[l1] = list(range(s + 1))
        l2 = N - s + 1
        if 3 <= l2 < N and l2 not in cycles_by_len:
            cycles_by_len[l2] = [0] + list(range(N - 1, s - 1, -1))

    for i in range(len(S_sorted)):
        for j in range(i + 1, len(S_sorted)):
            a = S_sorted[i]
            b = S_sorted[j]
            l = b - a + 2
            if 3 <= l < N and l not in cycles_by_len:
                cycles_by_len[l] = [0] + list(range(b, a - 1, -1))

    # Fill any missing lengths (shouldn't happen if best_S is correct)
    for L in range(3, N + 1):
        if L not in cycles_by_len:
            if L <= 21:
                cycles_by_len[L] = list(range(L))
            elif L < N:
                k = N - L + 1
                cycles_by_len[L] = [0] + list(range(N - 1, k - 1, -1))
            else:
                cycles_by_len[L] = list(range(N))

    final_cycles = [cycles_by_len[L] for L in range(3, N + 1)]

    # Verify
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    for L, cyc in zip(range(3, N + 1), final_cycles):
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            if (min(a, b), max(a, b)) not in edge_set:
                # Fallback to safe 19-chord solution
                fb_S = set(range(2, 21))
                fb_edges = list(base_edges) + [(0, s) for s in sorted(fb_S)]
                fb_cycles = {}
                for L2 in range(3, N + 1):
                    if L2 == N:
                        fb_cycles[L2] = list(range(N))
                    elif L2 <= 21:
                        fb_cycles[L2] = list(range(L2))
                    else:
                        k = N - L2 + 1
                        fb_cycles[L2] = [0] + list(range(N - 1, k - 1, -1))
                return {'edges': fb_edges, 'cycles': [fb_cycles[L] for L in range(3, N + 1)]}

    return {'edges': edges, 'cycles': final_cycles}
