import time

N = 40
DEADLINE = 80.0  # safety margin below 90s wall-clock kill

def construct():
    t0 = time.time()
    # Base Hamiltonian cycle: 0-1-2-...-39-0
    base = [(i, (i + 1) % N) for i in range(N)]
    base_set = set(tuple(sorted(e)) for e in base)

    # Chord chords (u,v) where 1 <= u < v <= N-1, not adjacent on cycle
    chords = [(u, v) for u in range(1, N - 1) for v in range(u + 1, N - 1)
              if (u, v) not in base_set and (v, u) not in base_set]

    # Lengths each chord certifies: {d+1, N-d+1} where d = v-u (cycle distance)
    cert = {}
    for (u, v) in chords:
        d = v - u
        cert[(u, v)] = frozenset({d + 1, N - d + 1})

    # Greedy set cover over required lengths 3..40
    needed = set(range(3, N + 1))
    chosen = []
    for _ in range(len(chords)):
        if not needed:
            break
        if time.time() - t0 > DEADLINE - 10:
            break
        best, best_gain = None, 0
        for c in chords:
            if c in cert and c not in chosen:
                g = len(cert[c] & needed)
                if g > best_gain:
                    best, best_gain = c, g
                    if g == 2:
                        break
        if best is None:
            break
        chosen.append(best)
        needed -= cert[best]

    # Fallback: add chords until all covered (shouldn't trigger)
    if needed:
        for c in chords:
            if c not in chosen:
                chosen.append(c)
                needed -= cert[c]
                if not needed:
                    break

    # Build edge set
    edge_list = list(base) + list(chosen)
    edge_set = set(tuple(sorted(e)) for e in edge_list)

    # Generate cycles: length L uses the chord that certifies L
    cycle_chord = {}
    for c in chosen:
        for L in cert[c]:
            cycle_chord.setdefault(L, c)
    for L in range(3, N + 1):
        if L not in cycle_chord:
            # assign any chord certifying L
            for c in chords:
                if L in cert[c]:
                    cycle_chord[L] = c
                    break

    cycles = []
    for L in range(3, N + 1):
        u, v = cycle_chord[L]
        # cycle: 0..u then chord to v then v..N-1 then back to 0
        cyc = list(range(u + 1)) + list(range(v, N))
        if len(cyc) != L:
            # fallback: use a base-cycle path of length L (only works if L<=N)
            cyc = list(range(L))
        cycles.append(cyc)

    # Verify every cycle's edges exist; fix any broken cycle
    for idx, cyc in enumerate(cycles):
        L = idx + 3
        ok = True
        for j in range(L):
            a, b = cyc[j], cyc[(j + 1) % L]
            if (a, b) not in edge_set and (b, a) not in edge_set:
                ok = False
                break
        if not ok:
            # brute-force find a valid L-cycle within edge_set
            cyc = _find_cycle(edge_set, L, N)
            if cyc is not None:
                cycles[idx] = cyc

    # Verify again; if still broken, add missing edges
    for idx, cyc in enumerate(cycles):
        L = len(cyc)
        for j in range(L):
            a, b = cyc[j], cyc[(j + 1) % L]
            if (a, b) not in edge_set and (b, a) not in edge_set:
                edge_list.append((a, b))
                edge_set.add(tuple(sorted((a, b))))

    # Try to prune redundant edges (keep within time budget)
    if time.time() - t0 < DEADLINE - 20:
        edge_list, edge_set = _prune(edge_list, edge_set, cycles)

    return {'edges': edge_list, 'cycles': cycles}


def _find_cycle(edge_set, L, N):
    """Find a simple cycle of length L using DFS with backtracking (bounded)."""
    if L > N:
        return None
    adj = {i: [] for i in range(N)}
    for (a, b) in edge_set:
        adj[a].append(b)
        adj[b].append(a)
    for a in adj:
        adj[a].sort()
    # start at smallest vertex to reduce symmetry
    start = 0
    if not adj[start]:
        return None
    path = [start]
    visited = {start}
    limit = 2000000
    steps = [0]

    def dfs():
        if steps[0] > limit:
            return None
        steps[0] += 1
        if len(path) == L:
            if start in adj[path[-1]]:
                return list(path)
            return None
        for nb in adj[path[-1]]:
            if nb == start and len(path) == L - 1:
                continue
            if nb not in visited:
                visited.add(nb)
                path.append(nb)
                r = dfs()
                if r is not None:
                    return r
                path.pop()
                visited.discard(nb)
        return None

    return dfs()


def _prune(edge_list, edge_set, cycles):
    """Try removing edges not used by any cycle."""
    used = set()
    for cyc in cycles:
        L = len(cyc)
        for j in range(L):
            a, b = cyc[j], cyc[(j + 1) % L]
            used.add(tuple(sorted((a, b))))
    new_list = [e for e in edge_list if tuple(sorted(e)) in used]
    new_set = set(tuple(sorted(e)) for e in new_list)
    # verify cycles still valid
    for cyc in cycles:
        L = len(cyc)
        for j in range(L):
            a, b = cyc[j], cyc[(j + 1) % L]
            if (a, b) not in new_set and (b, a) not in new_set:
                return edge_list, edge_set
    return new_list, new_set
