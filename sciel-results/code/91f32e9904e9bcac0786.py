"""Optimized pancyclic graph construction on 40 vertices using cycle + chords."""
N = 40
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 50  # safety margin before 60s limit

    # Structural change: use two hub vertices to reduce total chord count
    # Hub 0: chords (0,2)..(0,11) -> cycles of length 3..12
    # Hub 20: chords (20,22)..(20,31) -> cycles of length 3..12 (shifted)
    # Cross chords: (0,20), (0,25), (0,30), (0,35) -> longer cycles
    # Additional: (10,25), (10,30) -> more coverage

    edges = set()
    # Base cycle
    for i in range(N):
        edges.add((i, (i + 1) % N))

    # Hub 0 chords
    for k in range(2, 12):
        edges.add((0, k))

    # Hub 20 chords
    for k in range(22, 32):
        edges.add((20, k))

    # Cross chords for longer cycles
    edges.add((0, 20))   # length 21
    edges.add((0, 25))   # length 26
    edges.add((0, 30))   # length 31
    edges.add((0, 35))   # length 36
    edges.add((10, 25))  # additional
    edges.add((10, 30))  # additional
    edges.add((15, 35))  # additional

    # Fill in remaining lengths with more chords from 0
    # Need: lengths 13-20, 22-25, 27-30, 32-35, 37-39
    for k in range(12, 20):
        edges.add((0, k))  # lengths 13-20
    for k in range(21, 25):
        edges.add((0, k))  # lengths 22-25
    for k in range(26, 30):
        edges.add((0, k))  # lengths 27-30
    for k in range(31, 35):
        edges.add((0, k))  # lengths 32-35
    edges.add((0, 36))  # length 37
    edges.add((0, 37))  # length 38
    edges.add((0, 38))  # length 39

    edge_list = list(edges)

    # Generate cycles
    adj = [set() for _ in range(N)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)

    def edge_exists(u, v):
        return (min(u, v), max(u, v)) in edges

    cycles = {}
    for L in range(3, N + 1):
        if L == N:
            cycles[L] = list(range(N))
        else:
            # Try chord-based cycle from hub 0
            k = L - 1
            if 2 <= k <= 38:
                # Check if edge (0,k) exists
                if edge_exists(0, k):
                    # Cycle: 0, 1, 2, ..., k, 0
                    cyc = list(range(k + 1))
                    if all(edge_exists(cyc[i], cyc[(i+1) % (k+1)]) for i in range(k+1)):
                        cycles[L] = cyc
                    else:
                        # Try reverse: 0, 39, 38, ..., k, 0
                        k2 = 41 - L
                        if 2 <= k2 <= 38 and edge_exists(0, k2):
                            cyc = [0] + list(range(39, k2 - 1, -1))
                            if all(edge_exists(cyc[i], cyc[(i+1) % L]) for i in range(L)):
                                cycles[L] = cyc
                            else:
                                # Fallback
                                cycles[L] = list(range(L))
                        else:
                            cycles[L] = list(range(L))
                else:
                    k2 = 41 - L
                    if 2 <= k2 <= 38 and edge_exists(0, k2):
                        cyc = [0] + list(range(39, k2 - 1, -1))
                        if all(edge_exists(cyc[i], cyc[(i+1) % L]) for i in range(L)):
                            cycles[L] = cyc
                        else:
                            cycles[L] = list(range(L))
                    else:
                        cycles[L] = list(range(L))

    # Verify
    def verify():
        for L in range(3, N + 1):
            cyc = cycles[L]
            if len(cyc) != L or len(set(cyc)) != L:
                return False
            for i in range(L):
                u, v = cyc[i], cyc[(i + 1) % L]
                if not edge_exists(u, v):
                    return False
        return True

    if not verify():
        # Fall back to known-good
        edge_list = [(i, (i + 1) % N) for i in range(N)]
        for k in range(2, 21):
            edge_list.append((0, k))
        cycles = {}
        for L in range(3, N + 1):
            if L == N:
                cycles[L] = list(range(N))
            else:
                k = L - 1
                if 2 <= k <= 20:
                    cycles[L] = list(range(k + 1))
                else:
                    k2 = 41 - L
                    cycles[L] = [0] + list(range(39, k2 - 1, -1))
        return {'edges': edge_list, 'cycles': [cycles[L] for L in range(3, N + 1)]}

    # Prune unused edges
    edge_usage = set()
    for L in range(3, N + 1):
        cyc = cycles[L]
        for i in range(L):
            u, v = cyc[i], cyc[(i + 1) % L]
            edge_usage.add((min(u, v), max(u, v)))

    for e in list(edges):
        if e not in edge_usage:
            edges.discard(e)
            if e in edge_list:
                edge_list.remove(e)

    return {'edges': edge_list, 'cycles': [cycles[L] for L in range(3, N + 1)]}
# EVOLVE-BLOCK-END
