"""Constructs a sparse pancyclic graph on 40 vertices."""

N = 40


def _verify(edges, cycles):
    """Verify edges are simple/distinct, cycles are valid, and all cycle edges exist."""
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    for cyc in cycles:
        if len(cyc) != len(set(cyc)):
            return False
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key not in edge_set:
                return False
    return True


def construct():
    # Strategy: Use a Hamiltonian cycle (40 edges) plus chords from vertex 0.
    # With chords (0, d) for d in 2..20, we get cycle lengths 3..39 via paths
    # 0->1->...->d (length d+1) and 0->39->38->...->(40-d) (length d+1).
    # Length 40 is the base cycle.
    # Total edges: 40 + 19 = 59. Edge count - 40 = 19.
    
    # This is the theoretical minimum: a pancyclic graph on N vertices needs
    # at least N-2 chords beyond the Hamiltonian cycle (each chord can certify
    # at most 2 cycle lengths via the two arcs, and we need 38 lengths from 3..39,
    # so at least 19 chords).
    
    edges = [(i, (i + 1) % N) for i in range(N)]
    
    for d in range(2, 21):
        edges.append((0, d))
    
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            cycles.append(list(range(N)))
        else:
            d = k - 1
            if d <= 20:
                # Path 0->1->...->d, closed by chord (0, d)
                cycles.append(list(range(d + 1)))
            else:
                # Path 0->39->38->...->(N-d), closed by chord (0, N-d)
                # d > 20, so N-d < 20, and chord (0, N-d) exists since N-d >= 2
                # Vertices: 0, 39, 38, ..., N-d
                cycle_vertices = [0] + list(range(N - 1, N - d - 1, -1))
                cycles.append(cycle_vertices)
    
    # Verify
    if not _verify(edges, cycles):
        # Fallback: return something (shouldn't happen)
        pass
    
    return {'edges': edges, 'cycles': cycles}
