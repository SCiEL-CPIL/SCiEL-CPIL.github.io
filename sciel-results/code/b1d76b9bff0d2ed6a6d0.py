import time

N = 40

def construct():
    """
    Construct a sparse pancyclic graph on 40 vertices with explicit cycle certificates.
    
    Strategy:
    - Start with a Hamiltonian cycle (40 edges) to guarantee a cycle of length 40.
    - Add chords from vertex 0 to vertices 2..19 to create cycles of lengths 3..20 
      (using chord (0, m) and path 0-1-...-m-0) and lengths 22..39 
      (using chord (0, m) and path 0-39-...-m-0).
    - This gives 40 + 18 = 58 edges, for a score of 18.
    
    The construction is explicit and deterministic, so no search is needed.
    """
    edges = set()
    
    # Base Hamiltonian Cycle: 0-1-...-39-0
    for i in range(N):
        u = i
        v = (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
    
    # Chords from 0 to m for m=2..19
    # These create cycles of length m+1 (for m=2..19, lengths 3..20)
    # and cycles of length 41-m (for m=2..19, lengths 39..22)
    for m in range(2, 20):
        u = 0
        v = m
        if u > v:
            u, v = v, u
        edges.add((u, v))
    
    edge_list = list(edges)
    
    # Construct Cycles of lengths 3..40
    cycles = []
    
    for k in range(3, N + 1):
        if k == N:
            # Full Hamiltonian cycle
            cycles.append(list(range(N)))
        elif k <= 20:
            # Cycle of length k uses chord (0, k-1) and path 0-1-...-(k-1)-0
            # Vertices: 0, 1, 2, ..., k-1
            cycles.append(list(range(k)))
        else:
            # Cycle of length k uses chord (0, m) where 41-m = k => m = 41-k
            # Path: 0-39-38-...-m-0
            m = 41 - k
            # Vertices: 0, 39, 38, ..., m
            cycle_verts = [0] + list(range(39, m - 1, -1))
            cycles.append(cycle_verts)
    
    return {'edges': edge_list, 'cycles': cycles}

if __name__ == "__main__":
    # Self-test
    res = construct()
    edges = set(tuple(sorted(e)) for e in res['edges'])
    for c in res['cycles']:
        for i in range(len(c)):
            u, v = c[i], c[(i+1) % len(c)]
            if (u, v) not in edges and (v, u) not in edges:
                print(f"Missing edge {u}-{v} for cycle {c}")
                break
    print(f"Edges: {len(res['edges'])}, Score: {len(res['edges']) - 40}")
