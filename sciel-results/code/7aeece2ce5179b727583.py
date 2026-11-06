import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Start with K16
    edge_set = set(all_edges)
    adj = [set() for _ in range(n)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all 8-vertex combinations
    all_8_sets = list(itertools.combinations(range(n), 8))
    
    def induced_degree(verts):
        """Return degree of each vertex in the induced subgraph."""
        degs = [0] * 8
        for i in range(8):
            for j in range(8):
                if i != j and verts[j] in adj[verts[i]]:
                    degs[i] += 1
        return degs
    
    def is_q3_induced(verts):
        """Check if induced subgraph on verts is Q3 (3-regular, bipartite, parts of 4)."""
        degs = induced_degree(verts)
        if any(d != 3 for d in degs):
            return False
        # Check bipartite with parts of 4
        # Try all ways to pick 4 vertices as one part
        for part_a in itertools.combinations(range(8), 4):
            a_set = set(part_a)
            b_set = set(range(8)) - a_set
            # Check no edges within a_set
            ok = True
            for i in part_a:
                for j in part_a:
                    if i < j and verts[j] in adj[verts[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check no edges within b_set
            for i in b_set:
                for j in b_set:
                    if i < j and verts[j] in adj[verts[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
        return False
    
    # Greedy: find Q3s and remove edges
    # To speed up, only check 8-sets that could potentially be Q3
    # (all 8 vertices have degree >= 3 in induced subgraph)
    
    iterations = 0
    while time.time() < deadline - 3:
        found = False
        for verts in all_8_sets:
            if time.time() > deadline - 3:
                break
            # Quick check: all vertices must have degree >= 3 in induced subgraph
            min_deg = min(1 if verts[j] in adj[verts[i]] else 0 for i in range(8) for j in range(8) if i != j)
            # Actually just check if any vertex has induced degree < 3
            skip = False
            for i in range(8):
                d = 0
                for j in range(8):
                    if i != j and verts[j] in adj[verts[i]]:
                        d += 1
                if d < 3:
                    skip = True
                    break
            if skip:
                continue
            if is_q3_induced(verts):
                # Found Q3, remove an edge
                # Pick an edge in this Q3
                for i in range(8):
                    for j in range(i+1, 8):
                        if verts[j] in adj[verts[i]]:
                            u, v = verts[i], verts[j]
                            edge_set.discard((u, v))
                            edge_set.discard((v, u))
                            adj[u].discard(v)
                            adj[v].discard(u)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        if not found:
            break
        iterations += 1
    
    result = [list(e) for e in edge_set if e[0] < e[1]]
    return result
