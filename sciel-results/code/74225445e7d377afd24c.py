import time
import random

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Strategy: Use 8 disjoint copies of K4.
    n = 32.
    alpha = 8 (one vertex from each K4).
    C5-free: K4 has no C5, and components are disjoint.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s limit
    
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Check if we have time to attempt a 33-vertex construction
    if time.time() < deadline - 5:
        # Try to find a 33-vertex C5-free graph with alpha < 9
        # via local search from the 32-vertex base
        result = _try_33_vertex(start_time, deadline)
        if result is not None:
            return result
    
    return {'n': n, 'edges': edges}

def _has_c5(n, edge_set):
    """Check if graph has a 5-cycle. edge_set is a set of frozensets."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    for e in edge_set:
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
    
    # Check all 5-vertex subsets for C5
    from itertools import combinations
    for verts in combinations(range(n), 5):
        vs = set(verts)
        # Check if induced subgraph contains C5
        # A C5 on 5 vertices means the induced subgraph has a 5-cycle
        # Simplest: check if the induced subgraph on these 5 vertices is C5 or contains C5
        # For 5 vertices, C5 as subgraph means there's a cycle of length 5
        # Build subgraph
        sub_edges = []
        for i in range(5):
            for j in range(i+1, 5):
                if verts[j] in adj[verts[i]]:
                    sub_edges.append((i, j))
        # Check if this 5-vertex graph contains a 5-cycle
        # Enumerate all 5-cycles (12 distinct up to rotation/reflection)
        # Simpler: check if there's a Hamiltonian cycle in the subgraph
        # For 5 vertices, just check all permutations
        for perm in [(0,1,2,3,4), (0,1,2,4,3), (0,1,3,2,4), (0,1,3,4,2),
                     (0,1,4,2,3), (0,1,4,3,2), (0,2,1,3,4), (0,2,1,4,3),
                     (0,2,3,1,4), (0,2,4,1,3), (0,3,1,2,4), (0,3,1,4,2),
                     (0,3,2,1,4), (0,3,2,4,1), (0,3,4,1,2), (0,3,4,2,1),
                     (0,4,1,2,3), (0,4,1,3,2), (0,4,2,1,3), (0,4,2,3,1),
                     (0,4,3,1,2), (0,4,3,2,1)]:
            ok = True
            for k in range(5):
                a, b = perm[k], perm[(k+1)%5]
                if (a,b) not in sub_edges and (b,a) not in sub_edges:
                    ok = False
                    break
            if ok:
                return True
    return False

def _alpha_ge(n, edge_set, k=9):
    """Check if graph has independent set of size k."""
    adj = [set() for _ in range(n)]
    for e in edge_set:
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
    
    from itertools import combinations
    for verts in combinations(range(n), k):
        vs = set(verts)
        independent = True
        for i in range(k):
            for j in range(i+1, k):
                if verts[j] in adj[verts[i]]:
                    independent = False
                    break
            if not independent:
                break
        if independent:
            return True
    return False

def _try_33_vertex(start_time, deadline):
    """Try to construct a 33-vertex C5-free graph with alpha < 9."""
    # Start from 8xK4 and try adding vertex 32
    base_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                base_edges.append((base + i, base + j))
    
    # Vertex 32 must connect to all 4 of some K4 to avoid alpha=9
    # But that creates C5. Try other patterns.
    # Try: connect 32 to all of K4_0, but modify K4_0 to break C5
    # Alternative: use a different base structure
    
    # Try randomized local search
    random.seed(42)
    best = None
    
    for attempt in range(200):
        if time.time() > deadline - 2:
            break
        
        # Start with 8xK4
        edges_set = set()
        for e in base_edges:
            edges_set.add(frozenset(e))
        
        # Try adding vertex 32 with random connections
        # To avoid alpha=9, 32 must be adjacent to all of at least one K4
        # Pick a random K4 to fully connect to
        target_k4 = random.randint(0, 7)
        target_verts = [4*target_k4 + i for i in range(4)]
        
        for v in target_verts:
            edges_set.add(frozenset((32, v)))
        
        # Add some random other edges from 32
        other_verts = [i for i in range(32) if i not in target_verts]
        num_extra = random.randint(0, 8)
        for v in random.sample(other_verts, num_extra):
            edges_set.add(frozenset((32, v)))
        
        # Check C5-free
        if _has_c5(33, edges_set):
            continue
        
        # Check alpha < 9
        if _alpha_ge(33, edges_set, 9):
            continue
        
        # Valid!
        edges_list = [[u, v] for e in edges_set for u, v in [tuple(e)]]
        return {'n': 33, 'edges': edges_list}
    
    return None
