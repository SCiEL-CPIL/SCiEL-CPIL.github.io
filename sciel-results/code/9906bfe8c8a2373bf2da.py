"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _count_distinct_cycle_lengths(edges, n):
    """Count distinct simple cycle lengths in a graph."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    lengths = set()
    
    # For each start node, DFS only visiting nodes >= start to avoid duplicates
    for start in range(n):
        # DFS path: list of nodes, last node is current
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) >= 4:
                # Check if we can close the cycle back to start
                if start in adj[node]:
                    lengths.add(len(path))
            # Extend path
            for nb in adj[node]:
                if nb > start and nb not in path:
                    stack.append((nb, path + [nb]))
    
    return len(lengths)

def _is_valid(edges, n, min_deg):
    """Check if graph is valid: min degree, no loops, no multiedges, has cycle, triangle-free."""
    if len(edges) == 0:
        return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) < min_deg:
            return False
    # Check triangle-free
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and w in adj[u]:
                    return False
    # Check has a cycle
    # A graph with n vertices and m edges has a cycle if m >= n (for connected) 
    # but we need to actually check. Simple: if any vertex has degree >= 2 and graph is not a forest
    # Use: if number of edges >= n, likely has cycle. But let's be safe.
    # Actually, for min degree 3, n=12, edges >= 18 > 12, so always has cycle.
    return True

def construct():
    n = 12
    min_deg = 3
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    
    # Precomputed good candidate: two disjoint K_4 minus a perfect matching? 
    # Or try a known construction first, then search.
    
    # Try a bipartite construction: 6+6 split
    # K_{6,6} has too many cycle lengths.
    # Try: take 3 disjoint 4-cycles with added edges to boost degree.
    
    # First, try a fixed construction that's known to work well:
    # The "cube" Q3 has 8 vertices, all cycles 4 or 6. Add 4 more vertices.
    
    # Let's try: 12 vertices, bipartite with parts {0-5} and {6-11}
    # Each vertex in part A connects to 3 in part B, and vice versa (degree 3 each)
    # This is a 3-regular bipartite graph on 12 vertices.
    # Cycle lengths in 3-regular bipartite: 4, 6, 8, ...
    
    # Best strategy: random 3-regular bipartite graphs, check cycle length count.
    
    best_edges = None
    best_count = 999
    
    # Try fixed candidates first
    # Candidate 1: The original (two hexagons + rungs) - has 5 distinct lengths
    # Candidate 2: A 3-regular bipartite graph that minimizes cycle variety
    
    # Generate random 3-regular bipartite graphs
    random.seed(42)
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        # Random 3-regular bipartite graph: part A = [0..5], part B = [6..11]
        # Each A vertex connects to 3 B vertices, each B vertex to 3 A vertices
        # Use configuration model
        a_stubs = [0]*3 + [1]*3 + [2]*3 + [3]*3 + [4]*3 + [5]*3
        b_stubs = [6]*3 + [7]*3 + [8]*3 + [9]*3 + [10]*3 + [11]*3
        
        random.shuffle(b_stubs)
        edges = []
        for i in range(18):
            u = a_stubs[i]
            v = b_stubs[i]
            if (u, v) not in [(e[0], e[1]) for e in edges]:
                edges.append((u, v))
        
        if len(edges) != 18:
            continue
        
        if not _is_valid(edges, n, min_deg):
            continue
        
        cnt = _count_distinct_cycle_lengths(edges, n)
        if cnt < best_count:
            best_count = cnt
            best_edges = edges.copy()
            if best_count == 1:
                break
    
    if best_edges is not None:
        return [list(e) for e in sorted(best_edges)]
    
    # Fallback: return the original construction
    es = set()
    for i in range(6):
        for u, v in [(i, (i+1) % 6), (i+6, (i+1) % 6 + 6), (i, i+6)]:
            es.add(tuple(sorted((u, v))))
    return [list(e) for e in sorted(es)]
# EVOLVE-BLOCK-END
