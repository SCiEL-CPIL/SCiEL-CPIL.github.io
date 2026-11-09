"""Pentagon-free graphs without independent 9-sets (Ramsey number R(C5,K9))."""
import time
import random
import itertools
from collections import deque

# EVOLVE-BLOCK-START

def _has_c5(adj, n):
    """Check if graph with adjacency list `adj` on n vertices contains a C5 subgraph."""
    # For each 5-tuple of vertices, check if they form a cycle
    # Use a more efficient approach: check all combinations of 5 vertices
    for combo in itertools.combinations(range(n), 5):
        a, b, c, d, e = combo
        # Check if they form a 5-cycle in any order
        # A 5-cycle on {a,b,c,d,e} means there exists a permutation where consecutive are adjacent
        # The cycle graph on 5 vertices has edges forming a single cycle
        # Check: can we arrange these 5 in a cycle where each consecutive pair is adjacent?
        # There are (5-1)!/2 = 12 distinct 5-cycles on 5 labeled vertices
        # Let's check all 12 cycle orderings
        vertices = [a, b, c, d, e]
        for perm in itertools.permutations(vertices):
            p0, p1, p2, p3, p4 = perm
            if (adj[p0][p1] and adj[p1][p2] and adj[p2][p3] and adj[p3][p4] and adj[p4][p0]):
                return True
    return False

def _alpha_check(adj, n):
    """Check if graph has an independent set of size 9 (i.e., alpha >= 9)."""
    # Try to find an independent set of size 9
    # Use backtracking
    def backtrack(remaining, chosen, size):
        if size == 9:
            return True
        if len(remaining) < size:
            return False
        # Pick first vertex
        v = remaining[0]
        rest = remaining[1:]
        # Include v: recurse on non-neighbors of v in rest
        non_neighbors = [u for u in rest if not adj[v][u]]
        if backtrack(non_neighbors, chosen + [v], size + 1):
            return True
        # Exclude v
        if backtrack(rest, chosen, size):
            return True
        return False
    
    return backtrack(list(range(n)), [], 0)

def _validate(n, edges):
    """Check if graph is C5-free and has alpha < 9."""
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        if u >= n or v >= n:
            return False
        if u == v:
            return False
        adj[u][v] = True
        adj[v][u] = True
    
    if _has_c5(adj, n):
        return False
    if _alpha_check(adj, n):
        return False
    return True

def _try_n33(start_time, deadline):
    """Try to find a valid C5-free graph on 33 vertices with alpha < 9."""
    # Strategy: Start from 8 disjoint K4s (n=32, alpha=8) and try to add vertex 32
    # and modify edges to maintain C5-freeness and alpha < 9
    
    # Base: 8 disjoint K4s
    base_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    base_n = 32
    
    # We need to add vertex 32 and connect it to some vertices
    # to reduce the independence number
    # In the base graph, the maximum independent set picks 1 vertex from each K4
    # If vertex 32 is connected to at least one vertex in each K4, then any independent set
    # containing 32 can pick at most 1 from each K4 (the ones not adjacent to 32)
    # But we need total independent set size < 9
    # If 32 is adjacent to exactly 1 vertex in each K4, then:
    #   - Independent set with 32: can pick 1 non-neighbor from each K4 = 8 vertices + 32 = 9. BAD.
    #   - Independent set without 32: can pick 1 from each K4 = 8. OK.
    # So we need 32 to be adjacent to at least 2 vertices in at least one K4,
    # OR adjacent to 1 vertex in each K4 but with some cross-edges to prevent 9-set.
    
    # Alternative: Add cross-edges between K4s to reduce alpha
    # If we add an edge between two K4s, then we can't pick one vertex from each
    # This reduces alpha
    
    # Let's try: 8 disjoint K4s plus vertex 32 connected to 1 vertex in each K4
    # Then add cross-edges between K4s to ensure no 9-independent-set exists
    
    # Actually, the simplest approach: try random modifications
    # Start with 8 K4s + vertex 32 with random connections
    # Then check validity
    
    best_result = None
    
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        
        edges = list(base_edges)
        
        # Connect vertex 32 to k vertices (one from each K4, or more)
        for b in range(8):
            # Connect 32 to 1 or 2 vertices in K4_b
            num_conn = random.randint(1, 2)
            chosen = random.sample(range(4), num_conn)
            for i in chosen:
                edges.append([32, 4 * b + i])
        
        # Add some cross-edges between K4s to further reduce alpha
        num_cross = random.randint(0, 10)
        for _ in range(num_cross):
            b1 = random.randint(0, 7)
            b2 = random.randint(0, 7)
            if b1 == b2:
                continue
            i1 = random.randint(0, 3)
            i2 = random.randint(0, 3)
            u = 4 * b1 + i1
            v = 4 * b2 + i2
            # Check if edge already exists
            if not any((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u) for e in edges):
                edges.append([u, v])
        
        # Validate
        if _validate(33, edges):
            best_result = (33, edges)
            return best_result
    
    return best_result

def _try_n34(start_time, deadline):
    """Try to find a valid C5-free graph on 34 vertices with alpha < 9."""
    # This is harder. Start from best 33-vertex graph if found, or from modified 32-vertex.
    # Given time constraints, this is unlikely to succeed quickly.
    # We'll do a very limited search.
    
    # Start with 8 K4s + 2 extra vertices (32, 33)
    base_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    for attempt in range(100):
        if time.time() > deadline - 3:
            break
        
        edges = list(base_edges)
        
        # Connect vertices 32 and 33 to K4s
        for extra_v in [32, 33]:
            for b in range(8):
                num_conn = random.randint(1, 2)
                chosen = random.sample(range(4), num_conn)
                for i in chosen:
                    edges.append([extra_v, 4 * b + i])
        
        # Connect 32 and 33 to each other
        if random.random() < 0.5:
            edges.append([32, 33])
        
        # Add cross-edges
        num_cross = random.randint(5, 15)
        all_vertices = list(range(34))
        for _ in range(num_cross):
            u = random.choice(all_vertices)
            v = random.choice(all_vertices)
            if u == v:
                continue
            if not any((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u) for e in edges):
                edges.append([u, v])
        
        if _validate(34, edges):
            return (34, edges)
    
    return None

def _try_n35(start_time, deadline):
    """Try to find a valid C5-free graph on 35 vertices with alpha < 9."""
    # Very unlikely to succeed with random search in available time
    base_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    for attempt in range(50):
        if time.time() > deadline - 2:
            break
        
        edges = list(base_edges)
        
        for extra_v in [32, 33, 34]:
            for b in range(8):
                num_conn = random.randint(1, 3)
                chosen = random.sample(range(4), min(num_conn, 4))
                for i in chosen:
                    edges.append([extra_v, 4 * b + i])
        
        # Connect extras among themselves
        extras = [32, 33, 34]
        for i in range(len(extras)):
            for j in range(i + 1, len(extras)):
                if random.random() < 0.5:
                    edges.append([extras[i], extras[j]])
        
        # Cross-edges
        num_cross = random.randint(10, 20)
        all_vertices = list(range(35))
        for _ in range(num_cross):
            u = random.choice(all_vertices)
            v = random.choice(all_vertices)
            if u == v:
                continue
            if not any((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u) for e in edges):
                edges.append([u, v])
        
        if _validate(35, edges):
            return (35, edges)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with known valid solution: 8 disjoint K4s, n=32
    best_n = 32
    best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    # Try to improve to n=33
    result = _try_n33(start_time, deadline)
    if result is not None:
        best_n, best_edges = result
    
    # Try to improve to n=34
    if time.time() < deadline - 10:
        result = _try_n34(start_time, deadline)
        if result is not None:
            best_n, best_edges = result
    
    # Try to improve to n=35
    if time.time() < deadline - 5:
        result = _try_n35(start_time, deadline)
        if result is not None:
            best_n, best_edges = result
    
    # Deduplicate edges for cleanliness
    seen = set()
    deduped_edges = []
    for u, v in best_edges:
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            deduped_edges.append([u, v])
    
    return {'n': best_n, 'edges': deduped_edges}

# EVOLVE-BLOCK-END
