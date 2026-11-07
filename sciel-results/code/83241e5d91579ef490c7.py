"""Solve dense odd-order graphs avoiding equal-degree path endpoints.

Validity: no path s-u-v-t (s!=t) has deg(s)==deg(t).
Equivalent: for every vertex u, all neighbors of u have distinct degrees.
Strategy: CSP-style degree assignment with backtracking + edge maximization.
"""
import time
import random
from collections import Counter

N = 15
DEADLINE = 80.0  # safety margin under 90s kill


def _is_valid(adj, n):
    """Check: for every u, neighbors have distinct degrees."""
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        seen = set()
        for v in adj[u]:
            d = deg[v]
            if d in seen:
                return False
            seen.add(d)
    return True


def _edge_count(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2


def _try_add_edge(adj, n, i, j):
    """Try to add edge (i,j), return True if valid."""
    adj[i].add(j)
    adj[j].add(i)
    if _is_valid(adj, n):
        return True
    else:
        adj[i].discard(j)
        adj[j].discard(i)
        return False


def _remove_edge(adj, n, i, j):
    adj[i].discard(j)
    adj[j].discard(i)


def _maximize_edges(adj, n, deadline):
    """Greedy edge addition with randomization."""
    adj = [set(s) for s in adj]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all missing edges in random order
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
        random.shuffle(missing)
        for i, j in missing:
            if time.time() >= deadline:
                break
            if _try_add_edge(adj, n, i, j):
                improved = True
    return adj


def _backtrack_construct(n, deadline):
    """
    Backtracking search to build a valid graph.
    We assign edges one by one, trying to maximize count.
    Key insight: the constraint is that for each vertex u, its neighbors have distinct degrees.
    
    We'll use a different approach: start with no edges, try to add edges in a smart order.
    Order edges by a heuristic that's likely to lead to valid dense graphs.
    """
    best_adj = [set() for _ in range(n)]
    best_count = 0
    
    # Try multiple random orderings of edge additions
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for attempt in range(200):
        if time.time() >= deadline:
            break
        rng = random.Random(attempt)
        order = all_edges[:]
        rng.shuffle(order)
        
        adj = [set() for _ in range(n)]
        count = 0
        for i, j in order:
            if time.time() >= deadline:
                break
            if _try_add_edge(adj, n, i, j):
                count += 1
        
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
    
    return best_adj, best_count


def _smart_greedy(n, deadline):
    """
    Smarter greedy: prioritize edges that are 'safe' to add.
    An edge (i,j) is safe if adding it doesn't create a violation.
    We process vertices in order of current degree, adding edges to lower-degree vertices first.
    """
    best_adj = [set() for _ in range(n)]
    best_count = 0
    
    for attempt in range(100):
        if time.time() >= deadline:
            break
        rng = random.Random(10000 + attempt)
        adj = [set() for _ in range(n)]
        
        # Build up the graph
        for _ in range(500):  # max iterations
            if time.time() >= deadline:
                break
            
            # Find all missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
            if not missing:
                break
            
            # Score each missing edge by how 'safe' it is
            deg = [len(adj[i]) for i in range(n)]
            scored = []
            for i, j in missing:
                # Check if adding (i,j) would be valid
                adj[i].add(j)
                adj[j].add(i)
                valid = _is_valid(adj, n)
                adj[i].discard(j)
                adj[j].discard(i)
                
                if valid:
                    # Score: prefer edges that connect vertices with very different degrees
                    # or that don't create near-violations
                    score = abs(deg[i] - deg[j]) + rng.random() * 0.1
                    scored.append((score, i, j))
            
            if not scored:
                break
            
            # Add the best-scoring edge
            scored.sort(reverse=True)
            _, i, j = scored[0]
            _try_add_edge(adj, n, i, j)
        
        count = _edge_count(adj, n)
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
    
    return best_adj, best_count


def _degree_partition_search(n, deadline):
    """
    Try specific degree partition structures.
    The key constraint: for each vertex u, its neighbors must have distinct degrees.
    
    This means: if vertex u has degree d, it can be adjacent to at most one vertex of each degree.
    So the maximum degree of any vertex is bounded by the number of distinct degrees present.
    
    For n=15, degrees range from 0 to 14. But we need neighbors of each vertex to have distinct degrees.
    
    Let's think about what degree sequences are possible.
    If all 15 vertices have distinct degrees, then degrees are 0,1,2,...,14.
    But in a simple graph on 15 vertices, having degree 0 and degree 14 is impossible 
    (the vertex with degree 14 is connected to all others, including the one with degree 0).
    
    So we need to find a valid degree sequence where no two vertices have the same degree,
    OR we need to carefully manage which vertices share degrees.
    
    Actually, the constraint is NOT that all vertices have distinct degrees.
    The constraint is: for each vertex u, the set {deg(v) : v adjacent to u} has no duplicates.
    So two non-adjacent vertices CAN have the same degree.
    
    Let's try: can we have all vertices with distinct degrees?
    Degrees must be a permutation of some subset of {0,...,14} of size 15, so {0,1,...,14}.
    But as noted, deg 0 and deg 14 can't coexist.
    
    What about degrees {1,2,...,14, x} where x is something? No, we need 15 distinct values from 0-14.
    The only set of 15 distinct values from 0-14 is {0,1,...,14}, which is infeasible.
    
    So at least two vertices must share a degree. Let's say degrees are:
    a, a, b, c, d, e, f, g, h, i, j, k, l, m, n where all except the pair (a,a) are distinct.
    
    The two vertices with degree a must NOT be adjacent to each other (otherwise they'd see each other's degree).
    Wait, they CAN be adjacent to each other. If u and v both have degree a and are adjacent,
    then u sees v with degree a, and v sees u with degree a. That's fine as long as no OTHER neighbor of u has degree a.
    
    Actually wait: if u and v are both degree a and adjacent, then u's neighbor set includes v (degree a).
    No other neighbor of u can have degree a. Similarly for v.
    But u and v themselves have degree a, and they're not neighbors of themselves.
    
    The constraint is: for vertex u, among its NEIGHBORS, no two have the same degree.
    So if u is adjacent to v (degree a), no other neighbor of u can have degree a.
    u itself is not a neighbor of u.
    
    So the key question: can we construct a graph where the degree sequence has exactly one pair of equal degrees,
    and the rest are all distinct?
    
    Let's try degrees: 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
    Sum = 1+1+2+3+...+14 = 1 + 105 = 106. Must be even. 106 is even. Good.
    
    But can we realize this degree sequence? The vertex with degree 14 is connected to all others.
    The two vertices with degree 1 are each connected to exactly one vertex.
    The degree-14 vertex is connected to both degree-1 vertices.
    So each degree-1 vertex has its single edge to the degree-14 vertex.
    Now the degree-14 vertex has 14 neighbors: the two degree-1 vertices and 12 others.
    Those 12 others have degrees 2,3,...,13.
    For the degree-14 vertex, its neighbors have degrees {1,1,2,3,...,13}.
    There are TWO neighbors with degree 1! Violation!
    
    So this doesn't work. The degree-14 vertex sees two neighbors of degree 1.
    
    What if we avoid degree 14? Use degrees 0,1,2,...,13,14? No, same issue.
    
    What if the two equal-degree vertices are NOT both adjacent to the highest-degree vertex?
    But the highest-degree vertex (say degree 13) is adjacent to 13 of the 14 other vertices.
    So it's adjacent to all but one. If the two degree-a vertices are the one not adjacent... but there are two of them, and only one can be non-adjacent.
    
    Let me think differently. Let's say degrees are d_1 <= d_2 <= ... <= d_15.
    For the vertex with maximum degree D, it has D neighbors. Those D neighbors must all have distinct degrees.
    So D <= (number of distinct degrees among the other 14 vertices).
    
    If all 14 other vertices have distinct degrees, then D <= 14, which is always true.
    But the D neighbors of the max-degree vertex must have distinct degrees, and they're a subset of the 14 others.
    
    Let's try: degrees = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13
    Sum = 0+1+...+12 + 13+13 = 78 + 26 = 104. Even. Good.
    
    Vertex with degree 13: adjacent to 13 of the 14 others. Not adjacent to exactly one.
    The 13 neighbors must have distinct degrees.
    The 14 others have degrees: 0,1,2,...,12,13.
    If the non-adjacent vertex has degree 0, then the 13 neighbors have degrees 1,2,...,13. All distinct. Good.
    
    Now check the vertex with degree 0: no neighbors. Trivially valid.
    
    Now we need to check all other vertices. This is getting complex. Let me just try to construct it.
    
    Actually, let me just use the search approach and see what works.
    """
    best_adj = [set() for _ in range(n)]
    best_count = 0
    
    # Try the random greedy approach with many restarts
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for attempt in range(500):
        if time.time() >= deadline:
            break
        rng = random.Random(attempt * 7 + 12345)
        order = all_edges[:]
        rng.shuffle(order)
        
        adj = [set() for _ in range(n)]
        count = 0
        for i, j in order:
            if time.time() >= deadline:
                break
            if _try_add_edge(adj, n, i, j):
                count += 1
        
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
        
        # Local improvement
        if time.time() < deadline - 5:
            adj = _maximize_edges(adj, n, deadline)
            count = _edge_count(adj, n)
            if count > best_count:
                best_count = count
                best_adj = [set(s) for s in adj]
    
    return best_adj, best_count


def _structured_search(n, deadline):
    """
    Try to build graphs with specific structural properties.
    
    Key insight: if we can make all vertices have distinct degrees, the constraint is automatically satisfied
    (since no two neighbors can have the same degree if all degrees are distinct).
    
    But as analyzed, having all 15 distinct degrees {0,...,14} is infeasible.
    
    What about: make most vertices have distinct degrees, with careful handling of the constraint.
    
    Alternative: use a graph where the degree sequence is such that for each vertex,
    its neighbors happen to have distinct degrees.
    
    Let's try: complete graph minus a matching. n=15, remove 7 edges from a perfect matching (one vertex unmatched).
    Degrees: one vertex has degree 14, seven vertices have degree 13, seven vertices have degree 13.
    Wait: K_15 has all degrees 14. Remove a matching of 7 edges: 14 vertices lose 1 degree, 1 vertex keeps 14.
    So degrees: one vertex degree 14, fourteen vertices degree 13.
    
    Check constraint: the degree-14 vertex is adjacent to all 14 others, all of degree 13.
    So it has 14 neighbors all of degree 13. VIOLATION.
    
    What about K_15 minus a star? Remove edges from vertex 0 to vertices 1-7.
    Vertex 0: degree 7. Vertices 1-7: degree 13. Vertices 8-14: degree 14.
    
    Check vertex 0 (degree 7): neighbors are 8,9,10,11,12,13,14, all degree 14. All same degree! VIOLATION.
    
    What about a bipartite-like structure?
    
    Let me try a different approach: use the fact that we can have at most one neighbor of each degree.
    So if a vertex has degree d, it needs d neighbors with distinct degrees.
    The maximum possible degree is bounded by the number of distinct degrees available.
    
    For n=15, if we use degrees from a set of size k, then max degree <= k.
    To maximize edges, we want high degrees, so we want many distinct degrees.
    
    Let's try: assign degrees as evenly as possible while ensuring the constraint.
    
    Actually, let me just run the search and see what the best we can get is.
    The current best is 14 (star). Let's see if we can do better.
    """
    best_adj = [set() for _ in range(n)]
    best_count = 0
    
    # Start with empty graph and try to build up
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Multiple strategies
    for strategy in range(30):
        if time.time() >= deadline:
            break
        
        adj = [set() for _ in range(n)]
        
        if strategy < 10:
            # Random order
            rng = random.Random(strategy * 111)
            order = all_edges[:]
            rng.shuffle(order)
            for i, j in order:
                if time.time() >= deadline:
                    break
                if _try_add_edge(adj, n, i, j):
                    pass
        elif strategy < 20:
            # Order by vertex index
            order = [(i, j) for i in range(n) for j in range(i+1, n)]
            for i, j in order:
                if time.time() >= deadline:
                    break
                if _try_add_edge(adj, n, i, j):
                    pass
        else:
            # Random with preference for low-degree vertices
            rng = random.Random(strategy * 222)
            remaining = all_edges[:]
            while remaining and time.time() < deadline:
                deg = [len(adj[i]) for i in range(n)]
                # Score edges by sum of degrees of endpoints (prefer low)
                scored = []
                for idx, (i, j) in enumerate(remaining):
                    score = -(deg[i] + deg[j]) + rng.random() * 0.5
                    scored.append((score, idx))
                scored.sort()
                _, idx = scored[0]
                i, j = remaining[idx]
                remaining.pop(idx)
                if _try_add_edge(adj, n, i, j):
                    pass
        
        count = _edge_count(adj, n)
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
        
        # Improve
        if time.time() < deadline - 3:
            adj = _maximize_edges(adj, n, deadline)
            count = _edge_count(adj, n)
            if count > best_count:
                best_count = count
                best_adj = [set(s) for s in adj]
    
    return best_adj, best_count


def construct():
    start = time.time()
    
    # Baseline: star with 14 edges
    star = [set() for _ in range(N)]
    for v in range(1, N):
        star[0].add(v)
        star[v].add(0)
    best_adj = [set(s) for s in star]
    best_count = 14
    
    # Try structured search
    deadline = start + DEADLINE
    adj, count = _structured_search(N, deadline)
    if count > best_count:
        best_count = count
        best_adj = adj
    
    # Try backtracking
    if time.time() < deadline:
        adj, count = _backtrack_construct(N, deadline)
        if count > best_count:
            best_count = count
            best_adj = adj
    
    # Final maximization
    if time.time() < deadline - 2:
        adj = _maximize_edges(best_adj, N, deadline)
        count = _edge_count(adj, N)
        if count > best_count:
            best_count = count
            best_adj = adj
    
    # Convert to list of [u,v] with proper types
    edges = []
    for i in range(N):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
    
    # Final validity check
    adj_final = [set() for _ in range(N)]
    for e in edges:
        adj_final[e[0]].add(e[1])
        adj_final[e[1]].add(e[0])
    if not _is_valid(adj_final, N):
        # Fallback to star
        edges = [[0, v] for v in range(1, N)]
    
    return edges
