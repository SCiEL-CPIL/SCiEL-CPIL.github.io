"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
from itertools import combinations

# We want a K4-free graph on 18 vertices that minimizes the size of the
# largest triangle-free induced subgraph.

# Strategy: Use a graph that is "dense" in a way that forces every large
# subset to contain a triangle, while keeping the graph K4-free.
#
# The complete tripartite graph K6,6,6 gives a score of 12 (any two parts
# form a triangle-free induced subgraph of size 12).
#
# We can do better by using a more structured graph. Consider the following:
# Partition vertices into 3 groups of 6. Make all edges between groups (as in
# tripartite), but also add some edges WITHIN groups in a way that breaks the
# triangle-free property of pairs of groups, while maintaining K4-freeness.
#
# Actually, a better approach: use a graph based on a known construction.
# The complement of a triangle-free graph is K4-free iff the original graph
# has no independent set of size 4. So we want a graph G on 18 vertices such
# that:
#   1. G is K4-free (no 4-clique)
#   2. The largest triangle-free induced subgraph is as small as possible.
#
# Let's think about it differently. A triangle-free induced subgraph on k
# vertices means those k vertices have no triangle in G. We want to minimize
# the maximum such k.
#
# By Ramsey theory, R(3,3) = 6, so any 6-vertex graph contains either a
# triangle or an independent set of size 3. But we need triangle-free, not
# independent set.
#
# Let me try a computational approach: start with a good candidate and use
# local search to improve it.

def _is_k4_free(adj):
    """Check if graph with adjacency matrix is K4-free."""
    n = len(adj)
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _max_triangle_free(adj):
    """Find the largest triangle-free induced subgraph size via branch and bound."""
    n = len(adj)
    # adj is n x n boolean matrix
    # We search for the largest subset with no triangle.
    
    best = 0
    
    # Use bitmasks for efficiency
    # A subset S is triangle-free if for every triangle (i,j,k) in G, S doesn't contain all three.
    
    # Precompute all triangles
    triangles = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                continue
            for k in range(j+1, n):
                if adj[i][k] and adj[j][k]:
                    triangles.append((i, j, k))
    
    if not triangles:
        return n
    
    # Branch and bound: build up a set, ensure no triangle is fully contained.
    # State: current set S, next vertex to consider, upper bound on remaining.
    
    def dfs(S_mask, start, size):
        nonlocal best
        if size > best:
            best = size
        # Upper bound: size + (n - start)
        if size + (n - start) <= best:
            return
        for v in range(start, n):
            if S_mask & (1 << v):
                continue
            # Check if adding v creates a triangle
            # A triangle is created if there exist i, j in S such that (i,j,v) is a triangle
            # i.e., adj[i][j] and adj[i][v] and adj[j][v]
            ok = True
            for i in range(n):
                if not (S_mask & (1 << i)):
                    continue
                if not adj[i][v]:
                    continue
                for j in range(i+1, n):
                    if not (S_mask & (1 << j)):
                        continue
                    if adj[i][j] and adj[j][v]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                dfs(S_mask | (1 << v), v + 1, size + 1)
    
    dfs(0, 0, 0)
    return best

def _construct_initial():
    """Construct an initial K4-free graph on 18 vertices."""
    # Start with complete tripartite K6,6,6
    n = 18
    adj = [[False]*n for _ in range(n)]
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = True
                    adj[b][a] = True
    return adj

def _local_search(adj, max_time=100):
    """Local search to minimize the largest triangle-free induced subgraph."""
    n = len(adj)
    start_time = time.time()
    
    # Current score
    current_score = _max_triangle_free(adj)
    
    # Try removing edges to reduce the triangle-free subgraph size
    # Actually, removing edges makes it EASIER to be triangle-free, so that's bad.
    # We want to ADD edges to create more triangles, making it harder to find
    # large triangle-free induced subgraphs.
    
    # But we must maintain K4-freeness.
    
    # Try adding edges that don't create a K4.
    
    best_adj = [row[:] for row in adj]
    best_score = current_score
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Find candidate edges to add (non-edges that don't create K4)
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    continue
                # Check if adding (u,v) creates a K4
                # A K4 is created if there exist w, x such that u,v,w,x form a K4
                # i.e., w and x are both adjacent to u and v, and w-x is an edge
                creates_k4 = False
                for w in range(n):
                    if w == u or w == v or not adj[u][w] or not adj[v][w]:
                        continue
                    for x in range(w+1, n):
                        if x == u or x == v or x == w:
                            continue
                        if adj[u][x] and adj[v][x] and adj[w][x]:
                            creates_k4 = True
                            break
                    if creates_k4:
                        break
                if not creates_k4:
                    candidates.append((u, v))
        
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() - start_time > max_time - 5:
                break
            
            # Try adding edge (u, v)
            adj[u][v] = True
            adj[v][u] = True
            
            new_score = _max_triangle_free(adj)
            
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                improved = True
            else:
                # Revert
                adj[u][v] = False
                adj[v][u] = False
        
        if not improved:
            # Try removing some edges to allow more additions
            # Or restart with a different initial graph
            break
    
    return best_adj, best_score

def construct():
    deadline = time.time() + 140  # 140s safety margin
    
    # Start with complete tripartite K6,6,6
    adj = _construct_initial()
    
    # Verify it's K4-free
    assert _is_k4_free(adj), "Initial graph is not K4-free"
    
    current_score = _max_triangle_free(adj)
    
    # Local search: add edges to create more triangles
    adj, current_score = _local_search(adj, max_time=120)
    
    # Convert to edge list
    edges = []
    for u in range(18):
        for v in range(u+1, 18):
            if adj[u][v]:
                edges.append([u, v])
    
    return edges
