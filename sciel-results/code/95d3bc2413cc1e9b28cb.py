"""Rewritten program: hybrid local search with greedy pruning and exact verification."""
import random
import time
import itertools
from collections import defaultdict

# ---------- helpers ----------

def _build_adjacency(edges):
    """Return a set of frozensets for fast lookup and an adjacency list."""
    adj_set = set()
    for u, v in edges:
        if u != v:
            a, b = (u, v) if u < v else (v, u)
            adj_set.add(frozenset((a, b)))
    return adj_set

def _is_k4_free(adj_set, n=18):
    """Check that no K4 exists in the graph."""
    # Precompute for each vertex the set of neighbors
    nbrs = [set() for _ in range(n)]
    for e in adj_set:
        u, v = tuple(e)
        nbrs[u].add(v)
        nbrs[v].add(u)
    
    # Check all 4-cliques
    for a in range(n):
        for b in nbrs[a]:
            if b <= a:
                continue
            for c in nbrs[b]:
                if c <= b:
                    continue
                if c not in nbrs[a]:
                    continue
                for d in nbrs[c]:
                    if d <= c:
                        continue
                    if d in nbrs[a] and d in nbrs[b]:
                        return False
    return True

def _max_triangle_free_size(adj_set, n=18):
    """Exact branch and bound: largest induced subgraph that is triangle-free."""
    # For each subset of vertices, check if it's triangle-free.
    # We want the largest such subset.
    # Use branch and bound over vertex inclusion.
    
    # Precompute for each pair whether they form an edge
    edge_exists = [[False]*n for _ in range(n)]
    for e in adj_set:
        u, v = tuple(e)
        edge_exists[u][v] = True
        edge_exists[v][u] = True
    
    # A subset S is triangle-free if no three vertices in S form a triangle.
    # Equivalently, the induced subgraph on S has no triangles.
    
    # Branch and bound: try to include or exclude each vertex.
    # State: (current_set, next_vertex_to_consider)
    # Triangle-free check: maintain the set of "forbidden" vertices that would create a triangle.
    
    best = [0]
    
    def _is_triangle_free_subset(vertices):
        """Check if the given set of vertices induces a triangle-free subgraph."""
        if len(vertices) <= 2:
            return True
        # Check all triples
        vlist = list(vertices)
        for i in range(len(vlist)):
            for j in range(i+1, len(vlist)):
                a, b = vlist[i], vlist[j]
                if not edge_exists[a][b]:
                    continue
                for k in range(j+1, len(vlist)):
                    c = vlist[k]
                    if edge_exists[a][c] and edge_exists[b][c]:
                        return False
        return True
    
    # Branch and bound over subsets
    # We'll use a different approach: enumerate all subsets by size descending
    # For n=18, 2^18 = 262144 subsets. We can check them in order of size.
    
    # Actually, let's do a smarter approach:
    # Start with all 18 vertices. If not triangle-free, remove one vertex at a time.
    # But that's expensive. Let's just enumerate subsets by decreasing size.
    
    all_vertices = list(range(n))
    
    for size in range(n, 0, -1):
        found = False
        for combo in itertools.combinations(all_vertices, size):
            if _is_triangle_free_subset(set(combo)):
                best[0] = size
                return best[0]
    
    return best[0]

def _greedy_max_triangle_free(adj_set, n=18):
    """Fast greedy estimate of max triangle-free induced subgraph size."""
    # Try multiple random orderings and take the best
    best = 0
    for _ in range(20):
        order = list(range(n))
        random.shuffle(order)
        # Greedily build a triangle-free set
        selected = set()
        for v in order:
            # Check if adding v creates a triangle
            ok = True
            # v creates a triangle with two existing vertices a, b if a-b is edge and a-v is edge and b-v is edge
            for a in selected:
                if not _edge_in_adj(adj_set, a, v):
                    continue
                for b in selected:
                    if b <= a:
                        continue
                    if b == v:
                        continue
                    if _edge_in_adj(adj_set, a, b) and _edge_in_adj(adj_set, b, v):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.add(v)
        if len(selected) > best:
            best = len(selected)
    return best

def _edge_in_adj(adj_set, u, v):
    a, b = (u, v) if u < v else (v, u)
    return frozenset((a, b)) in adj_set

def _toggle_edge(edges, adj_set, u, v, add):
    """Toggle an edge, return new edges list and adj_set."""
    a, b = (u, v) if u < v else (v, u)
    key = frozenset((a, b))
    if add:
        if key not in adj_set:
            new_adj = adj_set | {key}
            new_edges = edges + [[a, b]]
            return new_edges, new_adj
    else:
        if key in adj_set:
            new_adj = adj_set - {key}
            new_edges = [e for e in edges if not (e[0] == a and e[1] == b) and not (e[0] == b and e[1] == a)]
            return new_edges, new_adj
    return edges, adj_set

# ---------- main construct ----------

def construct():
    n = 18
    deadline = time.time() + 110  # safety margin before 120s
    
    # Start with complete tripartite K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    
    adj_set = _build_adjacency(edges)
    current_score = _max_triangle_free_size(adj_set, n)
    best_edges = edges[:]
    best_score = current_score
    
    # Local search: try to reduce the max triangle-free induced subgraph
    # by adding edges (within parts) or removing edges (between parts)
    
    all_pairs = list(itertools.combinations(range(n), 2))
    
    # Precompute current edge status
    edge_status = {}
    for u, v in all_pairs:
        edge_status[(u, v)] = frozenset((u, v)) in adj_set
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Try random edge toggles
        improved = False
        for _ in range(200):
            if time.time() >= deadline:
                break
            
            u, v = random.choice(all_pairs)
            currently_present = edge_status[(u, v)]
            
            # Decide whether to add or remove
            if currently_present:
                # Remove edge
                new_edges, new_adj = _toggle_edge(edges, adj_set, u, v, add=False)
                edge_status[(u, v)] = False
            else:
                # Add edge - but must keep K4-free
                new_edges, new_adj = _toggle_edge(edges, adj_set, u, v, add=True)
                if not _is_k4_free(new_adj, n):
                    # Revert
                    new_edges, new_adj = edges, adj_set
                    edge_status[(u, v)] = True
                    continue
                edge_status[(u, v)] = True
            
            # Greedy check first
            greedy_score = _greedy_max_triangle_free(new_adj, n)
            if greedy_score >= best_score:
                # Revert
                new_edges, new_adj = _toggle_edge(new_edges, new_adj, u, v, add=not edge_status[(u, v)])
                edge_status[(u, v)] = not edge_status[(u, v)]
                continue
            
            # Exact check
            exact_score = _max_triangle_free_size(new_adj, n)
            if exact_score < best_score:
                # Accept
                best_score = exact_score
                best_edges = new_edges[:]
                edges = new_edges
                adj_set = new_adj
                improved = True
            else:
                # Revert
                new_edges, new_adj = _toggle_edge(new_edges, new_adj, u, v, add=not edge_status[(u, v)])
                edge_status[(u, v)] = not edge_status[(u, v)]
        
        if improved:
            # Continue searching from new position
            edges = best_edges[:]
            adj_set = _build_adjacency(edges)
            # Rebuild edge_status
            for u, v in all_pairs:
                edge_status[(u, v)] = frozenset((u, v)) in adj_set
        elif step % 10 == 0:
            # Occasional restart with different random seed
            random.seed()
    
    return best_edges
