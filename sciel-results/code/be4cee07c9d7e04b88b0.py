import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145.0
    n = 20
    colors = 20
    
    # Precompute all 4-cycles
    # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges forming a cycle.
    # In a simple graph, a 4-cycle is defined by 4 distinct vertices and a cyclic ordering.
    # However, for the constraint "every 4-cycle is rainbow", we need to check every set of 4 distinct vertices
    # that forms a cycle in the graph. But we don't know the graph yet.
    # Instead, we can precompute all *potential* 4-cycles (sets of 4 vertices) and check if they form a cycle in the current graph.
    # Actually, a 4-cycle in a graph is a subgraph isomorphic to C4.
    # The vertices must be distinct. The edges must be present.
    # There are C(20,4) = 4845 ways to choose 4 vertices.
    # For each set of 4 vertices, there are 3 distinct 4-cycle structures (ways to connect them in a cycle).
    # Total potential 4-cycles = 4845 * 3 = 14535.
    # Let's precompute these.
    
    all_vertices = list(range(n))
    potential_cycles = []
    # Each cycle is represented as a tuple of 4 vertex indices (a,b,c,d) meaning edges (a,b), (b,c), (c,d), (d,a)
    from itertools import combinations
    
    for quad in combinations(all_vertices, 4):
        a,b,c,d = quad
        # The 3 distinct 4-cycles on vertices {a,b,c,d} are:
        # (a,b,c,d), (a,b,d,c), (a,c,b,d)
        potential_cycles.append((a,b,c,d))
        potential_cycles.append((a,b,d,c))
        potential_cycles.append((a,c,b,d))
    
    # For each edge, precompute which cycles it belongs to
    edge_to_cycles = defaultdict(list)
    for cycle in potential_cycles:
        a,b,c,d = cycle
        edges_in_cycle = [frozenset((a,b)), frozenset((b,c)), frozenset((c,d)), frozenset((d,a))]
        for e in edges_in_cycle:
            edge_to_cycles[e].append(cycle)
    
    def is_valid_addition(edge, color, current_adj, current_color):
        """Check if adding (u,v) with 'color' violates any 4-cycle rainbow constraint."""
        u, v = edge
        for cycle in edge_to_cycles[frozenset(edge)]:
            a,b,c,d = cycle
            edges_in_cycle = [frozenset((a,b)), frozenset((b,c)), frozenset((c,d)), frozenset((d,a))]
            # Check if all 4 edges are present in the graph
            # If they are, check if colors are distinct
            present = True
            used_colors = set()
            for e in edges_in_cycle:
                if e == frozenset(edge):
                    used_colors.add(color)
                else:
                    # Check if this edge is in the graph
                    x, y = tuple(e)
                    if y in current_adj[x]:
                        used_colors.add(current_color[(x,y)] if x < y else current_color[(y,x)])
                    else:
                        present = False
                        break
            if present:
                if len(used_colors) != 4:
                    return False
        return True
    
    def get_adj(u):
        return list(current_adj[u])
    
    best_edges = []
    best_score = 0
    
    def score_graph(edges):
        return len(edges)
    
    # Start with a base structure: Latin Square on K_{10,10}
    # Partition vertices into two sets A and B of size 10
    A = list(range(10))
    B = list(range(10, 20))
    
    current_adj = defaultdict(set)
    current_color = {}
    edges = []
    
    # Add K_{10,10} with Latin Square coloring
    # Use a 10x10 Latin Square
    # Simple Latin Square: M[i][j] = (i + j) % 10
    for i, u in enumerate(A):
        for j, v in enumerate(B):
            c = (i + j) % 10
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_color[(u, v)] = c if u < v else c
            edges.append([u, v, c])
    
    # Verify validity of base graph
    valid = True
    for cycle in potential_cycles:
        a,b,c,d = cycle
        edges_in_cycle = [frozenset((a,b)), frozenset((b,c)), frozenset((c,d)), frozenset((d,a))]
        present = True
        used_colors = set()
        for e in edges_in_cycle:
            x, y = tuple(e)
            if y in current_adj[x]:
                used_colors.add(current_color[(x,y)] if x < y else current_color[(y,x)])
            else:
                present = False
                break
        if present and len(used_colors) != 4:
            valid = False
            break
    
    if valid:
        best_edges = list(edges)
        best_score = len(edges)
    
    # Try to add intra-part edges (within A or within B)
    # We have 10 colors (10-19) unused.
    # We need to add edges within A or B such that no 4-cycle becomes non-rainbow.
    
    # Let's use a greedy/local search approach to add more edges.
    # We'll iterate and try to add random edges with random colors, checking validity.
    
    # Candidate edges: all pairs not yet connected
    candidate_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if j not in current_adj[i]:
                candidate_edges.append((i, j))
    
    random.shuffle(candidate_edges)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(candidate_edges)
        for u, v in candidate_edges:
            if time.time() > deadline:
                break
            if v in current_adj[u]:
                continue
            # Try to add (u,v) with a random color
            # Which colors are safe?
            # Let's try all colors or a random subset
            # For efficiency, let's try a random color first, then others if it fails
            random.shuffle(list(range(colors)))
            for c in range(colors):
                if is_valid_addition((u,v), c, current_adj, current_color):
                    # Add the edge
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                    current_color[(u,v)] = c
                    edges.append([u, v, c])
                    candidate_edges.remove((u,v))
                    improved = True
                    if len(edges) > best_score:
                        best_score = len(edges)
                        best_edges = list(edges)
                    break
    
    # Try to optimize by swapping colors or adding more edges
    # Simple local search: try to add any remaining candidate edge with any color
    for u, v in candidate_edges:
        if time.time() > deadline:
            break
        if v in current_adj[u]:
            continue
        for c in range(colors):
            if is_valid_addition((u,v), c, current_adj, current_color):
                current_adj[u].add(v)
                current_adj[v].add(u)
                current_color[(u,v)] = c
                edges.append([u, v, c])
                if len(edges) > best_score:
                    best_score = len(edges)
                    best_edges = list(edges)
                break
    
    return best_edges
