<<<<<<< SEARCH
# Precompute all Q3 subgraphs as frozensets of edge tuples
# A Q3 has 8 vertices and 12 edges
def generate_all_q3s():
    """Generate all Q3 subgraphs on vertices 0..15 as frozensets of 12 edges."""
    q3s = set()
    std_q3_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (3, 7), (2, 6)
    ]
    
    # For each choice of 8 vertices from 16
    for vertex_subset in itertools.combinations(range(N), 8):
        # For each bijection from {0..7} to vertex_subset
        for perm in itertools.permutations(vertex_subset):
            # perm[i] is the image of standard vertex i
            new_edges = []
            for u, v in std_q3_edges:
                nu, nv = perm[u], perm[v]
                if nu > nv:
                    nu, nv = nv, nu
                new_edges.append((nu, nv))
            q3s.add(frozenset(new_edges))
    
    return list(q3s)

ALL_Q3S = generate_all_q3s()

def count_q3s(adj):
    """Count the number of Q3 subgraphs in the graph represented by adj (list of int bitsets)."""
    count = 0
    for q3 in ALL_Q3S:
        all_present = True
        for u, v in q3:
            if not ((adj[u] >> v) & 1):
                all_present = False
                break
        if all_present:
            count += 1
    return count

def find_one_q3(adj):
    """Find one Q3 subgraph if it exists. Returns frozenset of 12 edges or None."""
    for q3 in ALL_Q3S:
        all_present = True
        for u, v in q3:
            if not ((adj[u] >> v) & 1):
                all_present = False
                break
        if all_present:
            return q3
    return None
=======
def is_q3_induced(adj, verts):
    """
    Check if the induced subgraph on 'verts' (tuple of 8 vertices) is isomorphic to Q3.
    Q3 is the unique 3-regular bipartite graph on 8 vertices.
    """
    # Check 3-regularity
    for i, v in enumerate(verts):
        deg = 0
        for j, w in enumerate(verts):
            if i != j and ((adj[v] >> w) & 1):
                deg += 1
        if deg != 3:
            return False
            
    # Check bipartiteness
    color = [-1] * 8
    color[0] = 0
    
    # Build local adjacency list for BFS
    local_adj_list = [[] for _ in range(8)]
    for i, v in enumerate(verts):
        for j, w in enumerate(verts):
            if i != j and ((adj[v] >> w) & 1):
                local_adj_list[i].append(j)
                
    stack = [0]
    while stack:
        u = stack.pop()
        for v in local_adj_list[u]:
            if color[v] == -1:
                color[v] = 1 - color[u]
                stack.append(v)
            elif color[v] == color[u]:
                return False
                
    return True

def find_one_q3(adj):
    """
    Find one Q3 subgraph. Returns tuple of 8 vertices or None.
    """
    for verts in itertools.combinations(range(N), 8):
        if is_q3_induced(adj, verts):
            return verts
    return None
>>>>>>> REPLACE

