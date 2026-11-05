"""Longest simple path via maximum-weight matching formulation + local refinement."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']
ADJ = [set() for _ in range(N)]
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)
ADJ_LIST = [sorted(ADJ[i]) for i in range(N)]
EDGE_SET = set()
for u, v in EDGES:
    EDGE_SET.add((min(u, v), max(u, v)))

def is_edge(u, v):
    if u == v:
        return False
    if u < v:
        return (u, v) in EDGE_SET
    return (v, u) in EDGE_SET

def is_valid_path(path):
    if not path:
        return False
    if len(set(path)) != len(path):
        return False
    for i in range(len(path) - 1):
        if not is_edge(path[i], path[i + 1]):
            return False
    return True

def max_weight_matching(n, edge_weights):
    """Find maximum weight matching in a graph with n vertices.
    edge_weights: dict mapping (u,v) with u<v to weight.
    Returns list of matched edges as (u,v) tuples.
    Uses a simple augmenting path approach for small graphs.
    """
    # For our transformed graph (2N vertices), use a greedy + local improvement approach
    # Since the graph is sparse and N=100, we use a practical approach.
    
    # Build adjacency for matching
    match_adj = defaultdict(list)
    for (u, v), w in edge_weights.items():
        match_adj[u].append((v, w))
        match_adj[v].append((u, w))
    
    # Greedy matching: sort edges by weight descending, add if both endpoints free
    matched = {}  # vertex -> matched partner
    edges_sorted = sorted(edge_weights.items(), key=lambda x: -x[1])
    for (u, v), w in edges_sorted:
        if u not in matched and v not in matched:
            matched[u] = v
            matched[v] = u
    
    # Local improvement: try to find augmenting paths or 2-flips
    improved = True
    while improved:
        improved = False
        # Try 2-flips: for pairs of matched edges (a,b) and (c,d),
        # if (a,c) and (b,d) are edges with higher total weight, swap
        matched_pairs = [(u, v) for u, v in matched.items() if u < v]
        for i in range(len(matched_pairs)):
            a, b = matched_pairs[i]
            for j in range(i + 1, len(matched_pairs)):
                c, d = matched_pairs[j]
                # Check if swapping improves
                w_ab = edge_weights.get((min(a,b), max(a,b)), 0)
                w_cd = edge_weights.get((min(c,d), max(c,d)), 0)
                w_ac = edge_weights.get((min(a,c), max(a,c)), 0)
                w_bd = edge_weights.get((min(b,d), max(b,d)), 0)
                w_ad = edge_weights.get((min(a,d), max(a,d)), 0)
                w_bc = edge_weights.get((min(b,c), max(b,c)), 0)
                
                if w_ac + w_bd > w_ab + w_cd:
                    # Swap: remove (a,b) and (c,d), add (a,c) and (b,d)
                    del matched[a]
                    del matched[b]
                    del matched[c]
                    del matched[d]
                    matched[a] = c
                    matched[c] = a
                    matched[b] = d
                    matched[d] = b
                    improved = True
                elif w_ad + w_bc > w_ab + w_cd:
                    del matched[a]
                    del matched[b]
                    del matched[c]
                    del matched[d]
                    matched[a] = d
                    matched[d] = a
                    matched[b] = c
                    matched[c] = b
                    improved = True
        
        # Try augmenting paths (length 3): unmatched u - matched (a,b) - unmatched v
        # if edges (u,a) and (b,v) exist
        for u in range(n):
            if u in matched:
                continue
            for v in range(n):
                if v in matched or v == u:
                    continue
                # u is unmatched, v is unmatched
                # Try: u - a, b - v where (a,b) is matched
                for a, b in list(matched.items()):
                    if a > b:
                        continue  # only consider each pair once
                    if a == u or a == v or b == u or b == v:
                        continue
                    w_ua = edge_weights.get((min(u,a), max(u,a)), 0)
                    w_bv = edge_weights.get((min(b,v), max(b,v)), 0)
                    w_ab = edge_weights.get((min(a,b), max(a,b)), 0)
                    if w_ua + w_bv > w_ab:
                        # Augment: remove (a,b), add (u,a) and (b,v)
                        del matched[a]
                        del matched[b]
                        matched[u] = a
                        matched[a] = u
                        matched[b] = v
                        matched[v] = b
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return [(u, matched[u]) for u in matched if u < matched[u]]

def build_transformed_graph():
    """Build a transformed graph for maximum weight matching formulation of longest path.
    
    For each vertex v in original graph, create two nodes: v_in and v_out.
    - Edge (v_in, v_out) with weight = large value (encourages including v)
    - For each original edge (u,v), add edge (u_out, v_in) and (v_out, u_in) with weight 1
    
    A matching in this graph corresponds to a set of vertex-disjoint paths/cycles.
    The largest component that is a path gives our answer.
    """
    # Nodes: 0..N-1 are v_in, N..2N-1 are v_out
    n_transformed = 2 * N
    edge_weights = {}
    
    # Vertex inclusion edges: v_in - v_out with high weight
    LARGE = 1000
    for v in range(N):
        edge_weights[(v, N + v)] = LARGE
    
    # Original edges: u_out - v_in and v_out - u_in with weight 1
    for u, v in EDGES:
        edge_weights[(N + u, v)] = 1
        edge_weights[(N + v, u)] = 1
    
    return n_transformed, edge_weights

def extract_path_from_matching(matching_edges, n_transformed):
    """Extract the longest path from a matching in the transformed graph.
    
    The matching defines a set of vertex-disjoint paths (and possibly cycles).
    We find the longest path component.
    """
    # Build adjacency from matching
    adj = defaultdict(list)
    for u, v in matching_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find all connected components
    visited = set()
    best_path = []
    
    for start in range(n_transformed):
        if start in visited:
            continue
        # BFS/DFS to find component
        component = []
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        
        # Check if this component is a path (all nodes have degree <= 2, and at most 2 have degree 1)
        deg = defaultdict(int)
        for node in component:
            deg[node] = len(adj[node])
        
        degree_counts = defaultdict(int)
        for d in deg.values():
            degree_counts[d] += 1
        
        # A path has: 0 or 2 nodes with degree 1, rest with degree 2, no cycles
        # Actually for a simple path: exactly 2 endpoints (degree 1) and rest degree 2
        # For a cycle: all degree 2
        # For a single node: degree 0
        # For a matching edge: two nodes each degree 1
        
        is_path = True
        if degree_counts[0] > 0:
            # Isolated nodes, skip
            continue
        if degree_counts[1] == 2 and degree_counts[2] == len(component) - 2:
            is_path = True
        elif degree_counts[1] == 0 and degree_counts[2] == len(component):
            # It's a cycle, not a path - skip or break it
            is_path = False
        elif degree_counts[1] == 2 and len(component) == 2:
            # Single edge, it's a path of 2 vertices
            is_path = True
        else:
            is_path = False
        
        if not is_path:
            continue
        
        # Find the endpoints (degree 1 nodes)
        endpoints = [node for node in component if deg[node] == 1]
        if len(endpoints) != 2:
            continue
        
        # Trace the path from one endpoint to the other
        path = [endpoints[0]]
        current = endpoints[0]
        while current != endpoints[1]:
            for neighbor in adj[current]:
                if neighbor not in path:
                    path.append(neighbor)
                    current = neighbor
                    break
        
        # Convert transformed node IDs to original vertex IDs
        # v_in = v, v_out = N + v
        # A path in transformed graph: v1_in - v1_out - v2_in - v2_out - ...
        # We need to extract the sequence of original vertices
        
        orig_path = []
        for node in path:
            if node < N:
                # This is v_in
                orig_path.append(node)
            else:
                # This is v_out, the vertex is node - N
                # But we already added it when we saw v_in
                pass
        
        # Actually, let me re-extract: the path alternates in/out
        # v1_in, v1_out, v2_in, v2_out, ...
        # So the original vertices are at positions 0, 2, 4, ... (the v_in nodes)
        orig_vertices = []
        for i, node in enumerate(path):
            if node < N:
                orig_vertices.append(node)
        
        if len(orig_vertices) > len(best_path):
            best_path = orig_vertices
    
    return best_path

def extend_path(path, seen):
    """Greedy extend from both ends using Warnsdorff-like heuristic."""
    changed = True
    while changed:
        changed = False
        front = path[0]
        cands = [w for w in ADJ[front] if w not in seen]
        if cands:
            w = min(cands, key=lambda x: (len(ADJ[x]) - len(seen & ADJ[x]), x))
            path.insert(0, w)
            seen.add(w)
            changed = True
        back = path[-1]
        cands = [w for w in ADJ[back] if w not in seen]
        if cands:
            w = min(cands, key=lambda x: (len(ADJ[x]) - len(seen & ADJ[x]), x))
            path.append(w)
            seen.add(w)
            changed = True
    return path

def greedy_path(start, rng):
    """Build a path starting at `start` using randomized Warnsdorff heuristic."""
    path = [start]
    seen = {start}
    while True:
        back = path[-1]
        cands = [w for w in ADJ[back] if w not in seen]
        if not cands:
            break
        cands.sort(key=lambda w: (len([x for x in ADJ[w] if x not in seen]), w))
        k = min(3, len(cands))
        chosen = rng.choice(cands[:k])
        path.append(chosen)
        seen.add(chosen)
    extend_path(path, seen)
    return path

def two_opt(path):
    """Apply 2-opt reversals."""
    n = len(path)
    if n < 3:
        return path
    best = list(path)
    best_len = n
    for i in range(n - 1):
        for j in range(i + 2, n):
            cand = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
            ok = True
            if i > 0:
                if not is_edge(cand[i - 1], cand[i]):
                    ok = False
            if ok and j + 1 < n:
                if not is_edge(cand[j], cand[j + 1]):
                    ok = False
            if not ok:
                continue
            seen = set(cand)
            ext = extend_path(cand, seen)
            if len(ext) > best_len:
                best = ext
                best_len = len(ext)
    return best

def local_search(path, rng, deadline):
    """Iterate 2-opt + extension until no improvement or deadline."""
    cur = list(path)
    seen = set(cur)
    cur = extend_path(cur, seen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        nxt = two_opt(cur)
        if len(nxt) > len(cur):
            cur = nxt
            improved = True
    return cur

def construct():
    deadline = time.time() + 85.0  # safety margin under 90s
    rng = random.Random(748)
    best = []
    
    # Phase 1: Maximum weight matching formulation
    if time.time() < deadline:
        n_transformed, edge_weights = build_transformed_graph()
        matching = max_weight_matching(n_transformed, edge_weights)
        path_from_matching = extract_path_from_matching(matching, n_transformed)
        
        if path_from_matching and is_valid_path(path_from_matching):
            best = list(path_from_matching)
            seen = set(best)
            best = extend_path(best, seen)
            if is_valid_path(best):
                pass
            else:
                best = list(path_from_matching)
    
    # Phase 2: Deterministic greedy from every start
    for start in range(N):
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        while True:
            back = path[-1]
            cands = [w for w in ADJ[back] if w not in seen]
            if not cands:
                break
            cands.sort(key=lambda w: (len([x for x in ADJ[w] if x not in seen]), w))
            path.append(cands[0])
            seen.add(cands[0])
        path = extend_path(path, seen)
        if len(path) > len(best):
            best = list(path)
    
    # Phase 3: Multi-start random greedy + 2-opt + extension
    attempts = 0
    max_attempts = 300
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        start = rng.randrange(N)
        path = greedy_path(start, rng)
        path = local_search(path, rng, deadline)
        if len(path) > len(best):
            best = list(path)
    
    # Phase 4: Final local search on best
    if time.time() < deadline:
        best = local_search(best, rng, deadline)
    
    # Fallback
    if not best:
        best = [0]
    if not is_valid_path(best):
        best = [0]
    return best
# EVOLVE-BLOCK-END
