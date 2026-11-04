import time
import random
from collections import deque

def _verify_girth(adj, n, g):
    """
    Verify that no cycle is shorter than g using non-backtracking walks.
    For each vertex s, do a BFS in the non-backtracking state space.
    States are (u, v) where v is the "came from" vertex.
    If we reach a state (s, v) with v != -1 and the path length < g, we have a cycle < g.
    """
    for s in range(n):
        visited = set()
        visited.add((s, -1))
        q = deque()
        q.append((s, -1, 0))
        
        while q:
            u, p, d = q.popleft()
            if d >= g:
                continue
            for v in adj[u]:
                if v == p:
                    continue
                if (v, u) in visited:
                    continue
                visited.add((v, u))
                if v == s:
                    if d + 1 < g:
                        return False
                else:
                    if d + 1 < g:
                        q.append((v, u, d + 1))
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # --- Step 1: Guaranteed valid baseline from the seed algebraic construction (LUW) ---
    # This is the LUW construction.
    q = k
    # For k=3, g=16, we need a specific parameter d.
    # The seed used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=16: g-5=11 (odd) -> d=11.
    # Let's re-verify the LUW parameters for k=3, g=16.
    # The standard LUW construction for (3, 16) might yield n=936 or similar.
    # We will use the exact logic from the seed to ensure we have a valid baseline.
    
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    baseline_n = len(order)
    baseline_edges = edges
    
    # Build adjacency for baseline
    adj_base = [[] for _ in range(baseline_n)]
    for u, v in baseline_edges:
        adj_base[u].append(v)
        adj_base[v].append(u)
        
    best_n = baseline_n
    best_edges = baseline_edges

    # --- Step 2: Structural Change - Tree Merging / Cage Completion ---
    # Strategy:
    # 1. Construct the unique depth-h non-backtracking tree from a root vertex 0.
    #    h = g // 2 - 1. For g=16, h=7.
    #    The tree has N_tree vertices.
    #    Boundary vertices are those at depth h.
    #    Each boundary vertex has exactly 1 "free" stub (the edge not connecting to its parent).
    #    Total free stubs = N_boundary.
    #    We need to pair these stubs to form a 3-regular graph.
    #    Total vertices N = N_tree.
    #    Wait, if we just pair stubs, the total number of vertices is N_tree.
    #    But the graph must be 3-regular.
    #    Internal vertices have degree 3 (1 parent + 2 children, or 1 parent + 1 child + 1 root-link? No).
    #    In a 3-regular tree:
    #    Root: 3 children.
    #    Internal (non-root, non-leaf): 1 parent, 2 children. Degree 3.
    #    Leaves (depth h): 1 parent, 0 children. Degree 1.
    #    So we have N_leaves leaves, each with 1 free stub.
    #    We must pair these N_leaves stubs.
    #    This requires N_leaves to be even.
    #    For k=3, h=7:
    #    N_leaves = 3 * 2^(h-1) = 3 * 2^6 = 192.
    #    N_tree = 1 + 3 + 3*2 + ... + 3*2^6 = 1 + 3 * (2^7 - 1) = 1 + 3*127 = 382.
    #    So we would construct a graph on 382 vertices.
    #    Is this possible?
    #    The Moore bound for (3, 16) is 512.
    #    382 < 512. So it is IMPOSSIBLE to have a 3-regular graph of girth 16 on 382 vertices.
    #    Why? Because the tree itself has "collisions" if we close the loops arbitrarily?
    #    No, the tree is a tree. The problem is that any pairing of leaves will create cycles.
    #    The shortest cycle in the resulting graph will be determined by the distance between paired leaves in the tree.
    #    Distance between two leaves u, v in the tree is dist(u, v) = 2*h - 2*LCA_depth(u, v).
    #    The cycle length is dist(u, v) + 2 (for the two stub edges)?
    #    Actually, the cycle is the path in the tree + the two stubs.
    #    Length = dist_tree(u, v) + 2.
    #    We need all cycles >= g = 16.
    #    So dist_tree(u, v) + 2 >= 16 => dist_tree(u, v) >= 14.
    #    Max distance between any two leaves is 2*h = 14.
    #    So we need dist_tree(u, v) = 14 for ALL pairs.
    #    This means LCA_depth(u, v) must be 0 (the root) for all pairs?
    #    If LCA is root, dist = 14. Cycle length = 16.
    #    But we have 192 leaves. We can only pair them such that they are in different subtrees of the root?
    #    The root has 3 children. Let the subtrees be T1, T2, T3.
    #    Leaves in T1 and T2 have LCA at root (dist 14).
    #    Leaves in T1 and T1 have LCA at depth >= 1 (dist <= 12). Cycle length <= 14 < 16. INVALID.
    #    So we cannot pair any two leaves within the same subtree T1.
    #    This implies we must pair leaves from different subtrees.
    #    But we have 192 leaves. If we distribute them roughly equally, 64 in each.
    #    We can pair T1-T2, T2-T3, T3-T1.
    #    But we have 64 leaves in T1. They must all be paired with leaves in T2 or T3.
    #    If we pair all T1 with T2, we use 64 from T2. Remaining T2: 0. Remaining T3: 64.
    #    Then we must pair T3 with... T1? But T1 is empty.
    #    So we can't just pair across subtrees easily unless the counts match.
    #    Actually, we can pair T1-T2, T2-T3, T3-T1.
    #    Let x12 be pairs between T1 and T2. x23 between T2 and T3. x31 between T3 and T1.
    #    x12 + x31 = 64 (all T1 leaves used)
    #    x12 + x23 = 64 (all T2 leaves used)
    #    x23 + x31 = 64 (all T3 leaves used)
    #    Solution: x12 = x23 = x31 = 32.
    #    So we need to pair 32 leaves from T1 with 32 from T2, etc.
    #    This is a perfect matching problem between the sets of leaves in different subtrees.
    #    Crucially, we must ensure that NO other cycles are formed.
    #    Are there other cycles?
    #    A cycle could involve multiple stubs? No, each vertex has only 1 stub.
    #    So every cycle must be a "lollipop" or a "dumbbell" formed by tree paths and stubs.
    #    Actually, since it's a tree + matching, any cycle is a simple cycle.
    #    The cycle consists of a path in the tree between two leaves u, v, and the stubs (u, u') and (v, v')?
    #    No, the stubs connect u to some u' and v to some v'.
    #    If the matching pairs u with v, the cycle is Path(u, v) + (u, v) edge? No, the edge is (u, v) in the matching.
    #    So the cycle is the path in the tree from u to v, plus the matching edge (u, v).
    #    Length = dist_tree(u, v) + 1.
    #    Wait, the stub is an edge. So if u is paired with v, we add edge (u, v).
    #    The cycle is the unique path in the tree between u and v, plus the edge (u, v).
    #    Length = dist_tree(u, v) + 1.
    #    We need dist_tree(u, v) + 1 >= 16 => dist_tree(u, v) >= 15.
    #    Max dist is 14. So dist_tree(u, v) + 1 <= 15.
    #    So it is IMPOSSIBLE to satisfy girth 16 with this simple tree-merging on 382 vertices.
    #    
    #    Wait, did I count the cycle length correctly?
    #    Tree path u -> ... -> v. Length L.
    #    Edge (u, v).
    #    Cycle length L + 1.
    #    Yes.
    #    So n=382 is impossible.
    #    
    #    What if we use a larger tree?
    #    If we go deeper, say h=8?
    #    N_leaves = 3 * 2^7 = 384.
    #    N_tree = 1 + 3 * (2^8 - 1) = 1 + 765 = 766.
    #    Max dist = 16.
    #    Cycle length = dist + 1.
    #    We need dist + 1 >= 16 => dist >= 15.
    #    So we can pair leaves with dist 15 or 16.
    #    Dist 16: LCA is root.
    #    Dist 15: LCA is depth 1.
    #    This gives us more flexibility.
    #    
    #    Let's try h=8. n=766.
    #    Baseline is 936. 766 < 936.
    #    Can we find a matching of 384 leaves such that all paired leaves have tree distance >= 15?
    #    
    #    Let's implement this search.
    
    if k == 3 and g == 16:
        h = g // 2 - 1 + 1  # h=8 for g=16
        # Build the tree
        # Node 0 is root.
        # We represent the tree by parent pointers and depth.
        # We need to generate the leaves.
        
        # Since the tree is unique for (3, 16), we can generate it.
        # But we don't need the actual vertex IDs of the final graph yet, just the tree structure.
        # We will assign IDs 0..N-1.
        
        # Generate tree nodes
        # Level 0: [0]
        # Level 1: [1, 2, 3] (children of 0)
        # Level d: children of nodes in level d-1.
        
        tree_nodes = []
        parents = {}
        depths = {}
        children = {}
        
        current_level = [0]
        next_id = 1
        depths[0] = 0
        parents[0] = -1
        children[0] = []
        
        for d in range(1, h + 1):
            next_level = []
            for node in current_level:
                # Root has 3 children. Others have 2.
                num_children = 3 if d == 1 else 2
                for _ in range(num_children):
                    new_node = next_id
                    next_id += 1
                    next_level.append(new_node)
                    parents[new_node] = node
                    depths[new_node] = d
                    children[node].append(new_node)
            
            current_level = next_level
        
        n_tree = next_id
        leaves = current_level
        
        # Check if n_tree is even?
        # n_tree = 1 + 3 * (2^h - 1).
        # For h=8, n_tree = 766. Even.
        # For h=7, n_tree = 382. Even.
        
        # We need to pair leaves.
        # Constraint: dist_tree(u, v) >= g - 1 = 15.
        
        # Precompute distances?
        # For 384 leaves, we can compute pairwise distances.
        # 384^2 / 2 ~ 70k pairs. Fast.
        
        # LCA computation
        # Since it's a tree, we can lift nodes.
        # Or just use parent pointers and depth.
        
        def get_lca(u, v):
            du = depths[u]
            dv = depths[v]
            if du > dv:
                u, v = v, u
                du, dv = dv, du
            # Lift u up to same depth
            while du < dv:
                v = parents[v]
                dv -= 1
            while u != v:
                u = parents[u]
                v = parents[v]
            return u
        
        def get_dist(u, v):
            lca = get_lca(u, v)
            return depths[u] + depths[v] - 2 * depths[lca]
        
        # Build compatibility graph for leaves
        # Edge between leaf i and leaf j if dist(i, j) >= 15.
        
        # We need a perfect matching in this compatibility graph.
        # Hopcroft-Karp or simple DFS matching?
        # 384 nodes. Simple DFS matching (Hungarian) might be O(VE) = 384 * (384*384) ~ 50M.
        # Might be slow in Python.
        # Let's try a greedy + backtracking approach or just a simple matching algorithm.
        
        # Actually, let's just try to find ANY valid matching.
        # We can use a simple augmenting path algorithm.
        
        # Build adjacency list for compatibility
        comp_adj = {i: [] for i in leaves}
        leaf_index = {leaf: i for i, leaf in enumerate(leaves)}
        
        for i in range(len(leaves)):
            u = leaves[i]
            for j in range(i + 1, len(leaves)):
                v = leaves[j]
                d = get_dist(u, v)
                if d >= g - 1:
                    comp_adj[i].append(j)
                    comp_adj[j].append(i)
        
        # Check if a perfect matching exists.
        # If not, this h is impossible.
        
        # Simple DFS matching
        match_u = [-1] * len(leaves)
        match_v = [-1] * len(leaves)
        
        def dfs(u, visited):
            for v in comp_adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_v[v] == -1 or dfs(match_v[v], visited):
                    match_u[u] = v
                    match_v[v] = u
                    return True
            return False
        
        found = 0
        for u in range(len(leaves)):
            visited = [False] * len(leaves)
            if dfs(u, visited):
                found += 1
        
        if found == len(leaves):
            # We have a perfect matching!
            # Construct the graph.
            # Vertices 0..n_tree-1.
            # Edges:
            # 1. Tree edges.
            # 2. Matching edges.
            
            edges_new = []
            # Tree edges
            for node in range(1, n_tree):
                p = parents[node]
                edges_new.append((p, node))
            
            # Matching edges
            for u in range(len(leaves)):
                v = match_u[u]
                if v > u: # Avoid duplicates
                    lu = leaves[u]
                    lv = leaves[v]
                    edges_new.append((lu, lv))
            
            # Verify girth
            adj_new = [[] for _ in range(n_tree)]
            for u, v in edges_new:
                adj_new[u].append(v)
                adj_new[v].append(u)
                
            if _verify_girth(adj_new, n_tree, g):
                if n_tree < best_n:
                    best_n = n_tree
                    best_edges = edges_new
                    # We found a smaller graph!
                    # We can stop here or try larger h?
                    # Larger h means larger n. We want smaller n.
                    # So h=8 is the smallest h that might work?
                    # h=7 was impossible.
                    # So h=8 is the best we can get with this strategy.
                    pass
        
    # If no better graph found, return baseline
    return best_n, best_edges
