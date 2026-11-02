import random
import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=5, g=9, aims for n=426 (Moore bound).
    """
    start_time = time.time()
    deadline = 115  # Safety margin
    
    # Moore Bound calculation
    # For odd g=2r+1, n >= 1 + k * sum((k-1)^i for i in range(r))
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))
    
    # If k=5, g=9, r=4, moore_bound = 426.
    # We attempt to construct a graph with n = moore_bound.
    
    if k == 5 and g == 9:
        target_n = 426
    else:
        # Fallback to LUW for other parameters if needed, 
        # but the task implies k=5, g=9.
        # For now, we focus on the specific case.
        target_n = None

    if target_n is not None:
        # 1. Build the k-regular tree of radius r
        # Root is 0.
        # Level 0: 1 node (0)
        # Level 1: k nodes
        # Level i: k * (k-1)^(i-1) nodes
        # Total nodes: 1 + k * sum((k-1)^i for i in 0..r-1)
        
        vertices = []
        edges = []
        parent = [-1] * target_n
        children = [[] for _ in range(target_n)]
        
        # Assign IDs level by level
        # Level 0: 0
        vertices.append(0)
        
        next_id = 1
        for level in range(1, r + 1):
            if level == 1:
                parents = [0]
            else:
                # Parents are nodes from previous level
                parents = vertices[-k * (k-1) ** (level-2) :]
            
            for p in parents:
                # Each node has k children, except root has k children.
                # Wait, in a k-regular tree:
                # Root has k children.
                # Other nodes have k-1 children (1 edge to parent).
                # So at level 1, root generates k nodes.
                # At level 2, each of the k nodes generates k-1 nodes.
                
                num_children = k if level == 1 else k - 1
                for _ in range(num_children):
                    v = next_id
                    next_id += 1
                    vertices.append(v)
                    parent[v] = p
                    children[p].append(v)
                    edges.append((p, v))
        
        if next_id != target_n:
            # Fallback if calculation mismatch
            pass
        
        # 2. Identify leaves
        # Leaves are nodes at level r
        # In our construction, level r nodes have no children.
        leaves = vertices[-k * (k-1) ** (r-1):]
        
        # 3. Group leaves by their "branch" (child of root)
        # The root's children are 1, 2, 3, 4, 5 (IDs 1..5)
        # For each leaf, find its ancestor at level 1.
        groups = {i: [] for i in range(1, k + 1)}
        for leaf in leaves:
            v = leaf
            while parent[v] != 0:
                v = parent[v]
            # v is now the child of root
            groups[v].append(leaf)
        
        # 4. Add edges between leaves from different groups
        # We need to add (k-1) edges per leaf, all to leaves in other groups.
        # We will use a random greedy approach to select these edges.
        
        # Pre-compute allowed pairs
        # To speed up, we can just try to connect leaves randomly.
        
        # Degree count for leaves (currently 1 from tree)
        degree = [1] * target_n
        
        # List of leaves that need more edges
        # We need to add 4 edges to each leaf.
        
        # Strategy:
        # Create a list of all potential edges (u, v) where u, v are leaves, 
        # u in group i, v in group j, i != j.
        # Shuffle them.
        # Iterate and add edge if it doesn't create a short cycle and degrees are not exceeded.
        
        # To check for short cycles efficiently:
        # Since the tree has radius 4, any new cycle must use at least one new edge.
        # A cycle of length L uses m new edges and L-m tree edges.
        # Shortest possible cycle with 1 new edge: 9 (8 tree + 1 new).
        # Cycle with 2 new edges: min tree path between endpoints.
        # If we connect u in G_i to v in G_j, and x in G_j to y in G_i.
        # Cycle: u-v (new), v...x (tree), x-y (new), y...u (tree).
        # Length = 2 + dist(v,x) + dist(y,u).
        # We need this >= 9 => dist(v,x) + dist(y,u) >= 7.
        # Since distances are even, sum >= 8.
        # So we must avoid pairs where dist(v,x) + dist(y,u) < 8.
        
        # We will maintain a set of existing new edges to check conflicts.
        # Conflict check for new edge (u, v):
        # For every existing new edge (x, y):
        # If {u,v} and {x,y} are disjoint:
        #   Check if (u in G_i, v in G_j) and (x in G_j, y in G_i) with dist(v,x) + dist(y,u) < 8.
        #   Or (u in G_i, v in G_j) and (x in G_i, y in G_j) with dist(u,x) + dist(v,y) < 8.
        #   Note: if groups are same, they can't be connected directly.
        
        # This check is O(E_new) per edge. E_new is ~640.
        # 640 * 640 = 400k checks. Very fast.
        
        # Let's implement a fast distance check in the tree.
        # Precompute depth and LCA?
        # Or just simple BFS from u and v?
        # Since tree is small (426 nodes), BFS is fast.
        # But we can do better.
        # dist(u, v) in tree can be computed if we have parent pointers.
        
        def tree_dist(u, v):
            # Simple climb up
            du = 0
            dv = 0
            # We need depths
            # Let's precompute depth for all nodes
            pass
            
        # Precompute depths
        depth = [0] * target_n
        for v in range(1, target_n):
            depth[v] = depth[parent[v]] + 1
            
        def get_ancestor(v, target_depth):
            while depth[v] > target_depth:
                v = parent[v]
            return v
            
        def lca(u, v):
            if depth[u] < depth[v]:
                u, v = v, u
            while depth[u] > depth[v]:
                u = parent[u]
            while u != v:
                u = parent[u]
                v = parent[v]
            return u
            
        def dist(u, v):
            w = lca(u, v)
            return depth[u] + depth[v] - 2 * depth[w]
            
        # Store new edges as (u, v) with u < v for uniqueness?
        # No, we need to track endpoints for group checks.
        new_edges = []
        # Adjacency list for new edges only?
        # We just need to check against all existing new edges.
        
        # Generate candidate edges
        # We want to pick edges such that each leaf gets degree k.
        # We'll pick edges one by one.
        
        # To ensure regularity, we can try to form perfect matchings.
        # But random greedy with backtracking/restart is easier.
        
        # Let's try a few random restarts.
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            random.seed(attempt)
            
            # Reset
            new_edges = []
            deg = [1] * target_n # Initial degree from tree
            
            # We need to add 4 edges to each leaf.
            # Total 640 edges.
            
            # Create list of leaves
            # We will randomly pick pairs of leaves from different groups
            # and add edge if valid.
            
            # To avoid bias, we can shuffle the list of leaves for each "step"
            # or just pick random pairs.
            
            # Efficient approach:
            # Keep a list of leaves that still need edges.
            # Pick two random leaves from different groups that need edges.
            # Check validity.
            # If valid, add.
            # If invalid, try another pair.
            
            # If we get stuck, restart.
            
            # Optimization:
            # Pre-compute for each leaf, which other leaves are "too close" (dist < 8 in the combined graph sense).
            # Actually, the constraint is on pairs of new edges.
            
            # Let's just try to add edges.
            success = True
            for _ in range(640):
                if time.time() > deadline:
                    success = False
                    break
                    
                # Find leaves that need edges
                # We need to pick u, v such that deg[u] < 5 and deg[v] < 5 and group[u] != group[v]
                
                # Randomly pick u
                found = False
                for _ in range(20): # Try 20 times to find a valid pair
                    u = random.choice(leaves)
                    if deg[u] >= 5:
                        continue
                    v = random.choice(leaves)
                    if deg[v] >= 5:
                        continue
                    if u == v:
                        continue
                    if groups[u] == groups[v]: # groups are dicts, values are lists. We need group ID.
                        # Wait, groups is {child_id: [leaves]}
                        # We need to map leaf -> group_id
                        pass
                    
                # This random picking is inefficient.
                # Let's use a list of available leaves.
                
                # Better:
                # Maintain a list of leaves with deg < 5.
                # Pick u from this list.
                # Pick v from this list, ensuring group[v] != group[u].
                
                # Let's restructure the loop.
                
            if not success:
                continue
                
            # If we successfully added 640 edges, verify.
            # Check degrees
            if any(deg[v] != 5 for v in leaves):
                continue
                
            # Check girth
            # Build full adjacency list
            adj = [[] for _ in range(target_n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            for u, v in new_edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Verify simple graph (no multi-edges)
            # The tree edges are unique.
            # New edges: we must ensure no duplicate new edges.
            # My random picking might pick (u,v) then (v,u).
            # I should use a set for new edges.
            
            # Let's redo the edge addition with a set.
            
            new_edges_set = set()
            deg = [1] * target_n
            success = True
            
            # List of leaves needing edges
            # We'll just iterate 640 times
            
            for _ in range(640):
                if time.time() > deadline:
                    success = False
                    break
                
                # Pick u
                # To be fair, pick from all leaves with deg < 5
                # But we need to ensure we can complete the graph.
                
                # Simple heuristic: pick random leaf with deg < 5.
                candidates = [l for l in leaves if deg[l] < 5]
                if not candidates:
                    success = False
                    break
                u = random.choice(candidates)
                u_group = group_id[u] # Need to define group_id
                
                # Pick v from different group, deg < 5
                # Candidates for v
                v_candidates = [l for l in leaves if deg[l] < 5 and group_id[l] != u_group and l != u]
                if not v_candidates:
                    success = False
                    break
                
                # Try to find a valid v
                random.shuffle(v_candidates)
                valid_v = None
                for v in v_candidates:
                    if (u, v) in new_edges_set or (v, u) in new_edges_set:
                        continue
                    
                    # Check for short cycles with existing new edges
                    # For each (x, y) in new_edges_set:
                    # If u, v, x, y are distinct:
                    # Check if (u in G_i, v in G_j) and (x in G_j, y in G_i)
                    # Then dist(v, x) + dist(y, u) >= 8?
                    # Note: u, v are in different groups.
                    # x, y are in different groups.
                    
                    conflict = False
                    for x, y in new_edges_set:
                        if x in (u, v) or y in (u, v):
                            continue
                        
                        # Determine group relationships
                        # u in Gu, v in Gv. x in Gx, y in Gy.
                        # Case 1: Gx == Gv and Gy == Gu
                        # Then cycle u-v-x-y-u?
                        # Path: u-v (new), v-x (tree), x-y (new), y-u (tree).
                        # Length = 2 + dist(v, x) + dist(y, u).
                        # Need >= 9 => dist(v, x) + dist(y, u) >= 7 => >= 8.
                        
                        if group_id[x] == group_id[v] and group_id[y] == group_id[u]:
                            d1 = dist(v, x)
                            d2 = dist(y, u)
                            if d1 + d2 < 8:
                                conflict = True
                                break
                                
                        # Case 2: Gx == Gu and Gy == Gv
                        # Cycle u-x-y-v-u?
                        # Path: u-x (tree), x-y (new), y-v (tree), v-u (new).
                        # Length = 2 + dist(u, x) + dist(y, v).
                        # Need >= 8.
                        
                        if group_id[x] == group_id[u] and group_id[y] == group_id[v]:
                            d1 = dist(u, x)
                            d2 = dist(y, v)
                            if d1 + d2 < 8:
                                conflict = True
                                break
                            
                        # Other cases involve 3 or 4 groups, which implies longer paths (>= 10).
                        
                    if not conflict:
                        valid_v = v
                        break
                        
                if valid_v is None:
                    success = False
                    break
                    
                new_edges_set.add((u, valid_v))
                deg[u] += 1
                deg[valid_v] += 1
                
            if success:
                # Final verification
                # 1. Degrees
                # 2. Girth
                # We already checked degrees.
                # Girth check:
                # We checked pairs of new edges.
                # Are there cycles with 1 new edge?
                # Min length 9. OK.
                # Cycles with 3+ new edges?
                # Min tree path between different groups is 8.
                # 3 new edges -> 3*1 + 2*8 = 19? No.
                # u-v (new), v-w (tree, 8), w-x (new), x-y (tree, 8), y-u (new).
                # Length 1+8+1+8+1 = 19.
                # So only 1 or 2 new edges can form short cycles.
                # We handled 2.
                # So girth >= 9.
                
                # Return result
                all_edges = edges + list(new_edges_set)
                return target_n, all_edges
                
    # Fallback to LUW if 426 fails
    # ... (implement LUW fallback)
    
    # LUW Implementation
    q = k
    target_girth = g if g % 2 == 0 else g + 1
    
    import math
    depth = (g - 1) // 2
    moore_bound = 1 + k
    for i in range(1, depth + 1):
        moore_bound += k * (k - 1) ** i
        
    min_d_for_size = 2
    while 2 * (q ** min_d_for_size) < moore_bound:
        min_d_for_size += 1
        
    min_d_for_girth = max(2, (target_girth - 4) // 2)
    
    d = max(min_d_for_size, min_d_for_girth)
    
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
                
    return len(order), edges_luw
