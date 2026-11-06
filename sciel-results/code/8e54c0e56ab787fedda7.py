import time
import collections
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_distances(start, adj, blocked=None):
    """Returns (distances, parents) from start, avoiding blocked nodes."""
    if blocked is None:
        blocked = set()
    dist = {start: 0}
    parent = {start: -1}
    queue = collections.deque([start])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in dist and v not in blocked:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    return dist, parent

def _reconstruct_path(parent, start, end):
    path = []
    cur = end
    while cur != -1:
        path.append(cur)
        if cur == start:
            break
        cur = parent[cur]
    path.reverse()
    return path

def _tree_diameter(adj, nodes):
    """Find diameter of the tree induced by 'nodes' (which must form a tree)."""
    if not nodes:
        return []
    nodes_set = set(nodes)
    # Pick an arbitrary node
    start = next(iter(nodes_set))
    # BFS to find farthest node
    dist1, par1 = _bfs_distances(start, adj, blocked=set())
    # Restrict to nodes in component
    far1 = max(dist1, key=lambda x: dist1[x] if x in nodes_set else -1)
    # BFS from far1
    dist2, par2 = _bfs_distances(far1, adj, blocked=set())
    far2 = max(dist2, key=lambda x: dist2[x] if x in nodes_set else -1)
    # Reconstruct path
    path = _reconstruct_path(par2, far1, far2)
    # Filter to ensure only nodes in component (should be automatic if tree is isolated)
    return [n for n in path if n in nodes_set]

def _find_cycle(adj, nodes):
    """Find a cycle in the component 'nodes'. Returns list of nodes in cycle in order, or None."""
    nodes_set = set(nodes)
    # Use DFS to find back edge
    visited = set()
    parent = {}
    cycle_nodes = []
    
    def dfs(u, p):
        visited.add(u)
        parent[u] = p
        for v in adj[u]:
            if v not in nodes_set:
                continue
            if v not in visited:
                if dfs(v, u):
                    return True
            elif v != p:
                # Found back edge u-v
                # Reconstruct cycle
                cycle = [u]
                cur = u
                while cur != v:
                    cur = parent[cur]
                    cycle.append(cur)
                cycle_nodes.extend(cycle)
                return True
        return False
    
    for node in nodes:
        if node not in visited:
            if dfs(node, -1):
                break
    
    if not cycle_nodes:
        return None
    return cycle_nodes

def _longest_path_in_component(adj, nodes):
    """Find longest simple path in the component 'nodes'."""
    n_comp = len(nodes)
    if n_comp == 0:
        return []
    if n_comp == 1:
        return [nodes[0]]
    
    # Count edges in component
    e_count = 0
    nodes_set = set(nodes)
    for u in nodes:
        for v in adj[u]:
            if v in nodes_set and u < v:
                e_count += 1
                
    if e_count == n_comp - 1:
        # It's a tree
        return _tree_diameter(adj, nodes)
    elif e_count == n_comp:
        # It's unicyclic
        cycle = _find_cycle(adj, nodes)
        if not cycle:
            # Should not happen if e_count == n_comp, but fallback to tree diameter just in case
            return _tree_diameter(adj, nodes)
        
        cycle_set = set(cycle)
        
        # For each node in cycle, find the max depth of the tree attached to it
        # The "tree attached" is the component of G - (cycle edges) containing the cycle node
        # Actually, removing cycle edges turns the unicyclic graph into a tree.
        # Let's construct the tree by removing one edge from the cycle?
        # No, better to treat each cycle node as a root of a "branch" tree.
        
        # We need the height of each cycle node.
        # Height of u = max distance from u to any node in its branch (excluding other cycle nodes).
        
        # To find branches, we can do BFS from each cycle node, but not traversing cycle edges.
        # Cycle edges are (cycle[i], cycle[(i+1)%L]) and (cycle[i], cycle[(i-1)%L]).
        
        L = len(cycle)
        heights = {}
        # We also need to reconstruct the path for the best candidate
        # Let's store the farthest node in each branch and the path to it
        
        farthest_node = {}
        farthest_path = {}
        
        for i, u in enumerate(cycle):
            # Find neighbors of u that are NOT in the cycle
            # Or rather, we want to traverse the "tree part" attached to u.
            # We can do BFS from u, but we must not enter other cycle nodes.
            # So blocked = cycle_set - {u}
            blocked = cycle_set - {u}
            dist, par = _bfs_distances(u, adj, blocked=blocked)
            # Find farthest node in this BFS tree
            if not dist:
                heights[u] = 0
                farthest_node[u] = u
                farthest_path[u] = [u]
            else:
                # dist contains u and all nodes in the branch
                # We want the node with max distance
                far_v = max(dist.keys(), key=lambda x: dist[x])
                heights[u] = dist[far_v]
                farthest_node[u] = far_v
                # Reconstruct path from u to far_v
                path = _reconstruct_path(par, u, far_v)
                farthest_path[u] = path
        
        # Now, the longest path is either:
        # 1. Within a single branch (diameter of that branch tree).
        #    Wait, the diameter of the branch tree rooted at u is not necessarily 2*height[u].
        #    It could be between two nodes in the same branch.
        #    However, any path within a branch does not use the cycle.
        #    The diameter of the whole tree (if we cut the cycle) is the max of:
        #    - Diameters of individual branches
        #    - Paths going through the cycle
        
        # Let's calculate the diameter of each branch tree.
        # The branch tree is the subgraph induced by the nodes reachable from u without going through other cycle nodes.
        # We can get these nodes from the BFS 'dist' dict.
        
        best_path = []
        
        for i, u in enumerate(cycle):
            # Get nodes in this branch
            blocked = cycle_set - {u}
            dist_u, _ = _bfs_distances(u, adj, blocked=blocked)
            branch_nodes = list(dist_u.keys())
            if len(branch_nodes) > 1:
                # Find diameter of this branch tree
                # It's a tree, so we can use _tree_diameter
                # But _tree_diameter expects the adj to be restricted?
                # My _tree_diameter uses BFS on the full adj but restricts to 'nodes' set.
                # So we can pass branch_nodes.
                d_path = _tree_diameter(adj, branch_nodes)
                if len(d_path) > len(best_path):
                    best_path = d_path
            
            # Now consider paths going through the cycle
            # A path goes from a node in branch i to a node in branch j, traversing the cycle between i and j.
            # Or from branch i to branch i (using the cycle loop? No, simple path).
            # If it starts and ends in the same branch, it doesn't use the cycle, covered above.
            # If it uses the cycle, it must enter at i and exit at j (i != j).
            
            for j in range(i+1, L):
                v = cycle[j]
                # Two directions on the cycle: i->j and j->i (which is i->...->j the other way)
                # Path 1: branch_i -> i -> ... -> j -> branch_j
                # Length = height[i] + dist_cycle(i, j) + height[j]
                # dist_cycle(i, j) is the number of edges on the cycle between i and j.
                
                # Distance along cycle from i to j (clockwise)
                dist_cw = (j - i) % L
                # Distance along cycle from i to j (counter-clockwise)
                dist_ccw = L - dist_cw
                
                # We can traverse either way. The path is simple, so we can't traverse the whole cycle and come back.
                # The path on the cycle is a simple path between i and j.
                # So we can use min(dist_cw, dist_ccw)?
                # No, the path is determined by the sequence of vertices.
                # If we go i->...->j (cw), the vertices are cycle[i], cycle[i+1], ..., cycle[j].
                # If we go i->...->j (ccw), the vertices are cycle[i], cycle[i-1], ..., cycle[j].
                # Both are valid simple paths on the cycle.
                # We should check both?
                # Actually, the length of the path on the cycle is the number of edges.
                # Let's calculate the actual path for both directions and pick the longer one?
                # No, we want the LONGEST simple path.
                # So we should take the direction that gives the LONGER path on the cycle?
                # Wait. The total path is Branch_i + Cycle_Path + Branch_j.
                # To maximize length, we want the longest Cycle_Path between i and j.
                # So we use max(dist_cw, dist_ccw).
                
                cycle_len = max(dist_cw, dist_ccw)
                
                # Reconstruct the cycle path
                if dist_cw >= dist_ccw:
                    # Go clockwise
                    cycle_path = []
                    idx = i
                    for _ in range(dist_cw + 1):
                        cycle_path.append(cycle[idx])
                        idx = (idx + 1) % L
                else:
                    # Go counter-clockwise
                    cycle_path = []
                    idx = i
                    for _ in range(dist_ccw + 1):
                        cycle_path.append(cycle[idx])
                        idx = (idx - 1) % L
                
                # Combine:
                # Path from farthest_node[i] to i (reversed to go from i to farthest? No, farthest_path[i] is u->far)
                # We want path: farthest_node[i] -> ... -> i -> ... -> j -> ... -> farthest_node[j]
                # farthest_path[i] is [i, ..., far_i]. We need [far_i, ..., i]. So reverse it.
                part_i = list(reversed(farthest_path[i]))
                part_j = farthest_path[j]
                
                # part_i ends with i. cycle_path starts with i.
                # So we can merge: part_i[:-1] + cycle_path + part_j[1:]
                # Wait, part_i is [far_i, ..., i]. cycle_path is [i, ..., j].
                # So merged = part_i + cycle_path[1:] + part_j[1:]
                
                candidate = part_i + cycle_path[1:] + part_j[1:]
                
                # Check for duplicates? Should be fine if branches are disjoint.
                # Branches are disjoint by construction (blocked cycle nodes).
                # Cycle path is simple.
                
                if len(candidate) > len(best_path):
                    best_path = candidate
                    
        return best_path
    else:
        # More than one cycle. Fallback to heuristic DFS or just tree diameter of a spanning tree?
        # For this specific graph, this case is unlikely.
        # Let's just find a spanning tree and compute its diameter as a lower bound,
        # or use a simple DFS heuristic.
        # Given the low edge density, it's safer to assume it's a tree or unicyclic.
        # If it's more complex, the longest path is likely close to the diameter of a spanning tree.
        # Let's just return the tree diameter of the component (ignoring extra edges).
        return _tree_diameter(adj, nodes)

def construct():
    n = P['n']
    adj = _build_adj(n, P['edges'])
    
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            # BFS to find component
            comp = []
            queue = collections.deque([i])
            visited[i] = True
            while queue:
                u = queue.popleft()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            components.append(comp)
    
    best_path = []
    for comp in components:
        path = _longest_path_in_component(adj, comp)
        if len(path) > len(best_path):
            best_path = path
            
    return best_path
# EVOLVE-BLOCK-END
