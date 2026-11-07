"""CSP solver with forward checking, MRV, and incremental union-find for bicolored forest constraint."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _validate(coloring, edges):
    """Check adjacency constraint and bicolored forest constraint via union-find per pair."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu < cv:
            pair = (cu, cv)
        else:
            pair = (cv, cu)
        pair_edges[pair].append((u, v))
    
    for pair, elist in pair_edges.items():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True


def _csp_solve(k, n, edges, adj, deadline):
    """
    CSP backtracking with forward checking and MRV.
    Domains: each vertex can take colors 0..k-1.
    Constraints: 
      1. Adjacent vertices must have different colors.
      2. For any pair of colors (c1,c2), the subgraph induced by vertices colored c1 or c2 must be a forest.
    
    We use a union-find structure that tracks, for each color pair, the connectivity among vertices.
    When we assign a color to a vertex, we check if it creates a cycle in any bicolored subgraph.
    """
    # Precompute neighbor lists
    # State: assignment[v] = color or -1
    assignment = [-1] * n
    
    # For each color pair, maintain a union-find over vertices that have been assigned that pair
    # pair_uf[(c1,c2)] = dict mapping vertex -> parent
    # We'll use a simpler approach: for each pair, keep a list of edges and check connectivity
    
    # Actually, let's use a more efficient incremental approach:
    # For each pair (c1, c2), maintain a DSU. When we assign vertex v to color c, 
    # for each neighbor u already assigned to color d != c, the pair (min(c,d), max(c,d)) gets edge (u,v).
    # If adding this edge creates a cycle in the DSU for that pair, the assignment is invalid.
    
    # DSU with rollback for efficiency
    # We'll track changes for backtracking
    
    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.history = []  # stack of (x, old_parent, old_rank_y, y) for rollback
        
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False  # would create cycle
            # Union by rank
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.history.append((rx, ry, self.parent[ry], self.rank[rx]))
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        
        def rollback(self, count):
            for _ in range(count):
                rx, ry, old_parent_ry, old_rank_rx = self.history.pop()
                self.parent[ry] = old_parent_ry
                self.rank[rx] = old_rank_rx
    
    # For each color pair, we need a DSU. But we only need DSUs for pairs that actually have edges.
    # Let's use a dict: pair -> DSU instance
    # But DSU needs to be over all n vertices, which is expensive. 
    # Alternative: for each pair, just track the set of vertices involved and use a simple DSU on those.
    
    # Simpler approach: for each pair (c1,c2), maintain a set of edges. To check for cycles,
    # we can use DFS/BFS on the subgraph. But this is O(E) per check.
    
    # Let's use the incremental DSU approach with a global vertex index.
    # pair_dsu[(c1,c2)] = (dsu, num_unions) where dsu is a dict-based DSU
    
    pair_dsu = {}  # (c1,c2) -> {'parent': {}, 'changes': []}
    
    def dsu_find(parent, x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def dsu_union(parent, x, y):
        rx = dsu_find(parent, x)
        ry = dsu_find(parent, y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True
    
    # changes stack for backtracking: list of (pair, x, y, was_valid)
    # Actually, we need to undo unions. Let's track the parent changes.
    
    # For simplicity and correctness, let's use a different approach:
    # Maintain for each pair a list of edges. When checking if an assignment creates a cycle,
    # we do a BFS/DFS on the subgraph of that pair.
    
    # pair_edges[(c1,c2)] = set of (u,v) edges in the bicolored subgraph
    pair_edges = defaultdict(set)
    
    # For efficient cycle checking, we'll use the fact that a graph is a forest iff |E| <= |V| - components
    # But checking components requires BFS. Let's just do union-find on the fly for the check.
    
    def check_pair_forest(c1, c2):
        """Check if the subgraph induced by colors c1 and c2 is a forest."""
        # Collect all vertices with color c1 or c2
        vertices = set()
        for v in range(n):
            if assignment[v] in (c1, c2):
                vertices.add(v)
        
        # Collect edges between these vertices
        elist = []
        for u in vertices:
            for v in adj[u]:
                if v > u and assignment[v] in (c1, c2):
                    elist.append((u, v))
        
        # Union-find
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        
        for u, v in elist:
            if not union(u, v):
                return False
        return True
    
    def try_assign(v, c):
        """Try to assign color c to vertex v. Returns True if valid."""
        # Check adjacency
        for nb in adj[v]:
            if assignment[nb] == c:
                return False
        
        # Check bicolored forest constraints
        # For each neighbor with a different color, check the pair
        pairs_to_check = set()
        for nb in adj[v]:
            if assignment[nb] != -1 and assignment[nb] != c:
                c1, c2 = (min(c, assignment[nb]), max(c, assignment[nb]))
                pairs_to_check.add((c1, c2))
        
        # Temporarily assign
        assignment[v] = c
        
        for (c1, c2) in pairs_to_check:
            if not check_pair_forest(c1, c2):
                assignment[v] = -1
                return False
        
        return True
    
    def unassign(v):
        assignment[v] = -1
    
    # MRV: pick unassigned vertex with fewest available colors
    def get_mrv_vertex():
        best_v = -1
        best_count = k + 1
        for v in range(n):
            if assignment[v] != -1:
                continue
            # Count available colors
            available = set(range(k))
            for nb in adj[v]:
                if assignment[nb] != -1:
                    available.discard(assignment[nb])
            if len(available) < best_count:
                best_count = len(available)
                best_v = v
                if best_count == 0:
                    return best_v, available
        return best_v, available if best_v != -1 else set()
    
    # Backtracking
    nodes_explored = 0
    
    def backtrack():
        nonlocal nodes_explored
        if time.time() > deadline:
            return False
        nodes_explored += 1
        
        # Check if all assigned
        for v in range(n):
            if assignment[v] == -1:
                break
        else:
            return True
        
        # MRV
        v, available = get_mrv_vertex()
        if v == -1:
            return True
        if len(available) == 0:
            return False
        
        # Try each available color
        for c in sorted(available):
            if try_assign(v, c):
                if backtrack():
                    return True
                unassign(v)
        
        return False
    
    if backtrack():
        return assignment[:]
    return None


def _csp_solve_fast(k, n, edges, adj, deadline):
    """
    Optimized CSP solver with incremental cycle checking.
    Key insight: when assigning vertex v to color c, we only need to check pairs (c, d) 
    where d is the color of a neighbor. For each such pair, we check if adding the edges 
    from v to its neighbors of color d creates a cycle.
    """
    assignment = [-1] * n
    
    # For each color pair, maintain a DSU with rollback
    # pair_state[(c1,c2)] = (parent_dict, changes_list)
    # But this is complex. Let's use a simpler incremental check.
    
    # For each pair (c1, c2), maintain:
    #   - a set of vertices with color c1 or c2
    #   - a DSU over those vertices
    # When we assign v to color c, for each neighbor u with color d != c:
    #   pair = (min(c,d), max(c,d))
    #   Add edge (u,v) to the pair's DSU. If cycle, backtrack.
    
    # We'll use a global DSU per pair with explicit rollback.
    
    # pair_data[(c1,c2)] = {'parent': {}, 'changes': []}
    # changes: list of (x, old_parent) for path compression, and (child, parent) for union
    
    # Actually, path compression makes rollback hard. Let's use union by rank without path compression.
    
    pair_data = {}
    
    def get_pair_dsu(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        if (c1, c2) not in pair_data:
            pair_data[(c1, c2)] = {'parent': {}, 'rank': {}}
        return pair_data[(c1, c2)]
    
    def dsu_find(dsu, x):
        # No path compression for easy rollback
        while dsu['parent'].get(x, x) != x:
            x = dsu['parent'][x]
        return x
    
    def dsu_union(dsu, x, y):
        rx = dsu_find(dsu, x)
        ry = dsu_find(dsu, y)
        if rx == ry:
            return False  # cycle
        # Union by rank
        if dsu['rank'].get(rx, 0) < dsu['rank'].get(ry, 0):
            rx, ry = ry, rx
        # Record change
        dsu['changes'] = dsu.get('changes', [])
        dsu['changes'].append((ry, rx, dsu['parent'].get(ry, ry), dsu['rank'].get(rx, 0)))
        dsu['parent'][ry] = rx
        new_rank = max(dsu['rank'].get(rx, 0), dsu['rank'].get(ry, 0))
        if dsu['rank'].get(rx, 0) == dsu['rank'].get(ry, 0):
            new_rank += 1
        dsu['rank'][rx] = new_rank
        return True
    
    def dsu_rollback(dsu, num_changes):
        for _ in range(num_changes):
            if not dsu.get('changes'):
                break
            ry, rx, old_parent_ry, old_rank_rx = dsu['changes'].pop()
            dsu['parent'][ry] = old_parent_ry
            dsu['rank'][rx] = old_rank_rx
    
    # Track total changes for rollback
    total_changes = 0
    change_log = []  # list of (pair, num_changes)
    
    def assign(v, c):
        """Assign color c to vertex v. Returns list of (pair, changes) for rollback."""
        nonlocal total_changes
        # Check adjacency
        for nb in adj[v]:
            if assignment[nb] == c:
                return None
        
        assignment[v] = c
        changes = []
        
        for nb in adj[v]:
            if assignment[nb] != -1 and assignment[nb] != c:
                d = assignment[nb]
                c1, c2 = (min(c, d), max(c, d))
                dsu = get_pair_dsu(c1, c2)
                if 'changes' not in dsu:
                    dsu['changes'] = []
                # Initialize vertices if not present
                if v not in dsu['parent']:
                    dsu['parent'][v] = v
                    dsu['rank'][v] = 0
                if nb not in dsu['parent']:
                    dsu['parent'][nb] = nb
                    dsu['rank'][nb] = 0
                
                before = len(dsu['changes'])
                if not dsu_union(dsu, v, nb):
                    # Cycle detected, rollback
                    dsu_rollback(dsu, len(dsu['changes']) - before)
                    assignment[v] = -1
                    # Rollback previous changes in this call
                    for (p, nc) in reversed(changes):
                        dsu_rollback(pair_data[p], nc)
                    return None
                changes.append(((c1, c2), len(dsu['changes']) - before))
        
        return changes
    
    def unassign(v, changes):
        """Undo assignment of v, rolling back the DSU changes."""
        for (pair, nc) in reversed(changes):
            dsu_rollback(pair_data[pair], nc)
        assignment[v] = -1
    
    # MRV
    def get_mrv():
        best_v = -1
        best_count = k + 1
        best_avail = None
        for v in range(n):
            if assignment[v] != -1:
                continue
            forbidden = set()
            for nb in adj[v]:
                if assignment[nb] != -1:
                    forbidden.add(assignment[nb])
            avail = [c for c in range(k) if c not in forbidden]
            if len(avail) < best_count:
                best_count = len(avail)
                best_v = v
                best_avail = avail
                if best_count == 0:
                    return best_v, best_avail
        return best_v, best_avail
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        v, avail = get_mrv()
        if v == -1:
            return True
        if not avail:
            return False
        
        for c in avail:
            changes = assign(v, c)
            if changes is not None:
                if backtrack():
                    return True
                unassign(v, changes)
        
        return False
    
    if backtrack():
        return assignment[:]
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    # Try k=4 first (theoretical minimum for this type of constraint)
    for k in [4, 5, 6, 7]:
        if time.time() > deadline - 5:
            break
        result = _csp_solve_fast(k, n, edges, adj, deadline)
        if result is not None and _validate(result, edges):
            return [int(c) for c in result]
    
    # Fallback: proper coloring with n colors
    return [int(i) for i in range(n)]
