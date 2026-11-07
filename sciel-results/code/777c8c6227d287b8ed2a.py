"""Combined approach: incremental union-find backtracking + local search color elimination.
Exploit graph structure for better ordering and constraint propagation."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [[] for _ in range(N)]
adj_set = [set() for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)
    adj_set[u].add(v)
    adj_set[v].add(u)


def _check_coloring(coloring, edges, adj):
    """Validate: adjacent differ, and each color-pair subgraph is a forest."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Union-find per color pair
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
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


class UnionFind:
    __slots__ = ('parent', 'rank', 'n')
    def __init__(self, n=0):
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        r = self.rank
        if r[rx] < r[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if r[rx] == r[ry]:
            r[rx] += 1
        return True


def _find_coloring(k, edges, n, adj, deadline, order=None):
    """Backtracking with incremental per-pair union-find."""
    if order is None:
        # Use a heuristic ordering: process vertices in a way that maximizes constraints
        # For this graph, use row-by-row ordering
        order = list(range(n))
    
    coloring = [-1] * n
    pair_uf = {}
    
    # Precompute: for each vertex, which colors are forbidden by already-colored neighbors
    # This is dynamic, so we check at each step
    
    def get_pair_uf(c1, c2):
        key = (min(c1, c2), max(c1, c2))
        if key not in pair_uf:
            pair_uf[key] = UnionFind(n)
        return pair_uf[key]
    
    def try_assign(v, c):
        """Try to assign color c to vertex v. Returns (success, affected_keys, snapshots)."""
        # Check adjacency
        for nb in adj[v]:
            if coloring[nb] == c:
                return False, [], {}
        
        # Collect affected pairs
        affected_keys = []
        for nb in adj[v]:
            cn = coloring[nb]
            if cn != -1 and cn != c:
                key = (min(cn, c), max(cn, c))
                if key not in affected_keys:
                    affected_keys.append(key)
        
        # Snapshot
        snapshots = {}
        for key in affected_keys:
            if key in pair_uf:
                snapshots[key] = (pair_uf[key].parent[:], pair_uf[key].rank[:])
        
        # Try unions
        valid = True
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            key = (min(cn, c), max(cn, c))
            if key not in pair_uf:
                pair_uf[key] = UnionFind(n)
            uf = pair_uf[key]
            if not uf.union(v, nb):
                valid = False
                break
        
        if not valid:
            return False, [], {}
        
        return True, affected_keys, snapshots
    
    def undo_assign(v, c, affected_keys, snapshots):
        """Undo assignment of color c to vertex v."""
        coloring[v] = -1
        for key in affected_keys:
            if key in snapshots:
                pair_uf[key].parent = snapshots[key][0]
                pair_uf[key].rank = snapshots[key][1]
            elif key in pair_uf:
                # Was created during this assignment, remove if no longer needed
                del pair_uf[key]
    
    def backtrack(pos, deadline):
        if time.time() > deadline:
            return None
        if pos == n:
            # Final validation (should be valid if we maintained constraints)
            return coloring[:]
        
        v = order[pos]
        
        # Determine available colors (those not used by any colored neighbor)
        forbidden = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                forbidden.add(coloring[nb])
        
        # Try each non-forbidden color
        for c in range(k):
            if c in forbidden:
                continue
            
            success, affected_keys, snapshots = try_assign(v, c)
            if success:
                coloring[v] = c
                result = backtrack(pos + 1, deadline)
                if result is not None:
                    return result
                undo_assign(v, c, affected_keys, snapshots)
            
            if time.time() > deadline:
                return None
        
        return None
    
    result = backtrack(0, deadline)
    return result


def _local_search_reduce(colors, k, adj, edges, deadline):
    """Try to reduce the number of colors by recoloring vertices."""
    n = len(colors)
    current = list(colors)
    current_k = max(current) + 1
    
    # Try to eliminate colors from highest to lowest
    for target_color in range(current_k - 1, 0, -1):
        if time.time() > deadline:
            break
        
        # Find vertices with this color
        verts_to_recolor = [i for i in range(n) if current[i] == target_color]
        if not verts_to_recolor:
            continue
        
        # Try to recolor each vertex to a lower color
        success = True
        new_current = list(current)
        for v in verts_to_recolor:
            recolored = False
            for c in range(target_color):
                # Check if c is valid for v
                valid = True
                for nb in adj[v]:
                    if new_current[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_current[v] = c
                # Full validation
                if _check_coloring(new_current, edges, adj):
                    recolored = True
                    break
                new_current[v] = target_color  # restore
            
            if not recolored:
                success = False
                break
        
        if success:
            current = new_current
            current_k = target_color
    
    return current, current_k


def construct():
    start = time.time()
    deadline = start + 85  # 5s safety margin before 90s limit
    
    n = N
    edges = EDGES
    
    # Determine a good vertex ordering
    # For this graph, use a snake-like ordering that processes connected vertices together
    # Top row: 0-24, Bottom row: 25-49
    # Process in pairs (i, i+25) to maximize constraint propagation
    order = []
    for i in range(25):
        order.append(i)
        order.append(i + 25)
    
    # Try k = 3 first (most aggressive), then 4, 5, ...
    best_result = None
    
    for k in range(3, 12):
        if time.time() > deadline - 10:
            break
        
        result = _find_coloring(k, edges, n, adj, deadline, order)
        if result is not None:
            if _check_coloring(result, edges, adj):
                # Try to reduce further with local search
                reduced, reduced_k = _local_search_reduce(result, k, adj, edges, deadline)
                if _check_coloring(reduced, edges, adj):
                    if reduced_k < len(set(result)):
                        best_result = reduced
                    else:
                        best_result = result
                else:
                    best_result = result
                break
    
    if best_result is None:
        # Fallback: try k=4 with different ordering
        for k in range(4, 12):
            if time.time() > deadline - 5:
                break
            # Different ordering: reverse
            order2 = list(range(n))
            result = _find_coloring(k, edges, n, adj, deadline, order2)
            if result is not None and _check_coloring(result, edges, adj):
                best_result = result
                break
    
    if best_result is None:
        # Ultimate fallback: n colors
        return list(range(n))
    
    # Ensure we use the minimum number of distinct colors
    # Renumber colors to be 0, 1, 2, ...
    distinct = sorted(set(best_result))
    mapping = {c: i for i, c in enumerate(distinct)}
    final = [mapping[c] for c in best_result]
    
    # Verify
    if _check_coloring(final, edges, adj):
        return [int(c) for c in final]
    
    # If renumbering broke something (shouldn't happen), return original
    return [int(c) for c in best_result]
