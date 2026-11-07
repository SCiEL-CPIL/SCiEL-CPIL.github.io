"""Rewritten program with improved tree packing algorithm."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 115.0
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute tree properties
    tree_info = []
    for idx in range(n):
        k = idx + 1
        edges = trees[idx]
        if k == 1:
            tree_info.append((k, 0, edges, [], []))
        else:
            adj = [[] for _ in range(k)]
            deg = [0] * k
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
            # Precompute parent in BFS tree for ordering
            bfs_order, parent = _bfs_order(adj, k)
            tree_info.append((k, deg, edges, adj, (bfs_order, parent)))
    
    total_possible = sum(k - 1 for k in range(1, n + 1))  # 190
    
    best_result = None
    best_score = -1
    
    max_restarts = 500
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(12345 + restart * 7919)
        
        used_edges = set()
        result = [None] * n
        
        # Embed in order: largest first
        order = list(range(n - 1, -1, -1))
        
        # Occasionally shuffle the middle trees for diversity
        if restart % 3 == 0 and len(order) > 4:
            mid = order[1:-1]
            rng.shuffle(mid)
            order = [order[0]] + mid + [order[-1]]
        
        success = True
        for idx in order:
            if time.time() > deadline:
                success = False
                break
            
            k, deg, edges, adj, (bfs_order, parent) = tree_info[idx]
            
            if k == 1:
                candidates = [v for v in range(n)]
                if not candidates:
                    success = False
                    break
                result[idx] = [rng.choice(candidates)]
                continue
            
            if k == 2:
                candidates = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (u, v) not in used_edges:
                            candidates.append((u, v))
                if not candidates:
                    success = False
                    break
                u, v = rng.choice(candidates)
                result[idx] = [u, v]
                used_edges.add((u, v))
                continue
            
            embedding = _embed_tree_v2(adj, deg, k, used_edges, n, deadline, rng, bfs_order, parent)
            if embedding is None:
                success = False
                break
            result[idx] = embedding
            for (a, b) in edges:
                u, v = embedding[a], embedding[b]
                if u > v:
                    u, v = v, u
                used_edges.add((u, v))
        
        if success:
            total_edges = sum(len(trees[idx]) for idx in range(n))
            if total_edges > best_score:
                best_score = total_edges
                best_result = [list(r) if r is not None else None for r in result]
                if best_score == total_possible:
                    break
    
    if best_result is None:
        best_result = [None] * n
    
    return best_result

def _bfs_order(adj, k):
    """BFS from vertex 0, return order and parent array."""
    visited = {0}
    parent = [-1] * k
    order = [0]
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                order.append(v)
                queue.append(v)
    return order, parent

def _embed_tree_v2(adj, deg, k, used_edges, n, deadline, rng, bfs_order, parent):
    """
    Improved tree embedding with LCV heuristic and dynamic variable ordering.
    """
    # Precompute for each tree vertex: which tree vertices are its neighbors
    tree_adj = [set(adj[v]) for v in range(k)]
    
    # Host vertex usage tracking
    used_host = [False] * n
    host_used_deg = [0] * n
    for (u, v) in used_edges:
        host_used_deg[u] += 1
        host_used_deg[v] += 1
    host_remaining = [n - 1 - host_used_deg[i] for i in range(n)]
    
    # Mapping: tree vertex -> host vertex
    mapping = [-1] * k
    
    # For each tree vertex, track which of its tree-neighbors are already mapped
    # and what host vertices they map to
    mapped_neighbors = [set() for _ in range(k)]  # set of (tree_neighbor, host_vertex)
    
    # Count how many tree-neighbors of each tree vertex are already mapped
    mapped_count = [0] * k
    
    # Which tree vertices have all their neighbors mapped (or are leaves with parent mapped)
    # We embed in BFS order: a vertex is ready when its parent is mapped
    # But we can also use dynamic ordering: pick the unmapped vertex with most constraints
    
    def get_candidates(tv):
        """Get valid host vertices for tree vertex tv."""
        cands = []
        # Neighbors of tv that are already mapped
        mapped_nbs = mapped_neighbors[tv]
        for hv in range(n):
            if used_host[hv]:
                continue
            if host_remaining[hv] < deg[tv]:
                continue
            valid = True
            for (tn, nhv) in mapped_nbs:
                u, v = hv, nhv
                if u > v:
                    u, v = v, u
                if (u, v) in used_edges:
                    valid = False
                    break
            if valid:
                cands.append(hv)
        return cands
    
    def try_assign(tv, hv):
        """Assign tree vertex tv to host vertex hv. Returns list of changes to undo."""
        changes = []
        mapping[tv] = hv
        used_host[hv] = True
        host_remaining[hv] -= deg[tv]
        changes.append(('map', tv, hv))
        
        # Update mapped_neighbors for all unmapped tree-neighbors of tv
        for tn in tree_adj[tv]:
            if mapping[tn] == -1:
                mapped_neighbors[tn].add((tv, hv))
                mapped_count[tn] += 1
                changes.append(('nb', tn, tv, hv))
        
        return changes
    
    def undo_assign(changes):
        """Undo an assignment."""
        for change in reversed(changes):
            if change[0] == 'nb':
                _, tn, tv, hv = change
                mapped_neighbors[tn].discard((tv, hv))
                mapped_count[tn] -= 1
            elif change[0] == 'map':
                _, tv, hv = change
                mapping[tv] = -1
                used_host[hv] = False
                host_remaining[hv] += deg[tv]
    
    # Dynamic variable ordering: at each step, pick the unmapped vertex with
    # the fewest valid candidates (MRV), breaking ties by highest degree (LCV)
    
    def backtrack(depth):
        if time.time() > deadline:
            return False
        if depth == k:
            return True
        
        # Find unmapped vertices
        unmapped = [v for v in range(k) if mapping[v] == -1]
        if not unmapped:
            return True
        
        # MRV: pick vertex with fewest candidates
        best_tv = None
        best_cands = None
        best_cand_count = float('inf')
        
        for tv in unmapped:
            cands = get_candidates(tv)
            if not cands:
                return False  # Dead end
            if len(cands) < best_cand_count:
                best_cand_count = len(cands)
                best_tv = tv
                best_cands = cands
                if best_cand_count == 1:
                    break
        
        # LCV: sort candidates by how many options they leave for neighbors
        if len(best_cands) > 1:
            def lcv_score(hv):
                """Score = number of valid options left for unmapped neighbors."""
                score = 0
                for tn in tree_adj[best_tv]:
                    if mapping[tn] == -1:
                        # Check how many host vertices tn could use
                        count = 0
                        for h2 in range(n):
                            if used_host[h2] or h2 == hv:
                                continue
                            if host_remaining[h2] - (1 if h2 == hv else 0) < deg[tn]:
                                continue
                            valid = True
                            for (tn2, nhv2) in mapped_neighbors[tn]:
                                u, v = h2, nhv2
                                if u > v:
                                    u, v = v, u
                                if (u, v) in used_edges:
                                    valid = False
                                    break
                            if valid:
                                count += 1
                        score += count
                return score
            
            best_cands.sort(key=lcv_score, reverse=True)
        
        # Try each candidate
        for hv in best_cands:
            changes = try_assign(best_tv, hv)
            if backtrack(depth + 1):
                return True
            undo_assign(changes)
        
        return False
    
    # Start: assign vertex 0 (root of BFS) first to reduce branching
    # Actually, let's just use the dynamic ordering from the start
    if backtrack(0):
        return mapping
    
    return None
# EVOLVE-BLOCK-END
