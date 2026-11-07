"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 115.0
    n = 20
    trees = instance_trees(n, 611)
    
    # Try multiple strategies with restarts
    best_result = None
    best_score = -1
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random seed for this restart
        rng = random.Random(hash((611, restart)) % (2**32))
        
        used_edges = set()
        result = [None] * n
        
        # Embed from largest to smallest (most constrained first)
        order = list(range(n - 1, -1, -1))  # k=20 down to k=1
        
        success = True
        for idx in order:
            if time.time() > deadline:
                success = False
                break
                
            k = idx + 1
            tree_edges = trees[idx]
            
            if k == 1:
                result[idx] = [rng.randint(0, n - 1)]
                continue
            
            if k == 2:
                # Find an unused edge
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
            
            # For k >= 3, use backtracking
            embedding = _embed_tree_largest_first(tree_edges, used_edges, n, deadline, rng)
            if embedding is None:
                success = False
                break
            result[idx] = embedding
            for (a, b) in tree_edges:
                u, v = embedding[a], embedding[b]
                if u > v:
                    u, v = v, u
                used_edges.add((u, v))
        
        if success:
            # Count total edges used
            total_edges = sum(len(trees[idx]) for idx in range(n))
            score = total_edges
            if score > best_score:
                best_score = score
                best_result = [list(r) if r is not None else None for r in result]
                if best_score == 190:  # Perfect packing
                    break
    
    if best_result is None:
        # Fallback: return all None
        best_result = [None] * n
    
    return best_result

def _embed_tree_largest_first(tree_edges, used_edges, n, deadline, rng):
    """
    Embeds a tree using backtracking with smart ordering.
    Returns list of length k (host vertices for tree vertices 0..k-1) or None.
    """
    k = len(tree_edges) + 1
    
    # Build adjacency list for the tree
    tree_adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        tree_adj[u].append(v)
        tree_adj[v].append(u)
    
    # Create BFS order starting from a leaf
    start_tv = 0
    for v in range(k):
        if len(tree_adj[v]) == 1:
            start_tv = v
            break
    
    order = [start_tv]
    visited = {start_tv}
    queue = deque([start_tv])
    while queue:
        u = queue.popleft()
        for v in tree_adj[u]:
            if v not in visited:
                visited.add(v)
                order.append(v)
                queue.append(v)
    
    # Randomize host vertex order
    host_order = list(range(n))
    rng.shuffle(host_order)
    
    mapping_order = [-1] * k
    used_host_verts = [False] * n
    
    # Precompute for each position in order, which previous positions are its neighbors in tree
    neighbor_indices = [[] for _ in range(k)]
    for i in range(k):
        tv = order[i]
        for j in range(i):
            if order[j] in tree_adj[tv]:
                neighbor_indices[i].append(j)
    
    def backtrack_idx(idx):
        if time.time() > deadline:
            return False
        if idx == k:
            return True
            
        tv = order[idx]
        
        # Filter host vertices: must not be used, and edges to already-mapped neighbors must be free
        valid_hosts = []
        for hv in host_order:
            if used_host_verts[hv]:
                continue
            valid = True
            for j in neighbor_indices[idx]:
                nhv = mapping_order[j]
                u, v = hv, nhv
                if u > v:
                    u, v = v, u
                if (u, v) in used_edges:
                    valid = False
                    break
            if valid:
                valid_hosts.append(hv)
        
        if not valid_hosts:
            return False
        
        # Try valid hosts in randomized order
        rng.shuffle(valid_hosts)
        
        for hv in valid_hosts:
            mapping_order[idx] = hv
            used_host_verts[hv] = True
            
            if backtrack_idx(idx + 1):
                return True
            
            mapping_order[idx] = -1
            used_host_verts[hv] = False
        
        return False
    
    if backtrack_idx(0):
        final_mapping = [0] * k
        for i, tv in enumerate(order):
            final_mapping[tv] = mapping_order[i]
        return final_mapping
    
    return None
# EVOLVE-BLOCK-END
