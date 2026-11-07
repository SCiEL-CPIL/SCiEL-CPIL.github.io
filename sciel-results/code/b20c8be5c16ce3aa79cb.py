import time
import random
from itertools import combinations

def _instance_trees(n=20, seed=611):
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

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _embed_tree_backtrack(tree_edges, n, available_edges_set, rng):
    """
    Tries to embed tree_edges into K_n using only edges in available_edges_set.
    Returns a list of length len(tree_edges) where i-th element is the host vertex of tree vertex i,
    or None if no embedding found.
    """
    if not tree_edges:
        return [0]  # Single vertex tree, map to 0
    
    # Build adjacency list for the tree
    m = len(tree_edges) + 1
    adj = [[] for _ in range(m)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find a root (e.g., vertex 0)
    root = 0
    # We fix root to 0 in K_n to break symmetry
    # But we can also try other roots if 0 fails? No, symmetry means if it exists, it exists with root at 0.
    
    # DFS to find a valid embedding
    # mapping: tree_vertex -> host_vertex
    mapping = [None] * m
    mapping[root] = 0
    
    # Pre-compute neighbors in K_n for fast lookup? 
    # available_edges_set contains frozenset or tuple of (u,v) with u<v.
    
    def dfs(node, parent):
        if time.time() > 110: # Safety margin
            return False
            
        # If this is not the root, check if the edge (parent, node) is available
        if parent is not None:
            p_host = mapping[parent]
            # Check if (p_host, candidate) is in available_edges_set
            # We will iterate over candidates for 'node'
            pass
        
        # Determine candidates for 'node'
        if node == root:
            candidates = [0] # Fixed
        else:
            p_host = mapping[parent]
            # Candidates are neighbors of p_host in K_n such that the edge is available
            # And the candidate must not be used by any other tree vertex (except parent)
            # We maintain a used_hosts set
            candidates = []
            for h in range(n):
                if h == p_host:
                    continue
                if h in used_hosts:
                    continue
                # Check edge availability
                e = (min(p_host, h), max(p_host, h))
                if e in available_edges_set:
                    candidates.append(h)
            
            # Heuristic: sort candidates by some criterion? 
            # Maybe degree in K_n remaining? Or just randomize to avoid bias
            rng.shuffle(candidates)
            
        for h in candidates:
            if node != root and h in used_hosts:
                continue
            mapping[node] = h
            if node != root:
                used_hosts.add(h)
            
            # Recurse to children
            success = True
            for child in adj[node]:
                if child == parent:
                    continue
                if not dfs(child, node):
                    success = False
                    break
            
            if success:
                return True
            else:
                if node != root:
                    used_hosts.remove(h)
                mapping[node] = None
        
        return False

    used_hosts = {0}
    # We need to pass used_hosts and mapping to dfs. 
    # Let's rewrite dfs to be cleaner.
    
    # Actually, the above dfs has issues with state management. Let's use a standard recursive backtracker.
    
    def solve(node, parent):
        if time.time() > 110:
            return False
            
        if node == root:
            # Already set to 0
            pass
        else:
            p_host = mapping[parent]
            # Find available neighbors of p_host
            cands = []
            for h in range(n):
                if h in used_hosts:
                    continue
                e = (min(p_host, h), max(p_host, h))
                if e in available_edges_set:
                    cands.append(h)
            rng.shuffle(cands)
            
            for h in cands:
                mapping[node] = h
                used_hosts.add(h)
                
                ok = True
                for child in adj[node]:
                    if child == parent:
                        continue
                    if not solve(child, node):
                        ok = False
                        break
                
                if ok:
                    return True
                else:
                    used_hosts.remove(h)
                    mapping[node] = None
            return False
        
        # Process children
        for child in adj[node]:
            if child == parent:
                continue
            if not solve(child, node):
                return False
        return True

    # Initialize
    mapping = [None] * m
    used_hosts = set()
    mapping[root] = 0
    used_hosts.add(0)
    
    if solve(root, None):
        return mapping
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115 # 5s margin
    
    trees = _instance_trees(20, 611)
    n = 20
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_edges.add((i, j))
    
    # Sort tree indices by size descending
    indices = list(range(n))
    indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    best_result = [None] * n
    
    # Try multiple random orders/restarts
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        # Randomize the order of trees slightly? 
        # Or just use the sorted order but randomize tie-breaking in embedding
        rng = random.Random(attempt)
        
        current_edges = set(all_edges)
        result = [None] * n
        
        success = True
        for idx in indices:
            tree = trees[idx]
            if time.time() > deadline:
                success = False
                break
                
            # If tree is empty (k=1), map to 0? 
            # The problem says "list of k distinct int host vertices". For k=1, it's a list of 1 vertex.
            # But the seed program returns None for k<20. 
            # Wait, the task says "Entry k-1 is None or a list...".
            # If we embed all, we should return lists for all.
            
            if not tree:
                # Single vertex tree. Any vertex works. Let's pick a random one? 
                # Or 0? 
                # If we pick 0, it doesn't consume edges.
                # But we must return a list of 1 distinct int.
                # Let's pick a random available vertex? Or just 0.
                # Since it consumes no edges, it's always possible.
                # But we need to ensure distinctness? "k distinct int host vertices".
                # For k=1, just [0].
                result[idx] = [0]
                continue
                
            mapping = _embed_tree_backtrack(tree, n, current_edges, rng)
            if mapping is None:
                success = False
                break
                
            # Update current_edges
            for u, v in tree:
                hu, hv = mapping[u], mapping[v]
                e = (min(hu, hv), max(hu, hv))
                current_edges.remove(e)
                
            result[idx] = mapping
            
        if success:
            # Verify
            # Check distinctness within each tree
            for i in range(n):
                if result[i] is None:
                    continue
                if len(result[i]) != len(trees[i]) + 1:
                    # Should not happen
                    pass
                if len(set(result[i])) != len(result[i]):
                    # Not distinct
                    success = False
                    break
            
            if success:
                # Check edge disjointness
                used = set()
                for i in range(n):
                    if result[i] is None:
                        continue
                    for u, v in trees[i]:
                        hu, hv = result[i][u], result[i][v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used:
                            success = False
                            break
                        used.add(e)
                    if not success:
                        break
                
                if success:
                    # Ensure types are correct
                    final_result = []
                    for i in range(n):
                        if result[i] is None:
                            final_result.append(None)
                        else:
                            final_result.append([int(x) for x in result[i]])
                    return final_result
                    
    # If we didn't find a full packing, return the best partial? 
    # The task says "The raw value is the total number of embedded tree edges".
    # So partial is better than None.
    # But we usually find full packing.
    # If not, we should return whatever we have.
    # However, the loop above only returns if success.
    # Let's modify to keep track of best partial.
    
    # Re-implement to track best partial
    best_val = -1
    best_res = [None] * n
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        rng = random.Random(attempt)
        current_edges = set(all_edges)
        result = [None] * n
        val = 0
        
        for idx in indices:
            tree = trees[idx]
            if time.time() > deadline:
                break
                
            if not tree:
                result[idx] = [0]
                continue
                
            mapping = _embed_tree_backtrack(tree, n, current_edges, rng)
            if mapping is None:
                break
                
            for u, v in tree:
                hu, hv = mapping[u], mapping[v]
                e = (min(hu, hv), max(hu, hv))
                current_edges.remove(e)
                
            result[idx] = mapping
            val += len(tree)
            
        if val > best_val:
            best_val = val
            best_res = [x[:] if x is not None else None for x in result]
            
    # Convert to correct types
    final_result = []
    for i in range(n):
        if best_res[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_res[i]])
            
    return final_result
