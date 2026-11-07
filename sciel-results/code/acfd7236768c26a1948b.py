import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: 
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    trees = instance_trees()
    
    # T1 is special: 1 vertex, 0 edges.
    # We need to assign a host vertex to it.
    # It doesn't consume edges.
    
    # Sort tree indices by size descending
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    # We will try to pack all trees.
    # If we fail, we return the best partial packing found.
    
    best_result = [None] * 20
    best_score = 0
    best_used_edges = set()
    
    # Fallback: Seed program result
    fallback_result = [None] * 19 + [list(range(20))]
    # Calculate score for fallback
    # T20 has 19 edges.
    best_score = 19
    best_result = fallback_result
    # Initialize best_used_edges with T20 edges
    for u, v in trees[19]:
        best_used_edges.add((min(u, v), max(u, v)))
        
    # Attempt to pack
    # We'll try a few random permutations of the tree order, prioritizing larger trees.
    
    rng = Random(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Create a shuffled order, but keep T1 (index 0) separate?
        # T1 is trivial.
        # Let's shuffle the non-trivial trees.
        non_trivial = [i for i in tree_indices if i != 0]
        rng.shuffle(non_trivial)
        current_order = [0] + non_trivial # T1 first? Or last?
        # T1 takes a vertex but no edge. It might block a vertex needed for others?
        # "list of k distinct int host vertices".
        # It does NOT say trees must be vertex-disjoint.
        # "Embedded trees must be edge-disjoint".
        # So T1 can share vertices with others.
        # So T1 is easy. Put it last or first, doesn't matter much.
        
        current_used = set(best_used_edges) if attempt == 0 else set()
        # If we restart from scratch, we should start with empty used_edges.
        # But we want to find a FULL packing.
        # So we should start with empty used_edges for each attempt.
        current_used = set()
        current_result = [None] * 20
        success = True
        
        for idx in current_order:
            tree = trees[idx]
            
            if not tree:
                # T1
                # Assign any vertex, say 0
                current_result[idx] = [0]
                continue
            
            # Find embedding
            mapping = find_embedding(tree, current_used, time.time())
            
            if mapping is None:
                success = False
                break
            
            current_result[idx] = mapping
            # Update used edges
            for u, v in tree:
                e = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                current_used.add(e)
        
        if success:
            # Full packing found
            return current_result
            
        # If not successful, check if we packed more than best_score
        # Count edges in current_result
        current_edges = 0
        for i in range(20):
            if current_result[i]:
                current_edges += len(trees[i])
        
        if current_edges > best_score:
            best_score = current_edges
            best_result = current_result
            best_used_edges = current_used

    return best_result

def find_embedding(tree, used_edges, current_time):
    # tree is a list of edges
    k = len(tree) + 1
    
    # Build adjacency
    adj = [[] for _ in range(k)]
    for u, v in tree:
        adj[u].append(v)
        adj[v].append(u)
    
    # Determine order of assignment: BFS from 0
    order = [0]
    visited = {0}
    i = 0
    while i < len(order):
        u = order[i]
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                order.append(v)
        i += 1
    
    # Mapping
    mapping = [-1] * k
    used_hosts = [False] * 20
    
    # Heuristic: Try to assign vertices with higher degree in the tree first?
    # Or just follow BFS order.
    
    # To speed up, we can check if the remaining graph has enough edges?
    # Simple backtracking
    
    def backtrack(pos):
        if time.time() - current_time > 5: # 5s timeout per tree
            return False
            
        if pos == k:
            return True
        
        u = order[pos]
        
        # Try all host vertices
        # Optimization: Only try vertices that are connected to already assigned neighbors via available edges
        # If u has no assigned neighbors, any unused host is valid.
        # If u has assigned neighbors, the host must be connected to all their hosts via available edges.
        
        candidates = []
        for h in range(20):
            if used_hosts[h]:
                continue
            
            valid = True
            for v in adj[u]:
                if mapping[v] != -1:
                    hv = mapping[v]
                    e = (min(h, hv), max(h, hv))
                    if e in used_edges:
                        valid = False
                        break
            
            if valid:
                candidates.append(h)
                
        # Shuffle candidates to break symmetry?
        # import random
        # random.shuffle(candidates)
        
        for h in candidates:
            mapping[u] = h
            used_hosts[h] = True
            
            if backtrack(pos + 1):
                return True
            
            mapping[u] = -1
            used_hosts[h] = False
        
        return False
    
    if backtrack(0):
        return mapping
    return None
