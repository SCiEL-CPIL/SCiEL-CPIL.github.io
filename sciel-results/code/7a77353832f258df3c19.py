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
    start_time = time.time()
    deadline = start_time + 145
    
    trees = instance_trees(20, 611)
    
    # Precompute tree properties
    tree_info = []
    for i in range(20):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_info.append({
            'k': k,
            'edges': edges,
            'adj': adj,
            'deg': deg,
            'num_edges': len(edges)
        })
    
    # Precompute a DFS order for each tree.
    def get_dfs_order(adj, k, root=0):
        visited = [False] * k
        order = []
        def dfs(u):
            visited[u] = True
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    dfs(v)
        dfs(root)
        return order

    dfs_orders = []
    dfs_pos = []
    for i in range(20):
        k = tree_info[i]['k']
        if k == 1:
            dfs_orders.append([0])
            dfs_pos.append([0])
        else:
            ord_list = get_dfs_order(tree_info[i]['adj'], k, 0)
            dfs_orders.append(ord_list)
            pos_map = [-1] * k
            for j, v in enumerate(ord_list):
                pos_map[v] = j
            dfs_pos.append(pos_map)

    def find_embedding(t_idx, used_edges, rng):
        info = tree_info[t_idx]
        k = info['k']
        adj_t = info['adj']
        deg_t = info['deg']
        dfs_order = dfs_orders[t_idx]
        pos_map = dfs_pos[t_idx]
        
        if k == 1:
            # Pick a random unused host? 
            # For k=1, no edges used. Any host is fine.
            # To maximize future flexibility, maybe pick one with high degree?
            # But degree is always 19 in K20 minus used edges.
            # Let's just pick 0 or random.
            # Since it doesn't consume edges, it doesn't matter for edge constraints.
            # But it consumes a vertex.
            # All vertices are equivalent for future edge availability?
            # No, if we pick a vertex, it's gone.
            # But since it has no edges, it doesn't block any specific edges.
            # It just removes a vertex from the pool.
            # So any vertex is fine.
            return [0]

        mapping = [-1] * k
        used_hosts = set()
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            
            if pos == k:
                return True
            
            tree_v = dfs_order[pos]
            
            mapped_neighbors = []
            for nb in adj_t[tree_v]:
                if pos_map[nb] < pos:
                    mapped_neighbors.append(nb)
            
            candidates = []
            for h in range(20):
                if h in used_hosts:
                    continue
                valid = True
                for nb in mapped_neighbors:
                    hn = mapping[nb]
                    e = (min(h, hn), max(h, hn))
                    if e in used_edges:
                        valid = False
                        break
                if valid:
                    candidates.append(h)
            
            if not candidates:
                return False
            
            # Heuristic: Most constrained first
            # Add random tie-breaker to explore different solutions
            def count_available_edges(host):
                count = 0
                for nb in range(20):
                    if nb == host or nb in used_hosts:
                        continue
                    e = (min(host, nb), max(host, nb))
                    if e not in used_edges:
                        count += 1
                return count
            
            # Sort by count, then by random value to break ties randomly
            candidates.sort(key=lambda h: (count_available_edges(h), rng.random()))
            
            for h in candidates:
                mapping[tree_v] = h
                used_hosts.add(h)
                
                # Forward Checking
                fc_ok = True
                for nb in adj_t[tree_v]:
                    if pos_map[nb] >= pos:
                        has_valid = False
                        for h2 in range(20):
                            if h2 in used_hosts:
                                continue
                            valid_h2 = True
                            for nb2 in adj_t[nb]:
                                if pos_map[nb2] < pos:
                                    h_nb2 = mapping[nb2]
                                    e = (min(h2, h_nb2), max(h2, h_nb2))
                                    if e in used_edges:
                                        valid_h2 = False
                                        break
                            if valid_h2:
                                has_valid = True
                                break
                        if not has_valid:
                            fc_ok = False
                            break
                
                if fc_ok and backtrack(pos + 1):
                    return True
                
                mapping[tree_v] = -1
                used_hosts.remove(h)
            
            return False
        
        # Try a few times with different random seeds for tie-breaking?
        # The rng is passed in. We can just call backtrack once.
        # If it fails, the caller can try again with a new rng or new order.
        if backtrack(0):
            return mapping
        return None

    def try_pack(attempt_seed):
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * 20
        
        # Randomize the order of trees
        order = list(range(20))
        local_rng.shuffle(order)
        
        # Optional: Sort by size descending? 
        # If we want to maximize edges, maybe large trees first?
        # But random order might find a packing that sorted order misses.
        # Let's try a mix: 50% random, 50% size-descending?
        # Or just random.
        
        for t_idx in order:
            if time.time() > deadline:
                break
            
            mapping = find_embedding(t_idx, used_edges, local_rng)
            if mapping is None:
                # Retry a few times
                found = False
                for _ in range(10):
                    mapping = find_embedding(t_idx, used_edges, local_rng)
                    if mapping is not None:
                        found = True
                        break
                if not found:
                    # Skip this tree
                    continue
            
            mappings[t_idx] = mapping
            for u, v in trees[t_idx]:
                hu = mapping[u]
                hv = mapping[v]
                e = (min(hu, hv), max(hu, hv))
                used_edges.add(e)
        
        score = 0
        for i in range(20):
            if mappings[i] is not None:
                score += tree_info[i]['num_edges']
        
        return score, mappings

    best_score = 0
    best_mappings = [None] * 20
    
    attempt = 0
    while time.time() < deadline:
        score, mappings = try_pack(attempt)
        if score > best_score:
            best_score = score
            best_mappings = mappings[:]
            if best_score == 190:
                break
        attempt += 1
        
    result = []
    for i in range(20):
        if best_mappings[i] is not None:
            result.append([int(v) for v in best_mappings[i]])
        else:
            result.append(None)
    
    return result

if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    
    trees = instance_trees(20, 611)
    used_edges = set()
    total_edges = 0
    valid = True
    for i in range(20):
        if result[i] is None:
            continue
        mapping = result[i]
        k = i + 1
        if len(mapping) != k:
            print(f"Tree {i}: wrong length {len(mapping)} vs {k}")
            valid = False
            break
        if len(set(mapping)) != k:
            print(f"Tree {i}: duplicate host vertices")
            valid = False
            break
        for v in mapping:
            if not isinstance(v, int) or v < 0 or v > 19:
                print(f"Tree {i}: invalid vertex {v}")
                valid = False
                break
        if not valid:
            break
        for u, v in trees[i]:
            hu = mapping[u]
            hv = mapping[v]
            e = (min(hu, hv), max(hu, hv))
            if e in used_edges:
                print(f"Tree {i}: edge conflict {e}")
                valid = False
                break
            used_edges.add(e)
            total_edges += 1
        if not valid:
            break
    
    print(f"Valid: {valid}")
    print(f"Total edges: {total_edges}")
    if valid:
        print(f"Score: {total_edges}")
    else:
        print("INVALID RESULT")
