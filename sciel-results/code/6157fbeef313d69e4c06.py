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
    # 150s limit. Use 145s as deadline to be safe.
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
    
    # Sort by size descending (largest first) to constrain early
    order = sorted(range(20), key=lambda i: -tree_info[i]['k'])
    
    # Precompute a DFS order for each tree.
    # We want an order where parents come before children.
    # Using DFS ensures that when we place a node, its parent is already placed.
    # We'll use a specific root (0) and traverse.
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
        """
        Finds a valid embedding for tree t_idx into K20 avoiding used_edges.
        Returns list of length k where mapping[i] is host vertex for tree vertex i.
        """
        info = tree_info[t_idx]
        k = info['k']
        adj_t = info['adj']
        deg_t = info['deg']
        dfs_order = dfs_orders[t_idx]
        pos_map = dfs_pos[t_idx]
        
        if k == 1:
            # Any host vertex works. Pick one that has the most available edges? 
            # Actually, for k=1, no edges are used. Just pick 0 for simplicity or random.
            # Since no edges are consumed, it doesn't matter for future constraints.
            return [0]

        mapping = [-1] * k
        used_hosts = set()
        
        # Pre-compute available hosts for each potential tree vertex?
        # No, availability depends on the current state of used_edges and used_hosts.
        
        # We will backtrack on the DFS order.
        # At each step, we assign a host vertex to the current tree vertex.
        
        # Heuristic: 
        # 1. The tree vertex is fixed by the order (dfs_order[pos]).
        # 2. The host vertex candidates are those not in used_hosts.
        # 3. Pruning: The host vertex must have unused edges to all already-mapped neighbors of the tree vertex.
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            
            if pos == k:
                return True
            
            tree_v = dfs_order[pos]
            
            # Identify neighbors of tree_v that are already mapped (parents in DFS tree)
            mapped_neighbors = []
            for nb in adj_t[tree_v]:
                if pos_map[nb] < pos:
                    mapped_neighbors.append(nb)
            
            # Candidates: Host vertices not in used_hosts
            # Optimization: We can iterate 0..19 and check.
            # Further optimization: If mapped_neighbors is not empty, 
            # the host must be connected to the hosts of mapped_neighbors via unused edges.
            
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
            
            # Heuristic: Sort candidates by number of available edges (ascending).
            # Most-constrained-first: pick hosts with fewer remaining connections first.
            # This prunes the search space more effectively.
            def count_available_edges(host):
                count = 0
                for nb in range(20):
                    if nb == host or nb in used_hosts:
                        continue
                    e = (min(host, nb), max(host, nb))
                    if e not in used_edges:
                        count += 1
                return count
            
            candidates.sort(key=count_available_edges)
            
            for h in candidates:
                mapping[tree_v] = h
                used_hosts.add(h)
                
                # Forward Checking: 
                # For each unmapped neighbor of tree_v, verify at least one valid host exists.
                fc_ok = True
                for nb in adj_t[tree_v]:
                    if pos_map[nb] >= pos:  # unmapped neighbor
                        # Check if nb has at least one valid host
                        has_valid = False
                        for h2 in range(20):
                            if h2 in used_hosts:
                                continue
                            # h2 must connect to all already-mapped neighbors of nb
                            valid_h2 = True
                            for nb2 in adj_t[nb]:
                                if pos_map[nb2] < pos:  # nb2 is mapped
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
        
        # With deterministic most-constrained-first ordering, fewer retries needed.
        # Try a few times in case of unlucky edge cases.
        for _ in range(10):
            mapping = [-1] * k
            used_hosts = set()
            if backtrack(0):
                return mapping
        return None

    def try_pack(attempt_seed):
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * 20
        
        for t_idx in order:
            if time.time() > deadline:
                break
            
            mapping = find_embedding(t_idx, used_edges, local_rng)
            if mapping is None:
                # If failed, try again a few more times
                found = False
                for _ in range(5):
                    mapping = find_embedding(t_idx, used_edges, local_rng)
                    if mapping is not None:
                        found = True
                        break
                if not found:
                    # Failed to pack this tree. Return what we have.
                    pass
                else:
                    mappings[t_idx] = mapping
                    for u, v in trees[t_idx]:
                        hu = mapping[u]
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        used_edges.add(e)
            else:
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
        attempt += 1
        if best_score == 190:
            break

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
