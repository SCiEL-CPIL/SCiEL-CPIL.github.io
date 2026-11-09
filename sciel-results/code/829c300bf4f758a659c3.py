import sys
from collections import defaultdict

def instance_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed)
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
    trees = instance_trees(20, 611)
    n = 20
    
    # Initialize available edges
    available_edges = set()
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            available_edges.add(frozenset((i, j)))
            adj[i].add(j)
            adj[j].add(i)
            
    results = [None] * 20
    
    for k in range(1, 21):
        tree_edges = trees[k-1]
        
        if k == 1:
            # Single vertex, no edges. Any host is fine.
            # We pick 0 for determinism.
            results[0] = [0]
            continue
        
        # Build tree adjacency
        tree_adj = defaultdict(list)
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
            
        # We will try to embed the tree
        # Root at 0
        root = 0
        mapping = {}
        used_hosts = set()
        
        def dfs(current_v, host_v):
            if current_v in mapping:
                return True
            
            mapping[current_v] = host_v
            used_hosts.add(host_v)
            
            # Get neighbors in the tree
            neighbors = tree_adj[current_v]
            
            # Optional: Sort neighbors by some heuristic?
            # Not strictly necessary for trees, but might help.
            
            for neighbor in neighbors:
                if neighbor in mapping:
                    continue
                
                # Find candidates for neighbor
                candidates = []
                for h in adj[host_v]:
                    if h not in used_hosts:
                        candidates.append(h)
                        
                # Heuristic: Sort by degree in host graph (ascending)
                candidates.sort(key=lambda h: len(adj[h]))
                
                success = False
                for h in candidates:
                    if dfs(neighbor, h):
                        success = True
                        break
                    # Undo
                    if neighbor in mapping:
                        del mapping[neighbor]
                    used_hosts.discard(h)
                    
                if success:
                    return True
                else:
                    # No candidate worked for this neighbor
                    # Backtrack current_v
                    return False
            
            return True
        
        found = False
        final_mapping = None
        
        # Try all hosts for the root
        for root_host in range(n):
            mapping.clear()
            used_hosts.clear()
            if dfs(root, root_host):
                found = True
                final_mapping = dict(mapping)
                break
        
        if not found:
            # Cannot embed this tree
            break
        else:
            # Construct result list
            res_list = [0] * k
            for tv, hv in final_mapping.items():
                res_list[tv] = hv
            results[k-1] = res_list
            
            # Remove used edges from available_edges
            for u, v in tree_edges:
                hu = final_mapping[u]
                hv = final_mapping[v]
                e = frozenset((hu, hv))
                if e in available_edges:
                    available_edges.remove(e)
                    adj[hu].remove(hv)
                    adj[hv].remove(hu)
                    
    return results
