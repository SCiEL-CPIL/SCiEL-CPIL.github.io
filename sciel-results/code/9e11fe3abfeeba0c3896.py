import time
from random import Random
from itertools import combinations

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
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
    deadline = start_time + 115  # Leave 35s margin for 150s limit
    trees = instance_trees(20, 611)
    n = 20
    
    # Pre-process trees
    tree_infos = []
    for i, edges in enumerate(trees):
        k = i + 1
        if k == 1:
            tree_infos.append((edges, {}, [0], 1))
            continue
        
        adj = {v: [] for v in range(k)}
        degs = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degs[u] += 1
            degs[v] += 1
        
        tree_infos.append((edges, adj, degs, k))

    # Sort tree indices: largest first, tie-break by max degree
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: (-(len(trees[i])), -max(tree_infos[i][2]) if trees[i] else 0))

    # State: available edges in K20 as a set
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    mappings = [None] * 20
    total_edges_used = [0]  # mutable counter
    
    def backtrack(tree_pos, available_edges):
        if time.time() > deadline:
            return False
            
        if tree_pos == 20:
            return len(available_edges) == 0

        t_idx = tree_indices[tree_pos]
        edges, adj, degs, k = tree_infos[t_idx]
        
        if k == 1:
            mappings[t_idx] = [0]
            if backtrack(tree_pos + 1, available_edges):
                return True
            mappings[t_idx] = None
            return False

        # For k > 1: assign tree vertices to host vertices
        tv_order = sorted(range(k), key=lambda v: -degs[v])
        host_map = {}
        
        # Pre-compute current degrees in available_edges for all hosts
        avail_deg = [0] * n
        for (u, v) in available_edges:
            avail_deg[u] += 1
            avail_deg[v] += 1
        
        def assign_tv(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                # All tree vertices assigned. Verify edges and remove them
                new_avail = set(available_edges)
                for u, v in edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in new_avail:
                        return False
                    new_avail.discard(edge)
                
                temp_map = [0] * k
                for tv, hv in host_map.items():
                    temp_map[tv] = hv
                mappings[t_idx] = temp_map
                total_edges_used[0] += len(edges)
                
                if backtrack(tree_pos + 1, new_avail):
                    return True
                
                mappings[t_idx] = None
                total_edges_used[0] -= len(edges)
                return False
            
            tv = tv_order[pos]
            deg_t = degs[tv]
            
            used_hosts = set(host_map.values())
            
            # Build candidate list sorted by avail_deg descending for better pruning
            candidates = [h for h in range(n) if h not in used_hosts]
            candidates.sort(key=lambda h: -avail_deg[h])
            
            for h in candidates:
                # Check degree constraint
                if avail_deg[h] < deg_t:
                    continue
                
                # Check neighbors: for each already-assigned neighbor, edge must be available
                valid = True
                for nb in adj[tv]:
                    if nb in host_map:
                        hn = host_map[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in available_edges:
                            valid = False
                            break
                if not valid:
                    continue
                
                host_map[tv] = h
                if assign_tv(pos + 1):
                    return True
                del host_map[tv]
                
            return False

        # Symmetry breaking for first tree
        if tree_pos == 0:
            tv0 = tv_order[0]
            host_map[tv0] = 0
            if assign_tv(1):
                return True
            del host_map[tv0]
        else:
            if assign_tv(0):
                return True

        return False

    if backtrack(0, all_edges):
        result = [None] * 20
        for i in range(20):
            if mappings[i] is not None:
                result[i] = [int(x) for x in mappings[i]]
        return result
    else:
        # Return best partial result found
        result = [None] * 20
        for i in range(20):
            if mappings[i] is not None:
                result[i] = [int(x) for x in mappings[i]]
        return result
