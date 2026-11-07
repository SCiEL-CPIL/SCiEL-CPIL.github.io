<<<<<<< SEARCH
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
    deadline = start_time + 140  # 10s margin before 150s kill
    
    trees = instance_trees(20, 611)
    
    # Precompute tree properties
    tree_info = []
    for i in range(20):
        edges = trees[i]
        k = i + 1
        # Build adjacency
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
    
    # Sort by size descending for backtracking efficiency
    # We pack larger trees first
    order = sorted(range(20), key=lambda i: -tree_info[i]['k'])
    
    # Precompute a BFS order for each tree for efficient mapping
    def get_bfs_order(adj, k, root=0):
        """Get BFS ordering starting from root"""
        visited = [False] * k
        visited[root] = True
        queue = deque([root])
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return order
    
    bfs_orders = []
    for i in range(20):
        k = tree_info[i]['k']
        if k == 1:
            bfs_orders.append([0])
        else:
            bfs_orders.append(get_bfs_order(tree_info[i]['adj'], k))
    
    # Precompute position in BFS order for fast lookup
    bfs_pos = []
    for i in range(20):
        k = tree_info[i]['k']
        pos = [-1] * k
        for j, v in enumerate(bfs_orders[i]):
            pos[v] = j
        bfs_pos.append(pos)
    
    def try_pack(attempt_seed):
        """Try to pack all trees using a given random seed for tie-breaking"""
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * 20
        
        def find_mapping(t_idx):
            """
            Find a valid embedding of tree t_idx into K20
            using backtracking with pruning.
            """
            info = tree_info[t_idx]
            k = info['k']
            adj_t = info['adj']
            bfs_order = bfs_orders[t_idx]
            pos_map = bfs_pos[t_idx]
            
            if k == 1:
                return [0]
            
            def backtrack(pos, mapping, used_hosts):
                if time.time() > deadline:
                    return False, None
                
                if pos == k:
                    # All vertices mapped - verify all edges are valid
                    valid = True
                    for u, v in info['edges']:
                        hu = mapping[u]
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            valid = False
                            break
                    if valid:
                        return True, mapping[:]
                    return False, None
                
                tree_vertex = bfs_order[pos]
                
                # Get candidates: host vertices not yet used
                candidates = [hv for hv in range(20) if hv not in used_hosts]
                
                # Additional pruning: if this tree vertex has neighbors already mapped,
                # the host must have unused edges to all mapped neighbors' hosts
                if pos > 0:
                    # Find mapped neighbors (in BFS order, only parent is before)
                    mapped_neighbors = []
                    for nb in adj_t[tree_vertex]:
                        if pos_map[nb] < pos:
                            mapped_neighbors.append((nb, mapping[nb]))
                    
                    if mapped_neighbors:
                        filtered = []
                        for hv in candidates:
                            ok = True
                            for nb, nh in mapped_neighbors:
                                e = (min(hv, nh), max(hv, nh))
                                if e in used_edges:
                                    ok = False
                                    break
                            if ok:
                                filtered.append(hv)
                        candidates = filtered
                
                if not candidates:
                    return False, None
                
                # Shuffle candidates for variety
                local_rng.shuffle(candidates)
                
                for hv in candidates:
                    # Check edges to already-mapped neighbors
                    valid = True
                    for nb in adj_t[tree_vertex]:
                        if pos_map[nb] < pos:
                            hn = mapping[nb]
                            e = (min(hv, hn), max(hv, hn))
                            if e in used_edges:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    # Tentatively assign
                    mapping[tree_vertex] = hv
                    used_hosts.add(hv)
                    
                    result, sol = backtrack(pos + 1, mapping, used_hosts)
                    if result:
                        return True, sol
                    
                    # Undo
                    mapping[tree_vertex] = -1
                    used_hosts.remove(hv)
                
                return False, None
            
            # Try multiple times with different shuffles
            for _ in range(100):
                mapping = [-1] * k
                used_hosts = set()
                result, sol = backtrack(0, mapping, used_hosts)
                if result:
                    return sol
            
            return None
        
        # Pack trees in order (largest first)
        for t_idx in order:
            if time.time() > deadline:
                break
            
            # Find a mapping for this tree
            mapping = find_mapping(t_idx)
            if mapping is None:
                # Try again with a fresh shuffle
                for _ in range(30):
                    mapping = find_mapping(t_idx)
                    if mapping is not None:
                        break
                if mapping is None:
                    continue  # Skip this tree, try next
            
            mappings[t_idx] = mapping
            
            # Mark edges as used
            for u, v in trees[t_idx]:
                hu = mapping[u]
                hv = mapping[v]
                e = (min(hu, hv), max(hu, hv))
                used_edges.add(e)
        
        # Calculate score
        score = 0
        for i in range(20):
            if mappings[i] is not None:
                score += tree_info[i]['num_edges']
        
        return score, mappings
    
    # Run multiple attempts
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
    
    # Ensure proper types
    result = []
    for i in range(20):
        if best_mappings[i] is not None:
            # Ensure it's a list of Python ints
            result.append([int(v) for v in best_mappings[i]])
        else:
            result.append(None)
    
    return result
=======
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
    deadline = start_time + 130  # 20s margin before 150s kill
    
    trees = instance_trees(20, 611)
    
    # Precompute tree properties
    tree_info = []
    for i in range(20):
        edges = trees[i]
        k = i + 1
        # Build adjacency
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
    
    # Sort by size descending for backtracking efficiency
    order = sorted(range(20), key=lambda i: -tree_info[i]['k'])
    
    # Get vertex ordering: highest degree first, then by index
    def get_vertex_order(adj, k, rng):
        """Get vertex ordering with highest degree first"""
        verts = list(range(k))
        deg = [len(adj[v]) for v in range(k)]
        # Sort by degree descending, break ties randomly for variety
        verts.sort(key=lambda v: (-deg[v], rng.random()))
        return verts
    
    # Precompute vertex orders for each tree
    vertex_orders = []
    for i in range(20):
        k = tree_info[i]['k']
        if k == 1:
            vertex_orders.append([0])
        else:
            vertex_orders.append(get_vertex_order(tree_info[i]['adj'], k, random.Random(i)))
    
    # Precompute position in vertex order for fast lookup
    vert_pos = []
    for i in range(20):
        k = tree_info[i]['k']
        pos = [-1] * k
        for j, v in enumerate(vertex_orders[i]):
            pos[v] = j
        vert_pos.append(pos)
    
    def try_pack(attempt_seed):
        """Try to pack all trees using a given random seed for tie-breaking"""
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * 20
        
        def find_mapping(t_idx, max_attempts=50):
            """
            Find a valid embedding of tree t_idx into K20
            using backtracking with pruning and forward checking.
            """
            info = tree_info[t_idx]
            k = info['k']
            adj_t = info['adj']
            v_order = vertex_orders[t_idx]
            pos_map = vert_pos[t_idx]
            
            if k == 1:
                return [0]
            
            # Precompute which vertices come before each position
            # For forward checking: when we place vertex at position p,
            # we need to check that all unplaced neighbors have available hosts
            
            def backtrack(pos, mapping, used_hosts, used_edges_local):
                if time.time() > deadline:
                    return False, None
                
                if pos == k:
                    # All vertices mapped - verify all edges are valid
                    for u, v in info['edges']:
                        hu = mapping[u]
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges_local:
                            return False, None
                    return True, mapping[:]
                
                tree_vertex = v_order[pos]
                
                # Get candidates: host vertices not yet used
                candidates = [hv for hv in range(20) if hv not in used_hosts]
                
                # Filter candidates based on edges to already-mapped neighbors
                mapped_neighbors = []
                for nb in adj_t[tree_vertex]:
                    if pos_map[nb] < pos:
                        mapped_neighbors.append((nb, mapping[nb]))
                
                if mapped_neighbors:
                    filtered = []
                    for hv in candidates:
                        ok = True
                        for nb, nh in mapped_neighbors:
                            e = (min(hv, nh), max(hv, nh))
                            if e in used_edges_local:
                                ok = False
                                break
                        if ok:
                            filtered.append(hv)
                    candidates = filtered
                
                if not candidates:
                    return False, None
                
                # Forward checking: for each candidate, verify that unplaced 
                # neighbors will have at least one available host
                # (simplified: just check that enough unused hosts remain)
                unplaced_count = k - pos - 1
                if len(candidates) - 1 < unplaced_count:
                    return False, None
                
                # Order candidates: prefer hosts with fewer conflicts
                # (heuristic: try hosts that have fewer used edges incident)
                def host_score(hv):
                    score = 0
                    for nh in used_hosts:
                        e = (min(hv, nh), max(hv, nh))
                        if e in used_edges_local:
                            score += 1
                    return score
                
                candidates.sort(key=lambda hv: (host_score(hv), local_rng.random()))
                
                for hv in candidates:
                    # Tentatively assign
                    mapping[tree_vertex] = hv
                    used_hosts.add(hv)
                    
                    # Add edges to used_edges_local
                    new_edges = set()
                    for nb in adj_t[tree_vertex]:
                        if pos_map[nb] < pos:
                            hn = mapping[nb]
                            e = (min(hv, hn), max(hv, hn))
                            new_edges.add(e)
                    
                    result, sol = backtrack(pos + 1, mapping, used_hosts, used_edges_local | new_edges)
                    if result:
                        return True, sol
                    
                    # Undo
                    mapping[tree_vertex] = -1
                    used_hosts.remove(hv)
                
                return False, None
            
            # Try multiple times with different shuffles
            for attempt in range(max_attempts):
                mapping = [-1] * k
                used_hosts = set()
                used_edges_local = set()
                result, sol = backtrack(0, mapping, used_hosts, used_edges_local)
                if result:
                    return sol
            
            return None
        
        # Pack trees in order (largest first)
        for t_idx in order:
            if time.time() > deadline:
                break
            
            # Find a mapping for this tree
            mapping = find_mapping(t_idx)
            if mapping is None:
                # Try again with a fresh shuffle
                for _ in range(20):
                    mapping = find_mapping(t_idx)
                    if mapping is not None:
                        break
                if mapping is None:
                    continue  # Skip this tree, try next
            
            mappings[t_idx] = mapping
            
            # Mark edges as used
            for u, v in trees[t_idx]:
                hu = mapping[u]
                hv = mapping[v]
                e = (min(hu, hv), max(hu, hv))
                used_edges.add(e)
        
        # Calculate score
        score = 0
        for i in range(20):
            if mappings[i] is not None:
                score += tree_info[i]['num_edges']
        
        return score, mappings
    
    # Run multiple attempts
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
    
    # Ensure proper types
    result = []
    for i in range(20):
        if best_mappings[i] is not None:
            # Ensure it's a list of Python ints
            result.append([int(v) for v in best_mappings[i]])
        else:
            result.append(None)
    
    return result
>>>>>>> REPLACE

