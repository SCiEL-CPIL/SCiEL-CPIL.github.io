import time
import random
from random import Random

def instance_trees(n=20, seed=611):
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
    start_time = time.time()
    deadline = start_time + 115.0
    
    trees = instance_trees()
    N_HOSTS = 20
    
    # Precompute tree data
    tree_data = []
    for i in range(20):
        k = i + 1
        if k == 1:
            tree_data.append({
                'k': 1,
                'edges': [],
                'adj': {0: []},
                'deg': [0]
            })
            continue
        adj = {v: [] for v in range(k)}
        deg = [0] * k
        for u, v in trees[i]:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_data.append({
            'k': k,
            'edges': trees[i],
            'adj': adj,
            'deg': deg
        })

    # Sort trees by complexity (larger first, higher max degree first)
    order = list(range(19, -1, -1))
    
    def solve_backtrack(order, time_limit):
        used_edges = set()
        mappings = [None] * 20
        remaining_hosts = set(range(N_HOSTS))
        
        def backtrack(idx, used_edges, remaining_hosts, mappings):
            if time.time() > time_limit:
                return False
            
            if idx == len(order):
                return True
            
            orig_idx = order[idx]
            info = tree_data[orig_idx]
            k = info['k']
            
            if k == 1:
                # Try any remaining host
                for host in list(remaining_hosts):
                    mappings[orig_idx] = [host]
                    new_remaining = remaining_hosts - {host}
                    if backtrack(idx + 1, used_edges, new_remaining, mappings):
                        return True
                    mappings[orig_idx] = None
                return False
            
            # Embedding search for this tree
            adj = info['adj']
            edges = info['edges']
            
            # Choose root: highest degree
            root = max(range(k), key=lambda v: info['deg'][v])
            
            # Recursive embedding
            def embed_vertex(v, partial_map, remaining):
                if time.time() > time_limit:
                    return False
                
                if len(partial_map) == k:
                    # Check all edges
                    for u, v in edges:
                        h1 = partial_map[u]
                        h2 = partial_map[v]
                        e = (min(h1, h2), max(h1, h2))
                        if e in used_edges:
                            return False
                    # Valid
                    mappings[orig_idx] = [partial_map[i] for i in range(k)]
                    new_used = set(used_edges)
                    for u, v in edges:
                        h1 = partial_map[u]
                        h2 = partial_map[v]
                        new_used.add((min(h1, h2), max(h1, h2)))
                    new_remaining = remaining - set(partial_map.values())
                    
                    if backtrack(idx + 1, new_used, new_remaining, mappings):
                        return True
                    mappings[orig_idx] = None
                    return False
                
                # Find next vertex to assign
                # Heuristic: unassigned vertex with most assigned neighbors
                best_v = -1
                max_assigned_neigh = -1
                for v in range(k):
                    if v in partial_map:
                        continue
                    count = 0
                    for n in adj[v]:
                        if n in partial_map:
                            count += 1
                    if count > max_assigned_neigh:
                        max_assigned_neigh = count
                        best_v = v
                
                if best_v == -1:
                    return False
                
                # Try hosts for best_v
                # Heuristic: try hosts that are "less constrained" or random
                # Let's try random order for diversification in restarts
                hosts_to_try = list(remaining)
                random.shuffle(hosts_to_try)
                
                for host in hosts_to_try:
                    # Check if assigning host to best_v creates immediate conflict
                    # with already assigned neighbors
                    valid = True
                    for n in adj[best_v]:
                        if n in partial_map:
                            h_n = partial_map[n]
                            e = (min(host, h_n), max(host, h_n))
                            if e in used_edges:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    partial_map[best_v] = host
                    if embed_vertex(best_v, partial_map, remaining - {host}):
                        return True
                    del partial_map[best_v]
                
                return False
            
            # Start embedding
            # Try all hosts for root
            hosts_for_root = list(remaining_hosts)
            random.shuffle(hosts_for_root)
            
            for host in hosts_for_root:
                partial_map = {root: host}
                if embed_vertex(root, partial_map, remaining_hosts - {host}):
                    return True
            
            return False

        return backtrack(0, used_edges, remaining_hosts, mappings)

    # Strategy: Multiple randomized restarts
    best_solution = None
    rng = Random(12345) # Fixed seed for reproducibility of the restarts themselves? Or use system random?
    # Using system random is better for diversity if the first few fail.
    
    max_restarts = 20
    restart_time_budget = (deadline - time.time()) / max_restarts
    
    for i in range(max_restarts):
        time_limit = time.time() + restart_time_budget
        if time.time() > deadline - 5:
            break
        
        # Shuffle the order of trees slightly? 
        # Largest first is usually best. Let's keep it.
        # But we can shuffle the "tie-breaking" in vertex selection by shuffling adj lists?
        # The random.shuffle in embed_vertex handles host ordering.
        
        result = solve_backtrack(order, time_limit)
        if result:
            # Verify
            used = set()
            valid = True
            for i in range(20):
                k = i + 1
                if k == 1:
                    continue
                m = mappings[i]
                if not m or len(set(m)) != k:
                    valid = False
                    break
                for u, v in trees[i]:
                    e = (min(m[u], m[v]), max(m[u], m[v]))
                    if e in used:
                        valid = False
                        break
                    used.add(e)
            if valid:
                return mappings
        
    # If backtracking fails, fallback to a very fast local search (Simulated Annealing)
    # This is just a safety net.
    
    def local_search():
        # Initialize
        state = [None] * 20
        for i in range(20):
            k = i + 1
            hosts = list(range(20))
            random.shuffle(hosts)
            state[i] = hosts[:k]
        
        def score(state):
            used = set()
            conflicts = 0
            for i in range(20):
                k = i + 1
                if k == 1: continue
                m = state[i]
                for u, v in trees[i]:
                    e = (min(m[u], m[v]), max(m[u], m[v]))
                    if e in used:
                        conflicts += 1
                    else:
                        used.add(e)
            return conflicts
        
        current_score = score(state)
        best_state = [s[:] for s in state]
        best_score = current_score
        
        t = 10.0
        for step in range(50000):
            if time.time() > deadline - 1:
                break
            if current_score == 0:
                break
            
            i = random.randint(0, 19)
            k = i + 1
            if k == 1: continue
            
            old = state[i][:]
            # Move: swap two vertices in tree i
            a, b = random.sample(range(k), 2)
            state[i][a], state[i][b] = state[i][b], state[i][a]
            
            new_score = score(state)
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.718, -delta / t):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_state = [s[:] for s in state]
            else:
                state[i] = old
            
            t *= 0.999
            if t < 0.1: t = 0.1
        
        return best_state

    result = local_search()
    # Verify local search result
    used = set()
    valid = True
    for i in range(20):
        k = i + 1
        if k == 1: continue
        m = result[i]
        if not m or len(set(m)) != k:
            valid = False
            break
        for u, v in trees[i]:
            e = (min(m[u], m[v]), max(m[u], m[v]))
            if e in used:
                valid = False
                break
            used.add(e)
    
    if valid:
        return result
        
    # Return whatever we have (likely invalid, but must return something)
    return result

