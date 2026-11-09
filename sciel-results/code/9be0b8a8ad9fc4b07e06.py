import time
import random
from random import Random
from collections import Counter

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
    deadline = start_time + 145
    
    trees = instance_trees()
    N_HOSTS = 20
    
    # Precompute tree structures
    tree_info = []
    for i in range(20):
        k = i + 1
        if k <= 2:
            tree_info.append({'k': k, 'edges': trees[i], 'adj': {}, 'leaves': []})
            continue
        adj = {v: [] for v in range(k)}
        for u, v in trees[i]:
            adj[u].append(v)
            adj[v].append(u)
        leaves = [v for v in range(k) if len(adj[v]) == 1]
        tree_info.append({'k': k, 'edges': trees[i], 'adj': adj, 'leaves': leaves})
    
    def verify_result(result):
        """Verify a packing result is valid and complete."""
        if result is None:
            return False
        used = set()
        total_edges = 0
        for i in range(20):
            k = i + 1
            if k == 1:
                continue
            mapping = result[i]
            if mapping is None or len(mapping) != k or len(set(mapping)) != k:
                return False
            for h in mapping:
                if not isinstance(h, int) or isinstance(h, bool) or h < 0 or h >= 20:
                    return False
            for u, v in trees[i]:
                e1 = min(mapping[u], mapping[v])
                e2 = max(mapping[u], mapping[v])
                if (e1, e2) in used:
                    return False
                used.add((e1, e2))
                total_edges += 1
        return total_edges == 190
    
    # Fast conflict computation: returns set of used edges and count of conflicts
    def compute_conflicts(state):
        used = set()
        conflicts = 0
        for i in range(20):
            k = i + 1
            if k <= 2:
                continue
            mapping = state[i]
            for u, v in trees[i]:
                e1 = min(mapping[u], mapping[v])
                e2 = max(mapping[u], mapping[v])
                if (e1, e2) in used:
                    conflicts += 1
                else:
                    used.add((e1, e2))
        return conflicts, used
    
    # Try backtracking with a few seeds
    def try_backtrack(seed, time_limit=8):
        bt_start = time.time()
        bt_deadline = bt_start + time_limit
        rng = Random(seed)
        
        order = list(range(19, -1, -1))
        reordered_info = [tree_info[i] for i in order]
        
        used_edges = set()
        mappings_reordered = [None] * 20
        remaining_hosts = set(range(N_HOSTS))
        
        def place_reordered(idx, used_edges, remaining_hosts, mappings_reordered, order, tree_info_list):
            if time.time() > bt_deadline:
                return None
            
            if idx == 20:
                final_mappings = [None] * 20
                for pos, orig_idx in enumerate(order):
                    final_mappings[orig_idx] = mappings_reordered[pos]
                return final_mappings
            
            orig_idx = order[idx]
            info = tree_info_list[idx]
            k = info['k']
            
            if k == 1:
                hosts = list(remaining_hosts)
                rng.shuffle(hosts)
                for host in hosts:
                    new_remaining = remaining_hosts - {host}
                    mappings_reordered[idx] = [host]
                    result = place_reordered(idx + 1, used_edges, new_remaining, mappings_reordered, order, tree_info_list)
                    if result is not None:
                        return result
                    mappings_reordered[idx] = None
                return None
            
            adj = info['adj']
            edges = info['edges']
            
            def embed_tree_local(used_edges, remaining_hosts, adj, k, edges):
                if time.time() > bt_deadline:
                    return None
                
                root = max(range(k), key=lambda v: len(adj[v]))
                
                def dfs_assign(partial_map, remaining):
                    if time.time() > bt_deadline:
                        return None
                    
                    if len(partial_map) == k:
                        for u, v in edges:
                            e1 = min(partial_map[u], partial_map[v])
                            e2 = max(partial_map[u], partial_map[v])
                            if (e1, e2) in used_edges:
                                return None
                        mapping = [0] * k
                        for tv, h in partial_map.items():
                            mapping[tv] = h
                        return mapping
                    
                    unassigned = [v for v in range(k) if v not in partial_map]
                    if not unassigned:
                        return None
                    
                    scored = []
                    for v in unassigned:
                        assigned_neigh = [n for n in adj[v] if n in partial_map]
                        scored.append((len(assigned_neigh), rng.random(), v))
                    scored.sort(reverse=True)
                    
                    next_v = scored[0][2]
                    
                    host_scores = []
                    for host in remaining:
                        conflict_count = 0
                        for n in adj[next_v]:
                            if n in partial_map:
                                e1 = min(host, partial_map[n])
                                e2 = max(host, partial_map[n])
                                if (e1, e2) in used_edges:
                                    conflict_count += 1
                        host_scores.append((conflict_count, rng.random(), host))
                    host_scores.sort()
                    
                    for _, _, host in host_scores:
                        valid = True
                        for n in adj[next_v]:
                            if n in partial_map:
                                e1 = min(host, partial_map[n])
                                e2 = max(host, partial_map[n])
                                if (e1, e2) in used_edges:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        
                        new_partial = dict(partial_map)
                        new_partial[next_v] = host
                        new_remaining = remaining - {host}
                        
                        result = dfs_assign(new_partial, new_remaining)
                        if result is not None:
                            return result
                    
                    return None
                
                root_hosts = list(remaining_hosts)
                rng.shuffle(root_hosts)
                for host in root_hosts:
                    partial = {root: host}
                    remaining_after = remaining_hosts - {host}
                    result = dfs_assign(partial, remaining_after)
                    if result is not None:
                        return result
                
                return None
            
            embedding = embed_tree_local(used_edges, remaining_hosts, adj, k, edges)
            if embedding is None:
                return None
            
            mappings_reordered[idx] = embedding
            new_used = set(used_edges)
            for u, v in edges:
                e1 = min(embedding[u], embedding[v])
                e2 = max(embedding[u], embedding[v])
                new_used.add((e1, e2))
            
            new_remaining = remaining_hosts - set(embedding)
            
            result = place_reordered(idx + 1, new_used, new_remaining, mappings_reordered, order, tree_info_list)
            if result is not None:
                return result
            
            return None
        
        return place_reordered(0, used_edges, remaining_hosts, mappings_reordered, order, reordered_info)
    
    # Try backtracking with a few seeds
    for seed in range(5):
        if time.time() > deadline - 60:
            break
        result = try_backtrack(seed, time_limit=8)
        if result is not None and verify_result(result):
            return result
    
    # Simulated annealing
    state = [None] * 20
    for i in range(20):
        k = i + 1
        if k == 1:
            state[i] = [random.randint(0, 19)]
        else:
            hosts = list(range(N_HOSTS))
            random.shuffle(hosts)
            state[i] = hosts[:k]
    
    current_conflicts, _ = compute_conflicts(state)
    best_state = [s[:] if s else None for s in state]
    best_conflicts = current_conflicts
    
    if best_conflicts == 0:
        return best_state
    
    temperature = 1.0
    cooling = 0.995
    min_temp = 1e-4
    
    while time.time() < deadline - 5:
        if current_conflicts == 0:
            break
        
        i = random.randint(0, 19)
        k = i + 1
        if k == 1:
            continue
        
        old_mapping = state[i][:]
        
        strategy = random.random()
        
        if strategy < 0.4:
            new_mapping = old_mapping[:]
            a, b = random.sample(range(k), 2)
            new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
        elif strategy < 0.7:
            new_mapping = old_mapping[:]
            a = random.randint(0, k - 1)
            used_hosts = set(new_mapping)
            available = [h for h in range(N_HOSTS) if h not in used_hosts]
            if available:
                new_mapping[a] = random.choice(available)
            else:
                b = random.randint(0, k - 1)
                new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
        elif strategy < 0.85:
            new_mapping = old_mapping[:]
            a = random.randint(0, k - 1)
            used_hosts = set(new_mapping)
            available = [h for h in range(N_HOSTS) if h not in used_hosts]
            if available:
                new_mapping[a] = random.choice(available)
            else:
                b = random.randint(0, k - 1)
                new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
        else:
            hosts = list(range(N_HOSTS))
            random.shuffle(hosts)
            new_mapping = hosts[:k]
        
        state[i] = new_mapping
        new_conflicts, _ = compute_conflicts(state)
        
        delta = new_conflicts - current_conflicts
        
        if delta < 0 or random.random() < pow(2.718, -delta / max(temperature, 1e-9)):
            current_conflicts = new_conflicts
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_state = [s[:] if s else None for s in state]
        else:
            state[i] = old_mapping
        
        temperature *= cooling
        if temperature < min_temp:
            temperature = min_temp
    
    # Final polish
    if best_conflicts > 0 and time.time() < deadline - 2:
        polish_deadline = min(deadline, time.time() + 5)
        for _ in range(5000):
            if time.time() > polish_deadline:
                break
            if best_conflicts == 0:
                break
            i = random.randint(0, 19)
            k = i + 1
            if k == 1:
                continue
            old_mapping = best_state[i][:]
            new_mapping = old_mapping[:]
            a = random.randint(0, k - 1)
            used_hosts = set(new_mapping)
            available = [h for h in range(N_HOSTS) if h not in used_hosts]
            if available:
                new_mapping[a] = random.choice(available)
            else:
                b = random.randint(0, k - 1)
                new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
            
            test_state = best_state[:]
            test_state[i] = new_mapping
            test_conflicts, _ = compute_conflicts(test_state)
            if test_conflicts < best_conflicts:
                best_state[i] = new_mapping
                best_conflicts = test_conflicts
    
    return best_state
