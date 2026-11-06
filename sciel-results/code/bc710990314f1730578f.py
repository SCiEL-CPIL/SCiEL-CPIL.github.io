import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140
    results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 3:
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = _instance(*spec)
        paths = _solve_instance(edges, deadline)
        results.append(paths)
        
    return results

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _solve_instance(edges, deadline):
    if not edges:
        return []
        
    # Build adjacency
    adj = defaultdict(set)
    degree = defaultdict(int)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1
        
    odd_vertices = [v for v in adj if degree[v] % 2 == 1]
    k = len(odd_vertices) // 2
    
    if k == 0:
        # Eulerian graph - use greedy with merge
        paths = _greedy_decompose(edges, deadline)
        paths = _merge_paths(paths, edges, deadline)
        return paths

    # Local search on pairings
    best_paths = None
    best_count = float('inf')
    
    # Initial random pairing
    perm = odd_vertices[:]
    random.shuffle(perm)
    pairs = [(perm[i], perm[i+1]) for i in range(0, len(perm), 2)]
    
    current_paths = _solve_with_pairs(edges, pairs, deadline)
    if current_paths is not None:
        best_paths = current_paths
        best_count = len(current_paths)
    
    # Local search: try 2-opt swaps on pairings
    max_iters = 500
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        if best_count == k:
            break  # Can't do better than k
            
        # Pick two pairs to swap
        i = random.randint(0, len(pairs) - 1)
        j = random.randint(0, len(pairs) - 1)
        if i == j:
            continue
            
        # Swap: (a,b) and (c,d) -> (a,c) and (b,d) or (a,d) and (b,c)
        a, b = pairs[i]
        c, d = pairs[j]
        
        # Try both swap options
        for new_pairs in [
            [(a, c), (b, d)],
            [(a, d), (b, c)]
        ]:
            old_i, old_j = pairs[i], pairs[j]
            pairs[i], pairs[j] = new_pairs
            
            new_paths = _solve_with_pairs(edges, pairs, deadline)
            
            if new_paths is not None:
                new_count = len(new_paths)
                if new_count < best_count:
                    best_count = new_count
                    best_paths = new_paths
                    if best_count == k:
                        break
                else:
                    # Revert
                    pairs[i], pairs[j] = old_i, old_j
            else:
                # Revert
                pairs[i], pairs[j] = old_i, old_j
                
        if best_count == k:
            break
    
    if best_paths is not None:
        merged_paths = _merge_paths(best_paths, edges, deadline)
        return merged_paths
    else:
        # Fallback to greedy
        paths = _greedy_decompose(edges, deadline)
        paths = _merge_paths(paths, edges, deadline)
        return paths

def _solve_with_pairs(edges, pairs, deadline):
    # Add virtual edges
    virtual_edges = set(tuple(sorted(p)) for p in pairs)
    
    # Build full graph including virtual edges
    full_edges = set(tuple(sorted(e)) for e in edges)
    full_edges.update(virtual_edges)
    
    # Find Eulerian Circuit
    circuit = _find_eulerian_circuit(full_edges)
    if circuit is None:
        return None
        
    # Split circuit by virtual edges
    paths = []
    current_path = []
    for i in range(len(circuit) - 1):
        u, v = circuit[i], circuit[i+1]
        edge = tuple(sorted((u, v)))
        current_path.append(u)
        if edge in virtual_edges:
            current_path.append(v)
            # Check if current_path is simple
            if len(set(current_path)) == len(current_path):
                paths.append(current_path)
            else:
                return None # Failed simplicity check
            current_path = []
            
    if current_path:
        current_path.append(circuit[-1])
        if len(set(current_path)) == len(current_path):
            paths.append(current_path)
        else:
            return None
            
    if len(paths) != len(pairs):
        return None
        
    return paths

def _find_eulerian_circuit(edges):
    if not edges:
        return []
        
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Sort adjacency lists for deterministic behavior
    for u in adj:
        adj[u].sort()
        
    visited_edges = set()
    start = min(adj.keys())
    
    stack = [start]
    circuit = []
    
    while stack:
        u = stack[-1]
        if adj[u]:
            v = adj[u].pop()
            edge = tuple(sorted((u, v)))
            if edge not in visited_edges:
                visited_edges.add(edge)
                stack.append(v)
        else:
            circuit.append(stack.pop())
            
    result = circuit[::-1]
    
    # Verify all edges are in the circuit
    circuit_edges = set()
    for i in range(len(result) - 1):
        circuit_edges.add(tuple(sorted((result[i], result[i+1]))))
    if circuit_edges != edges:
        return None
        
    return result

def _greedy_decompose(edges, deadline):
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    degree = defaultdict(int)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1
        
    paths = []
    
    while remaining:
        if time.time() > deadline:
            break
        # Pick start
        odd_vertices = [v for v in adj if degree.get(v,0) % 2 == 1 and adj[v]]
        if odd_vertices:
            start = random.choice(odd_vertices)
        else:
            start = max(adj.keys(), key=lambda x: degree.get(x,0))
            
        # Find long path
        path = [start]
        visited = {start}
        current = start
        while True:
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                break
            # Warnsdorff
            neighbors.sort(key=lambda x: sum(1 for y in adj[x] if y not in visited))
            next_v = neighbors[0]
            path.append(next_v)
            visited.add(next_v)
            
            # Remove edge
            edge = tuple(sorted((current, next_v)))
            remaining.discard(edge)
            adj[current].discard(next_v)
            adj[next_v].discard(current)
            degree[current] -= 1
            degree[next_v] -= 1
            
            if not adj[current]: del adj[current]
            if not adj[next_v]: del adj[next_v]
            
            current = next_v
            
        paths.append(path)
        
    return paths

def _merge_paths(paths, edges, deadline):
    # Try to merge paths that share endpoints
    # If path A ends at u and path B starts at u, and A and B share no other vertices,
    # we can merge them into A + B (excluding the duplicate u).
    # The merged path is simple if A and B are disjoint except for u.
    
    # We'll do this iteratively until no more merges are possible.
    merged = True
    while merged:
        merged = False
        new_paths = []
        used = [False] * len(paths)
        
        for i in range(len(paths)):
            if used[i]:
                continue
                
            current_path = paths[i]
            start_v = current_path[0]
            end_v = current_path[-1]
            
            # Try to extend from end_v or start_v
            best_merge = None
            best_j = -1
            
            for j in range(len(paths)):
                if i == j or used[j]:
                    continue
                    
                other_path = paths[j]
                other_start = other_path[0]
                other_end = other_path[-1]
                
                # Case 1: current_path ends at u, other_path starts at u
                if end_v == other_start:
                    # Check if they share any other vertices
                    set_a = set(current_path[:-1])
                    set_b = set(other_path)
                    if not set_a & set_b:
                        # Merge: current_path + other_path[1:]
                        merged_path = current_path + other_path[1:]
                        if best_merge is None or len(merged_path) > len(best_merge):
                            best_merge = merged_path
                            best_j = j
                            
                # Case 2: current_path ends at u, other_path ends at u
                # We can reverse other_path and merge
                if end_v == other_end:
                    reversed_other = other_path[::-1]
                    set_a = set(current_path[:-1])
                    set_b = set(reversed_other)
                    if not set_a & set_b:
                        merged_path = current_path + reversed_other[1:]
                        if best_merge is None or len(merged_path) > len(best_merge):
                            best_merge = merged_path
                            best_j = j
                            
                # Case 3: current_path starts at u, other_path starts at u
                if start_v == other_start:
                    # We can reverse current_path and merge
                    reversed_current = current_path[::-1]
                    set_a = set(reversed_current[:-1])
                    set_b = set(other_path)
                    if not set_a & set_b:
                        merged_path = reversed_current + other_path[1:]
                        if best_merge is None or len(merged_path) > len(best_merge):
                            best_merge = merged_path
                            best_j = j
                            
                # Case 4: current_path starts at u, other_path ends at u
                if start_v == other_end:
                    reversed_other = other_path[::-1]
                    set_a = set(current_path)
                    set_b = set(reversed_other[:-1])
                    if not set_a & set_b:
                        merged_path = reversed_other + current_path[1:]
                        if best_merge is None or len(merged_path) > len(best_merge):
                            best_merge = merged_path
                            best_j = j
                            
            if best_merge is not None:
                new_paths.append(best_merge)
                used[i] = True
                used[best_j] = True
                merged = True
            else:
                new_paths.append(current_path)
                used[i] = True
                
        paths = new_paths
        if time.time() > deadline:
            break
            
    return paths
