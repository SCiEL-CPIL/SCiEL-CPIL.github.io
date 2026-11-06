import time
import random
import sys

def construct():
    d = 7
    n_vertices = 1 << d
    
    # Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    n_edges = len(all_edges)
    
    # Precompute conflict graph
    # Two edges conflict if they are in the same 2D face
    conflicts = [[] for _ in range(n_edges)]
    
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            # Three other edges in this face
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            for e in (e1, e2, e3):
                c_idx = edge_to_idx[e]
                if c_idx != idx:
                    conflicts[idx].append(c_idx)
    
    # Remove duplicates
    for i in range(n_edges):
        conflicts[i] = list(set(conflicts[i]))
    
    def greedy_construct(order):
        in_set = [False] * n_edges
        conflict_count = [0] * n_edges
        count = 0
        for idx in order:
            if conflict_count[idx] == 0:
                in_set[idx] = True
                count += 1
                for c in conflicts[idx]:
                    conflict_count[c] += 1
        return in_set, conflict_count, count
    
    def local_search(in_set, conflict_count, count, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any available edge
            for idx in range(n_edges):
                if not in_set[idx] and conflict_count[idx] == 0:
                    in_set[idx] = True
                    count += 1
                    for c in conflicts[idx]:
                        conflict_count[c] += 1
                    improved = True
            
            if improved:
                continue
            
            # Try 1-out, k-in moves
            best_gain = 0
            best_move = None
            
            for u in range(n_edges):
                if not in_set[u]:
                    continue
                
                # Find candidates that become available if we remove u
                candidates = []
                for v in conflicts[u]:
                    if not in_set[v] and conflict_count[v] == 1:
                        candidates.append(v)
                
                if len(candidates) <= 1:
                    continue
                
                # Simulate: remove u, greedily add candidates
                new_conf = conflict_count[:]
                for c in conflicts[u]:
                    new_conf[c] -= 1
                
                new_count = count - 1
                added = set()
                
                # Sort by number of conflicts (add less conflicting first)
                candidates.sort(key=lambda v: len([c for c in conflicts[v] if not in_set[c] or c == u]))
                
                for v in candidates:
                    if v in added:
                        continue
                    if new_conf[v] == 0:
                        new_count += 1
                        added.add(v)
                        for c in conflicts[v]:
                            new_conf[c] += 1
                
                gain = new_count - count
                if gain > best_gain:
                    best_gain = gain
                    best_move = (u, list(added), new_conf, new_count)
                    if best_gain >= 2:
                        break
            
            if best_gain > 0 and best_move:
                u, added, new_conf, new_count = best_move
                in_set[u] = False
                for c in conflicts[u]:
                    conflict_count[c] -= 1
                for v in added:
                    in_set[v] = True
                    for c in conflicts[v]:
                        conflict_count[c] += 1
                conflict_count = new_conf
                count = new_count
                improved = True
        
        return in_set, conflict_count, count
    
    deadline = time.time() + 110
    best_count = 0
    best_in_set = None
    
    # Multiple random restarts
    while time.time() < deadline:
        # Random order with some structure: shuffle dimensions
        order = list(range(n_edges))
        random.shuffle(order)
        
        in_set, conflict_count, count = greedy_construct(order)
        in_set, conflict_count, count = local_search(in_set, conflict_count, count, deadline)
        
        if count > best_count:
            best_count = count
            best_in_set = in_set[:]
        
        if best_count >= 308:
            break
    
    # Convert to edge list
    result = []
    for i in range(n_edges):
        if best_in_set[i]:
            result.append(list(all_edges[i]))
    
    return result
