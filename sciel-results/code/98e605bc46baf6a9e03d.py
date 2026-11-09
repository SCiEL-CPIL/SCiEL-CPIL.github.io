import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s kill
    n = 20
    best_edges = 0
    best_adj = None
    
    # Precompute triples and their indices
    triples = []
    triple_indices = [[] for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
                triple_indices[i].append(idx)
                triple_indices[j].append(idx)
                triple_indices[k].append(idx)
                idx += 1
                
    def get_popcount(x):
        return x.bit_count()

    def count_violations(adj):
        count = 0
        for i, j, k in triples:
            if get_popcount(adj[i] & adj[j] & adj[k]) >= 3:
                count += 1
        return count

    # Initialize with random graph
    adj = [0] * n
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
    curr_violations = count_violations(adj)
    curr_edges = sum(get_popcount(x) for x in adj) // 2
    
    # If it's already valid, update best
    if curr_violations == 0 and curr_edges > best_edges:
        best_edges = curr_edges
        best_adj = adj[:]

    T = 10.0
    T_min = 0.01
    alpha = 0.995
    penalty = 10.0
    
    # Simulated Annealing
    while time.time() < deadline - 10.0:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        is_present = (adj[u] >> v) & 1
        adding = not is_present
        
        # Get affected triple indices
        aff_idx = set()
        for t_idx in triple_indices[u]:
            aff_idx.add(t_idx)
        for t_idx in triple_indices[v]:
            aff_idx.add(t_idx)
            
        # Calculate old violations in affected set
        old_viol = 0
        for t_idx in aff_idx:
            i, j, k = triples[t_idx]
            if get_popcount(adj[i] & adj[j] & adj[k]) >= 3:
                old_viol += 1
        
        # Apply change
        if adding:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
        # Calculate new violations in affected set
        new_viol = 0
        for t_idx in aff_idx:
            i, j, k = triples[t_idx]
            if get_popcount(adj[i] & adj[j] & adj[k]) >= 3:
                new_viol += 1
                
        delta_viol = new_viol - old_viol
        new_violations = curr_violations + delta_viol
        
        delta_edges = 1 if adding else -1
        delta_E = -delta_edges + penalty * delta_viol
        
        accept = False
        if curr_violations == 0 and new_violations == 0:
            if delta_edges > 0:
                accept = True
            elif delta_edges == 0:
                accept = True
        elif curr_violations > 0 and new_violations == 0:
            accept = True
        elif curr_violations == 0 and new_violations > 0:
            accept = False
        else:
            # Both invalid
            if delta_E < 0:
                accept = True
            else:
                if random.random() < math.exp(-delta_E / T):
                    accept = True

        if accept:
            curr_violations = new_violations
            curr_edges += delta_edges
            if curr_violations == 0 and curr_edges > best_edges:
                best_edges = curr_edges
                best_adj = adj[:]
        else:
            # Revert
            if adding:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        T *= alpha
        if T < T_min:
            T = T_min

    # Final Polish: Greedy add on best_adj
    if best_adj:
        adj = best_adj[:]
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        # Check validity of affected triples
                        aff_idx = set()
                        for t_idx in triple_indices[u]:
                            aff_idx.add(t_idx)
                        for t_idx in triple_indices[v]:
                            aff_idx.add(t_idx)
                        
                        valid = True
                        for t_idx in aff_idx:
                            i, j, k = triples[t_idx]
                            if get_popcount(adj[i] & adj[j] & adj[k]) >= 3:
                                valid = False
                                break
                        
                        if valid:
                            changed = True
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
        
        ec = sum(get_popcount(x) for x in adj) // 2
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]

    if best_adj:
        return [[u, v] for u in range(n) for v in range(u+1, n) if (best_adj[u] >> v) & 1]
    else:
        return []
