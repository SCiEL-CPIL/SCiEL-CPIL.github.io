import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    n_target = 33
    
    def build_adj_list(adj, n):
        adj_list = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i,j]:
                    adj_list[i].add(j)
                    adj_list[j].add(i)
        return adj_list
    
    def has_c5(adj_list, n):
        for a in range(n):
            for b in adj_list[a]:
                if b <= a: continue
                for c in adj_list[b]:
                    if c <= b: continue
                    for d in adj_list[c]:
                        if d <= c: continue
                        if d in adj_list[a]:
                            # Found a 4-cycle a-b-c-d-a, but we need 5-cycle
                            # Actually, this is a 4-cycle, not 5
                            pass
        # This approach is wrong. Let me use the direct enumeration.
        for a in range(n):
            for b in range(a+1, n):
                if b not in adj_list[a]: continue
                for c in range(b+1, n):
                    if c not in adj_list[b]: continue
                    for d in range(c+1, n):
                        if d not in adj_list[c]: continue
                        for e in range(d+1, n):
                            if e in adj_list[d] and a in adj_list[e]:
                                return True
        return False
    
    def find_indep_set(adj_list, n, k):
        def backtrack(remaining, current_size):
            if current_size == k:
                return True
            if len(remaining) < k - current_size:
                return False
            # Order by degree (greedy heuristic)
            remaining_sorted = sorted(remaining, key=lambda v: len(adj_list[v]))
            for i, v in enumerate(remaining_sorted):
                new_remaining = [u for u in remaining_sorted[i+1:] if u not in adj_list[v]]
                if backtrack(new_remaining, current_size + 1):
                    return True
            return False
        return backtrack(list(range(n)), 0)
    
    # Try random constructions
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Start with 8 K4s + 1 vertex
        adj = np.zeros((n_target, n_target), dtype=bool)
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i+1, 4):
                    adj[base+i, base+j] = True
                    adj[base+j, base+i] = True
        
        # Random edges from vertex 32
        for b in range(8):
            base = b * 4
            for i in range(4):
                if random.random() < 0.6:
                    adj[32, base+i] = True
                    adj[base+i, 32] = True
        
        # Random edges between K4s (sparse)
        for _ in range(10):
            i = random.randint(0, 31)
            j = random.randint(0, 31)
            if i != j and i // 4 != j // 4:
                adj[i, j] = True
                adj[j, i] = True
        
        adj_list = build_adj_list(adj, n_target)
        
        if has_c5(adj_list, n_target):
            continue
        
        if not find_indep_set(adj_list, n_target, 9):
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if adj[i,j]:
                        edges.append([i, j])
            return {'n': n_target, 'edges': edges}
    
    # Fallback: 8 K4s
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': 32, 'edges': edges}
