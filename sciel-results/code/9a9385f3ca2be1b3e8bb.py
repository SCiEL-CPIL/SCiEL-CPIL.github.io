import sys
import itertools

sys.setrecursionlimit(10000)

def construct():
    n = 20
    edges = []
    edge_index = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = len(edges)
            edges.append((u, v))
    
    m = len(edges)
    
    k33s = []
    k33_set = set()
    for combo in itertools.combinations(range(n), 6):
        subsets_3 = list(itertools.combinations(combo, 3))
        for i in range(len(subsets_3)):
            A = subsets_3[i]
            B = tuple(x for x in combo if x not in A)
            if min(A) < min(B):
                edge_indices = []
                for a in A:
                    for b in B:
                        if a < b:
                            idx = edge_index[(a, b)]
                        else:
                            idx = edge_index[(b, a)]
                        edge_indices.append(idx)
                edge_tuple = tuple(sorted(edge_indices))
                if edge_tuple not in k33_set:
                    k33_set.add(edge_tuple)
                    k33s.append(edge_tuple)
    
    num_k33s = len(k33s)
    
    edge_to_k33s = [[] for _ in range(m)]
    for k_idx, k33 in enumerate(k33s):
        for e_idx in k33:
            edge_to_k33s[e_idx].append(k_idx)
    
    best_count = 0
    best_edges = []
    
    def greedy():
        nonlocal best_count, best_edges
        counts_g = [0] * num_k33s
        forbidden_g = [False] * m
        selected_g = [False] * m
        count = 0
        for e_idx in range(m):
            if forbidden_g[e_idx]:
                continue
            valid = True
            affected = edge_to_k33s[e_idx]
            new_forb = []
            for k_idx in affected:
                if counts_g[k_idx] == 8:
                    valid = False
                    break
                elif counts_g[k_idx] == 7:
                    for e_k in k33s[k_idx]:
                        if e_k != e_idx and not selected_g[e_k]:
                            if not forbidden_g[e_k]:
                                new_forb.append(e_k)
                            break
            if valid:
                selected_g[e_idx] = True
                count += 1
                for k_idx in affected:
                    counts_g[k_idx] += 1
                for f in new_forb:
                    forbidden_g[f] = True
        
        if count > best_count:
            best_count = count
            best_edges = [edges[i] for i in range(m) if selected_g[i]]
            
    greedy()
    
    counts = [0] * num_k33s
    forbidden = [False] * m
    selected = [False] * m
    
    def b_b(idx, current_count):
        nonlocal best_count, best_edges
        
        # Calculate bound
        # remaining = m - idx
        # forbidden_remaining = sum(1 for i in range(idx, m) if forbidden[i])
        # bound = current_count + remaining - forbidden_remaining
        # Optimization: Precompute suffix sums of forbidden? 
        # Forbidden changes dynamically, so we can't precompute easily.
        # But m=190 is small.
        forbidden_remaining = 0
        for i in range(idx, m):
            if forbidden[i]:
                forbidden_remaining += 1
        
        bound = current_count + (m - idx) - forbidden_remaining
        if bound <= best_count:
            return
            
        if idx == m:
            if current_count > best_count:
                best_count = current_count
                best_edges = [edges[i] for i in range(m) if selected[i]]
            return
        
        e_idx = idx
        
        # Option 1: Exclude
        b_b(idx + 1, current_count)
        
        # Option 2: Include
        if not forbidden[e_idx]:
            valid = True
            affected = edge_to_k33s[e_idx]
            new_forb = []
            
            for k_idx in affected:
                if counts[k_idx] == 8:
                    valid = False
                    break
                elif counts[k_idx] == 7:
                    for e_k in k33s[k_idx]:
                        if e_k != e_idx and not selected[e_k]:
                            if not forbidden[e_k]:
                                new_forb.append(e_k)
                            break
            
            if valid:
                selected[e_idx] = True
                for k_idx in affected:
                    counts[k_idx] += 1
                for f in new_forb:
                    forbidden[f] = True
                    
                b_b(idx + 1, current_count + 1)
                
                selected[e_idx] = False
                for k_idx in affected:
                    counts[k_idx] -= 1
                for f in new_forb:
                    forbidden[f] = False

    b_b(0, 0)
    
    return [list(e) for e in best_edges]

