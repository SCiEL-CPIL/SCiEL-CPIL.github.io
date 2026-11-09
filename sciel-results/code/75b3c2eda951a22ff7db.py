import time
import random
import numpy as np

def popcount(x):
    return bin(x).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def bron_kerbosch(adj, n):
        cliques = []
        neighbors = adj
        
        def expand(R, P, X):
            if P == 0 and X == 0:
                cliques.append(R)
                return
            
            union = P | X
            if union:
                best_u = 0
                best_count = -1
                temp = union
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    count = popcount(P & neighbors[u])
                    if count > best_count:
                        best_count = count
                        best_u = u
                    temp ^= lsb
                u = best_u
                candidates = P & ~neighbors[u]
            else:
                candidates = P
            
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                expand(R | lsb, P & neighbors[v], X & neighbors[v])
                P ^= lsb
                X |= lsb
                temp ^= lsb

        expand(0, (1 << n) - 1, 0)
        return cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        relevant_cliques = [c for c in all_cliques if popcount(c) >= 2]
        
        if not relevant_cliques:
            return []

        m = len(relevant_cliques)
        
        vertex_cliques = [0] * n
        for i, clique in enumerate(relevant_cliques):
            temp = clique
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                vertex_cliques[v] |= (1 << i)
                temp ^= lsb

        clique_conflict = [0] * m
        for i in range(m):
            ci = relevant_cliques[i]
            clique_conflict[i] |= (1 << i)
            for j in range(i + 1, m):
                cj = relevant_cliques[j]
                if (ci & cj) != 0:
                    clique_conflict[i] |= (1 << j)
                    clique_conflict[j] |= (1 << i)

        best_size = n + 1
        best_sol = []
        
        def calc_lb(remaining_mask):
            if remaining_mask == 0:
                return 0
            count = 0
            mask = remaining_mask
            while mask:
                lsb = mask & -mask
                i = lsb.bit_length() - 1
                count += 1
                mask &= ~clique_conflict[i]
            return count

        def search(current_size, remaining_mask, chosen):
            nonlocal best_size, best_sol
            
            if current_size >= best_size:
                return
            
            if remaining_mask == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_sol = chosen[:]
                return
            
            lb = calc_lb(remaining_mask)
            if current_size + lb >= best_size:
                return
            
            min_count = n + 1
            target_clique_idx = -1
            temp_mask = remaining_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                i = lsb.bit_length() - 1
                cnt = popcount(relevant_cliques[i])
                if cnt < min_count:
                    min_count = cnt
                    target_clique_idx = i
                temp_mask ^= lsb
            
            if target_clique_idx == -1:
                return

            target_clique = relevant_cliques[target_clique_idx]
            
            temp = target_clique
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                
                new_remaining = remaining_mask & ~vertex_cliques[v]
                new_chosen = chosen + [v]
                
                search(current_size + 1, new_remaining, new_chosen)
                
                temp ^= lsb

        search(0, (1 << m) - 1, [])
        
        if best_sol:
            return sorted(best_sol)
        
        # Fallback
        remaining = [set(c) for c in relevant_cliques]
        selected = []
        while any(remaining):
            counts = [0]*n
            for c in remaining:
                if c:
                    for v in c:
                        counts[v] += 1
            max_c = max(counts)
            if max_c == 0: break
            candidates = [v for v in range(n) if counts[v] == max_c]
            v = min(candidates)
            selected.append(v)
            for i, c in enumerate(remaining):
                if v in c:
                    remaining[i].discard(v)
                    if not remaining[i]:
                        remaining[i] = set()
        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sorted(sol))
        
        if time.time() > deadline:
             # Fill remaining with greedy quickly if we run out of time
             # But we should have enough time.
             break
             
    # If loop broke early, ensure we have 3 results
    # This shouldn't happen with the time allocation above, but safe.
    while len(results) < 3:
         results.append([])
         
    return results
