import time
import math
import sys
from collections import Counter

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit (evaluator kills at 90s)
    
    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set
    
    # Precompute LCMs
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Precompute bad triples and edge weights
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    edge_weight = [[0.0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lab == lcm(a, c) == lcm(b, c):
                    conflict[a][b] |= (1 << c)
                    conflict[b][a] |= (1 << c)
                    conflict[a][c] |= (1 << b)
                    conflict[c][a] |= (1 << b)
                    conflict[b][c] |= (1 << a)
                    conflict[c][b] |= (1 << a)
                    # Weight edges by frequency of bad triples they participate in
                    edge_weight[a][b] += 1.0
                    edge_weight[b][a] += 1.0
                    edge_weight[a][c] += 1.0
                    edge_weight[c][a] += 1.0
                    edge_weight[b][c] += 1.0
                    edge_weight[c][b] += 1.0
    
    # Build adjacency for conflict graph
    adj = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if edge_weight[a][b] > 0:
                adj[a].append((b, edge_weight[a][b]))
                adj[b].append((a, edge_weight[b][a]))
    
    # Compute degree-weighted conflict score
    conflict_score = [0.0] * (N + 1)
    for a in range(1, N + 1):
        for b, w in adj[a]:
            conflict_score[a] += w

    # Maximum clique upper bound using greedy coloring (Tomita algorithm)
    def max_clique_bound(available):
        """Upper bound on clique size in the conflict graph restricted to available nodes."""
        # Convert available mask to list
        avail_list = []
        temp = available
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            avail_list.append(x)
            temp ^= low
        
        if not avail_list:
            return 0
        
        # Greedy coloring for upper bound on maximum clique
        # Sort by degree in the available subgraph
        deg_in_avail = {}
        for x in avail_list:
            deg_in_avail[x] = 0
        for x in avail_list:
            for y, w in adj[x]:
                if (available >> y) & 1:
                    deg_in_avail[x] += 1
        
        # Color greedily: assign colors so that no two adjacent nodes share a color
        # The number of colors used is an upper bound on clique size
        order = sorted(avail_list, key=lambda x: -deg_in_avail[x])
        color = {}
        colors_used = 0
        for x in order:
            # Find smallest color not used by neighbors
            neighbor_colors = set()
            for y, w in adj[x]:
                if (available >> y) & 1 and y in color:
                    neighbor_colors.add(color[y])
            c = 1
            while c in neighbor_colors:
                c += 1
            color[x] = c
            colors_used = max(colors_used, c)
        
        return colors_used

    # Weighted greedy
    def weighted_greedy(order):
        S = []
        S_mask = 0
        for x in order:
            if not (full_mask >> x) & 1:
                continue
            # Check if x is compatible with S
            compatible = True
            for s in S:
                if conflict[s][x] & S_mask:
                    compatible = False
                    break
            if compatible:
                S.append(x)
                S_mask |= (1 << x)
        return S
    
    # Try several greedy orderings
    best_set = []
    best_size = 0
    
    # Natural order
    S = weighted_greedy(list(range(1, N + 1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Reverse order
    S = weighted_greedy(list(range(N, 0, -1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Sort by conflict score (ascending - prefer low-conflict elements)
    order = sorted(range(1, N + 1), key=lambda x: conflict_score[x])
    S = weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Sort by conflict score (descending)
    order = sorted(range(1, N + 1), key=lambda x: -conflict_score[x])
    S = weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Sort by number of prime factors
    def num_prime_factors(x):
        count = 0
        d = 2
        n = x
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    order = sorted(range(1, N + 1), key=num_prime_factors)
    S = weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Spectral ordering: use power iteration on (D - W) to get ranking
    import random
    random.seed(42)
    v = [0.0] * (N + 1)
    for i in range(1, N + 1):
        v[i] = random.random()
    
    norm = math.sqrt(sum(v[i]**2 for i in range(1, N + 1)))
    if norm > 0:
        for i in range(1, N + 1):
            v[i] /= norm
    
    for _ in range(100):
        if time.time() - start_time > deadline - 5:
            break
        new_v = [0.0] * (N + 1)
        for i in range(1, N + 1):
            new_v[i] = conflict_score[i] * v[i]
            for j, w in adj[i]:
                new_v[i] -= w * v[j]
        
        norm = math.sqrt(sum(new_v[i]**2 for i in range(1, N + 1)))
        if norm > 1e-10:
            for i in range(1, N + 1):
                new_v[i] /= norm
        v = new_v
    
    spectral_order = sorted(range(1, N + 1), key=lambda x: -v[x])
    S = weighted_greedy(spectral_order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    spectral_order_asc = sorted(range(1, N + 1), key=lambda x: v[x])
    S = weighted_greedy(spectral_order_asc)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Branch and bound with improved upper bound
    def solve(selected, available):
        nonlocal best_size, best_set, start_time
        
        if time.time() - start_time > deadline:
            return
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
        
        if not available:
            return
        
        # Upper bound: use maximum clique bound on the conflict graph
        # If we can't add more than (bound - len(selected)) elements, prune
        avail_count = available.bit_count()
        if len(selected) + avail_count <= best_size:
            return
        
        # Better upper bound using clique coloring
        if time.time() - start_time < deadline - 2:
            ub = max_clique_bound(available)
            if len(selected) + ub <= best_size:
                return
        
        # Find all available elements
        avail_list = []
        temp = available
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            avail_list.append(x)
            temp ^= low
        
        if not avail_list:
            return
        
        # Pick element with highest conflict degree in available
        best_x = avail_list[0]
        best_conflict = -1
        for x in avail_list:
            c = 0.0
            for y, w in adj[x]:
                if (available >> y) & 1:
                    c += w
            if c > best_conflict:
                best_conflict = c
                best_x = x
        x = best_x
        
        # Branch 1: Include x
        new_selected = selected + [x]
        new_avail = available & ~(1 << x)
        for s in selected:
            new_avail &= ~conflict[s][x]
        
        solve(new_selected, new_avail)
        
        if time.time() - start_time > deadline:
            return
        
        # Branch 2: Exclude x
        solve(selected, available & ~(1 << x))
    
    solve([], full_mask)
    
    # Local search polish: try to swap elements to increase size
    def local_search_polish(S, max_iterations=50):
        """Try to improve the set by swapping elements."""
        S_set = set(S)
        S_mask = 0
        for s in S_set:
            S_mask |= (1 << s)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() - start_time < deadline - 1:
            improved = False
            iterations += 1
            
            # Try adding an element
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                # Check if x is compatible
                compatible = True
                for s in S_set:
                    if conflict[s][x] & S_mask:
                        compatible = False
                        break
                if compatible:
                    S_set.add(x)
                    S_mask |= (1 << x)
                    improved = True
                    break
            
            # Try swapping: remove one element, add another
            if not improved:
                for remove in list(S_set):
                    # Check what becomes available
                    temp_S_set = S_set - {remove}
                    temp_S_mask = S_mask & ~(1 << remove)
                    
                    for add in range(1, N + 1):
                        if add in S_set:
                            continue
                        # Check if add is compatible with temp_S_set
                        compatible = True
                        for s in temp_S_set:
                            if conflict[s][add] & temp_S_mask:
                                compatible = False
                                break
                        if compatible:
                            S_set.remove(remove)
                            S_set.add(add)
                            S_mask = temp_S_mask | (1 << add)
                            improved = True
                            break
                    if improved:
                        break
        
        return list(S_set)
    
    # Apply local search polish
    if best_set:
        polished = local_search_polish(best_set)
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = polished
    
    return best_set
