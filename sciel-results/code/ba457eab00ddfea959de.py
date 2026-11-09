import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    n = 20
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    # Bitset helpers
    def popcount(x):
        return bin(x).count('1')

    def count_edges(adj):
        return sum(popcount(x) for x in adj) // 2

    def is_k23_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return False
        return True

    def valid_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if adj[u] & (1 << v):
            return False
        # 1. u and v must have at most 2 common neighbors
        if popcount(adj[u] & adj[v]) > 2:
            return False
        # 2. For each w in N(u), w and v must have at most 1 common neighbor
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if popcount(adj[w] & adj[v]) > 1:
                return False
        # 3. For each w in N(v), w and u must have at most 1 common neighbor
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if popcount(adj[w] & adj[u]) > 1:
                return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edges(adj):
        edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
                temp -= lsb
        return edges

    def get_non_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    edges.append((u, v))
        return edges

    def greedy_fill(adj, cn_matrix):
        """Greedily add all possible edges in random order using cn_matrix."""
        non_edges = get_non_edges(adj)
        random.shuffle(non_edges)
        for u, v in non_edges:
            if cn_matrix[u][v] > 2:
                continue
            valid = True
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                w = lsb.bit_length() - 1
                temp -= lsb
                if cn_matrix[w][v] > 1:
                    valid = False
                    break
            if valid:
                temp = adj[v]
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    temp -= lsb
                    if cn_matrix[w][u] > 1:
                        valid = False
                        break
            if valid:
                add_edge(adj, u, v)
                # Update cn_matrix efficiently
                _update_cn_after_add(cn_matrix, adj, u, v)

    def _update_cn_after_add(cn_matrix, adj, u, v):
        """Update cn_matrix after adding edge (u,v)."""
        cn_matrix[u][v] = popcount(adj[u] & adj[v])
        cn_matrix[v][u] = cn_matrix[u][v]
        # For each w adjacent to u, cn[w][v] increases by 1 if w is adjacent to v
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if w != v:
                if adj[w] & (1 << v):
                    cn_matrix[w][v] += 1
                    cn_matrix[v][w] += 1
                else:
                    # w is not adjacent to v, so cn[w][v] unchanged
                    pass
        # For each w adjacent to v, cn[w][u] increases by 1 if w is adjacent to u
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if w != u:
                if adj[w] & (1 << u):
                    cn_matrix[w][u] += 1
                    cn_matrix[u][w] += 1
                else:
                    pass

    def _update_cn_after_remove(cn_matrix, adj, u, v):
        """Update cn_matrix after removing edge (u,v)."""
        cn_matrix[u][v] = popcount(adj[u] & adj[v])
        cn_matrix[v][u] = cn_matrix[u][v]
        # For each w adjacent to u, cn[w][v] decreases by 1 if w is adjacent to v
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if w != v:
                if adj[w] & (1 << v):
                    cn_matrix[w][v] -= 1
                    cn_matrix[v][w] -= 1
        # For each w adjacent to v, cn[w][u] decreases by 1 if w is adjacent to u
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if w != u:
                if adj[w] & (1 << u):
                    cn_matrix[w][u] -= 1
                    cn_matrix[u][w] -= 1

    def build_cn_matrix(adj):
        """Build the common-neighbor count matrix from scratch."""
        cn = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                c = popcount(adj[a] & adj[b])
                cn[a][b] = c
                cn[b][a] = c
        return cn

    # Initial seed: 5 disjoint K4s
    def make_k4_union():
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(adj, base + a, base + b)
        return adj

    best_edges = 0
    best_adj = None

    # First, establish baseline
    adj0 = make_k4_union()
    cn0 = build_cn_matrix(adj0)
    greedy_fill(adj0, cn0)
    best_edges = count_edges(adj0)
    best_adj = adj0[:]

    # Multi-restart simulated annealing
    num_restarts = 25

    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break

        # Different starting points
        if r == 0:
            adj = make_k4_union()
        elif r < 5:
            # Start from best found, then perturb more aggressively
            adj = best_adj[:]
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:5]:
                remove_edge(adj, u, v)
        else:
            # Start from best found, then perturb
            adj = best_adj[:]
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:3]:
                remove_edge(adj, u, v)

        cn_matrix = build_cn_matrix(adj)
        greedy_fill(adj, cn_matrix)
        current_edges = count_edges(adj)

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

        # Simulated annealing
        T = 0.8
        T_min = 0.02
        cooling = 0.997
        steps = 0
        no_improve_count = 0
        max_no_improve = 2000

        while T > T_min and (time.time() - start_time < time_limit - 5) and no_improve_count < max_no_improve:
            steps += 1
            move_type = random.random()

            if move_type < 0.35:
                # Try to add a random non-edge
                non_edges = get_non_edges(adj)
                if not non_edges:
                    T *= cooling
                    no_improve_count += 1
                    continue
                u, v = random.choice(non_edges)
                if cn_matrix[u][v] > 2:
                    no_improve_count += 1
                    T *= cooling
                    continue
                valid = True
                temp = adj[u]
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    temp -= lsb
                    if cn_matrix[w][v] > 1:
                        valid = False
                        break
                if valid:
                    temp = adj[v]
                    while temp:
                        lsb = temp & -temp
                        w = lsb.bit_length() - 1
                        temp -= lsb
                        if cn_matrix[w][u] > 1:
                            valid = False
                            break
                if valid:
                    add_edge(adj, u, v)
                    _update_cn_after_add(cn_matrix, adj, u, v)
                    current_edges += 1
                    no_improve_count = 0
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    no_improve_count += 1
                T *= cooling

            elif move_type < 0.5:
                # Try to remove a random edge (accept with SA probability)
                edges = get_edges(adj)
                if not edges:
                    T *= cooling
                    no_improve_count += 1
                    continue
                u, v = random.choice(edges)
                if random.random() < math.exp(-1.0 / T):
                    remove_edge(adj, u, v)
                    _update_cn_after_remove(cn_matrix, adj, u, v)
                    current_edges -= 1
                    no_improve_count += 1
                T *= cooling

            elif move_type < 0.7:
                # 1-for-1 swap: remove one edge, try to add another
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or not non_edges:
                    T *= cooling
                    no_improve_count += 1
                    continue
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(non_edges)
                
                # Remove (u1,v1)
                remove_edge(adj, u1, v1)
                _update_cn_after_remove(cn_matrix, adj, u1, v1)
                
                # Check validity of adding (u2, v2)
                valid_swap = True
                if adj[u2] & (1 << v2):
                    valid_swap = False
                elif cn_matrix[u2][v2] > 2:
                    valid_swap = False
                else:
                    temp = adj[u2]
                    while temp:
                        lsb = temp & -temp
                        w = lsb.bit_length() - 1
                        temp -= lsb
                        if cn_matrix[w][v2] > 1:
                            valid_swap = False
                            break
                    if valid_swap:
                        temp = adj[v2]
                        while temp:
                            lsb = temp & -temp
                            w = lsb.bit_length() - 1
                            temp -= lsb
                            if cn_matrix[w][u2] > 1:
                                valid_swap = False
                                break
                
                if valid_swap:
                    add_edge(adj, u2, v2)
                    _update_cn_after_add(cn_matrix, adj, u2, v2)
                    # Edge count unchanged, but structure changed
                    if random.random() < 0.7:
                        pass  # keep the swap
                        no_improve_count += 1
                    else:
                        # Revert
                        remove_edge(adj, u2, v2)
                        _update_cn_after_remove(cn_matrix, adj, u2, v2)
                        add_edge(adj, u1, v1)
                        _update_cn_after_add(cn_matrix, adj, u1, v1)
                        no_improve_count += 1
                else:
                    # Revert
                    add_edge(adj, u1, v1)
                    _update_cn_after_add(cn_matrix, adj, u1, v1)
                    no_improve_count += 1
                T *= cooling

            else:
                # 1-for-2 swap: remove one edge, try to add two edges (net +1)
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or len(non_edges) < 2:
                    T *= cooling
                    no_improve_count += 1
                    continue
                u1, v1 = random.choice(edges)
                
                # Remove (u1,v1)
                remove_edge(adj, u1, v1)
                _update_cn_after_remove(cn_matrix, adj, u1, v1)
                
                # Try to add two non-edges
                random.shuffle(non_edges)
                added = 0
                added_edges = []
                for u2, v2 in non_edges:
                    if added >= 2:
                        break
                    if adj[u2] & (1 << v2):
                        continue
                    if cn_matrix[u2][v2] > 2:
                        continue
                    valid = True
                    temp = adj[u2]
                    while temp:
                        lsb = temp & -temp
                        w = lsb.bit_length() - 1
                        temp -= lsb
                        if cn_matrix[w][v2] > 1:
                            valid = False
                            break
                    if valid:
                        temp = adj[v2]
                        while temp:
                            lsb = temp & -temp
                            w = lsb.bit_length() - 1
                            temp -= lsb
                            if cn_matrix[w][u2] > 1:
                                valid = False
                                break
                    if valid:
                        add_edge(adj, u2, v2)
                        _update_cn_after_add(cn_matrix, adj, u2, v2)
                        added += 1
                        added_edges.append((u2, v2))
                
                if added >= 1:
                    # Net change: -1 + added
                    current_edges += added - 1
                    if added >= 2:
                        no_improve_count = 0
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                    else:
                        no_improve_count += 1
                else:
                    # Revert
                    add_edge(adj, u1, v1)
                    _update_cn_after_add(cn_matrix, adj, u1, v1)
                    no_improve_count += 1
                T *= cooling

    # Final greedy fill on best
    if best_adj:
        cn_final = build_cn_matrix(best_adj)
        greedy_fill(best_adj, cn_final)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges

    # Polish pass: try all possible single-edge additions and 1-for-2 swaps
    if best_adj and (time.time() - start_time < time_limit - 3):
        polish_adj = best_adj[:]
        cn_polish = build_cn_matrix(polish_adj)
        polish_edges = count_edges(polish_adj)
        improved = True
        polish_iter = 0
        max_polish_iters = 20
        
        while improved and (time.time() - start_time < time_limit - 3) and polish_iter < max_polish_iters:
            improved = False
            polish_iter += 1
            
            # Try adding any non-edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not (polish_adj[u] & (1 << v)):
                        if cn_polish[u][v] > 2:
                            continue
                        valid = True
                        temp = polish_adj[u]
                        while temp:
                            lsb = temp & -temp
                            w = lsb.bit_length() - 1
                            temp -= lsb
                            if cn_polish[w][v] > 1:
                                valid = False
                                break
                        if valid:
                            temp = polish_adj[v]
                            while temp:
                                lsb = temp & -temp
                                w = lsb.bit_length() - 1
                                temp -= lsb
                                if cn_polish[w][u] > 1:
                                    valid = False
                                    break
                        if valid:
                            add_edge(polish_adj, u, v)
                            _update_cn_after_add(cn_polish, polish_adj, u, v)
                            polish_edges += 1
                            improved = True
            
            # Try 1-for-2 swaps: remove edge (a,b), add two edges (c,d), (e,f)
            if time.time() - start_time > time_limit - 5:
                break
                
            edges = get_edges(polish_adj)
            non_edges = get_non_edges(polish_adj)
            
            for a, b in edges:
                if time.time() - start_time > time_limit - 5:
                    break
                if len(non_edges) < 2:
                    break
                    
                # Try pairs of non-edges
                for i in range(len(non_edges)):
                    if time.time() - start_time > time_limit - 5:
                        break
                    c, d = non_edges[i]
                    # Remove (a,b)
                    remove_edge(polish_adj, a, b)
                    _update_cn_after_remove(cn_polish, polish_adj, a, b)
                    
                    # Try to add (c,d)
                    valid_cd = True
                    if polish_adj[c] & (1 << d):
                        valid_cd = False
                    elif cn_polish[c][d] > 2:
                        valid_cd = False
                    else:
                        temp = polish_adj[c]
                        while temp:
                            lsb = temp & -temp
                            w = lsb.bit_length() - 1
                            temp -= lsb
                            if cn_polish[w][d] > 1:
                                valid_cd = False
                                break
                        if valid_cd:
                            temp = polish_adj[d]
                            while temp:
                                lsb = temp & -temp
                                w = lsb.bit_length() - 1
                                temp -= lsb
                                if cn_polish[w][c] > 1:
                                    valid_cd = False
                                    break
                    
                    if not valid_cd:
                        add_edge(polish_adj, a, b)
                        _update_cn_after_add(cn_polish, polish_adj, a, b)
                        continue
                    
                    add_edge(polish_adj, c, d)
                    _update_cn_after_add(cn_polish, polish_adj, c, d)
                    
                    # Try to add another non-edge
                    added_second = False
                    for j in range(len(non_edges)):
                        if i == j:
                            continue
                        e, f = non_edges[j]
                        if polish_adj[e] & (1 << f):
                            continue
                        if cn_polish[e][f] > 2:
                            continue
                        valid_ef = True
                        temp = polish_adj[e]
                        while temp:
                            lsb = temp & -temp
                            w = lsb.bit_length() - 1
                            temp -= lsb
                            if cn_polish[w][f] > 1:
                                valid_ef = False
                                break
                        if valid_ef:
                            temp = polish_adj[f]
                            while temp:
                                lsb = temp & -temp
                                w = lsb.bit_length() - 1
                                temp -= lsb
                                if cn_polish[w][e] > 1:
                                    valid_ef = False
                                    break
                        if valid_ef:
                            add_edge(polish_adj, e, f)
                            _update_cn_after_add(cn_polish, polish_adj, e, f)
                            polish_edges += 1  # net +1
                            added_second = True
                            improved = True
                            break
                    
                    if not added_second:
                        # Revert the addition of (c,d)
                        remove_edge(polish_adj, c, d)
                        _update_cn_after_remove(cn_polish, polish_adj, c, d)
                    
                    # Revert the removal of (a,b)
                    add_edge(polish_adj, a, b)
                    _update_cn_after_add(cn_polish, polish_adj, a, b)
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if polish_edges > best_edges:
                best_edges = polish_edges
                best_adj = polish_adj[:]

    # Convert to list of edges
    result = []
    for u, v in get_edges(best_adj):
        result.append([u, v])
    return result

# EVOLVE-BLOCK-END
