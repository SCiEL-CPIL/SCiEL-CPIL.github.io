import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def is_valid(n, adj):
        # K4 check
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj[x] & (common & ~(1 << x)):
                            return False
        # I6 check
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue
                for w in range(v + 1, n):
                    if (adj[u] >> w) & 1 or (adj[v] >> w) & 1:
                        continue
                    for x in range(w + 1, n):
                        if (adj[u] >> x) & 1 or (adj[v] >> x) & 1 or (adj[w] >> x) & 1:
                            continue
                        for y in range(x + 1, n):
                            if (adj[u] >> y) & 1 or (adj[v] >> y) & 1 or (adj[w] >> y) & 1 or (adj[x] >> y) & 1:
                                continue
                            for z in range(y + 1, n):
                                if (adj[u] >> z) & 1 or (adj[v] >> z) & 1 or (adj[w] >> z) & 1 or (adj[x] >> z) & 1 or (adj[y] >> z) & 1:
                                    continue
                                return False
        return True

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_edges_in_mask(mask, adj):
        total_deg = 0
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            total_deg += (adj[u] & mask).bit_count()
        return total_deg // 2

    def count_indep_4_in_mask(mask, adj):
        vertices = []
        m = mask
        while m:
            lsb = m & -m
            vertices.append(lsb.bit_length() - 1)
            m ^= lsb
        count = 0
        n_v = len(vertices)
        if n_v < 4:
            return 0
        for i in range(n_v):
            for j in range(i + 1, n_v):
                if (adj[vertices[i]] >> vertices[j]) & 1:
                    continue
                for k in range(j + 1, n_v):
                    if (adj[vertices[i]] >> vertices[k]) & 1 or (adj[vertices[j]] >> vertices[k]) & 1:
                        continue
                    for l in range(k + 1, n_v):
                        if (adj[vertices[i]] >> vertices[l]) & 1 or (adj[vertices[j]] >> vertices[l]) & 1 or (adj[vertices[k]] >> vertices[l]) & 1:
                            continue
                        count += 1
        return count

    def delta_violations(u, v, adj, n, is_edge):
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        non_common = ((1 << n) - 1) & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
        e_count = count_edges_in_mask(common, adj)
        i4_count = count_indep_4_in_mask(non_common, adj)
        if is_edge:
            return -e_count + i4_count
        else:
            return e_count - i4_count

    def make_paley_graph(p):
        adj = [0] * p
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def local_search(adj, n, deadline):
        max_rounds = 50
        for _round in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            best_delta = 0
            best_u = -1
            best_v = -1
            for u in range(n):
                for v in range(u + 1, n):
                    is_edge = bool((adj[u] >> v) & 1)
                    d = delta_violations(u, v, adj, n, is_edge)
                    if d < best_delta:
                        best_delta = d
                        best_u = u
                        best_v = v
                        improved = True
            if best_u != -1:
                if (adj[best_u] >> best_v) & 1:
                    adj[best_u] &= ~(1 << best_v)
                    adj[best_v] &= ~(1 << best_u)
                else:
                    adj[best_u] |= (1 << best_v)
                    adj[best_v] |= (1 << best_u)
            else:
                break
        return adj

    def random_graph(n, p=0.5):
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def extend_to_36(adj35):
        """Try to extend a 35-vertex graph to 36 by adding a new vertex"""
        n = 36
        adj = adj35 + [0]
        # Try multiple random connections for the new vertex
        for attempt in range(20):
            adj_new = adj35[:]
            adj_new.append(0)
            # Randomly connect new vertex
            for v in range(35):
                if random.random() < 0.5:
                    adj_new[v] |= (1 << 35)
                    adj_new[35] |= (1 << v)
            # Local search to fix violations
            adj_new = local_search(adj_new, 36, deadline)
            if is_valid(36, adj_new):
                return adj_new
        return None

    def polish(adj, n, deadline):
        """High-precision polish: multiple passes of local search with restarts"""
        best_adj = adj
        best_violations = float('inf')
        
        for _ in range(5):
            if time.time() > deadline:
                break
            # Try a few random perturbations
            for _pert in range(3):
                if time.time() > deadline:
                    break
                perturbed = adj[:]
                # Flip 2-3 random edges
                for _ in range(random.randint(2, 3)):
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v and u < v:
                        if (perturbed[u] >> v) & 1:
                            perturbed[u] &= ~(1 << v)
                            perturbed[v] &= ~(1 << u)
                        else:
                            perturbed[u] |= (1 << v)
                            perturbed[v] |= (1 << u)
                # Local search
                perturbed = local_search(perturbed, n, deadline)
                if is_valid(n, perturbed):
                    return perturbed
        return best_adj

    best_graph = None
    best_n = 0

    # Strategy 1: Try to find a valid 35-vertex graph, then extend to 36
    for target_n in [36, 35]:
        if time.time() > deadline:
            break
        
        found = False
        max_restarts = 15
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            if target_n == 36:
                # Start from Paley 29 + 7 random vertices
                if r == 0:
                    base_p = 29
                    adj = make_paley_graph(base_p)
                    adj = adj + [0] * (target_n - base_p)
                    for u in range(base_p, target_n):
                        for v in range(target_n):
                            if v != u and random.random() < 0.45:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                else:
                    adj = random_graph(target_n, 0.5)
            else:
                # For n=35
                if r == 0:
                    base_p = 29
                    adj = make_paley_graph(base_p)
                    adj = adj + [0] * (target_n - base_p)
                    for u in range(base_p, target_n):
                        for v in range(target_n):
                            if v != u and random.random() < 0.45:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                elif r == 1:
                    adj = [0] * target_n
                    for block in range(2):
                        start = block * 13
                        paley13 = make_paley_graph(13)
                        for i in range(13):
                            for j in range(13):
                                if i != j and (paley13[i] >> j) & 1:
                                    adj[start + i] |= (1 << (start + j))
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if (u < 13 and v >= 26) or (u >= 13 and u < 26 and v >= 26) or (u >= 26 and v >= 26):
                                if random.random() < 0.5:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                            elif (u < 13 and 13 <= v < 26) or (13 <= u < 26 and v >= 26):
                                if random.random() < 0.4:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                else:
                    adj = random_graph(target_n, 0.5)
            
            # Local search
            adj = local_search(adj, target_n, deadline)
            
            if is_valid(target_n, adj):
                best_graph = adj
                best_n = target_n
                found = True
                
                # If we found 35, try to extend to 36
                if target_n == 35:
                    extended = extend_to_36(adj)
                    if extended is not None:
                        best_graph = extended
                        best_n = 36
                break
        
        if found:
            break

    # Strategy 2: If we didn't find n>=35, try smaller n
    if best_n < 35:
        for target_n in range(34, 16, -1):
            if time.time() > deadline:
                break
            
            found = False
            max_restarts = 10
            for r in range(max_restarts):
                if time.time() > deadline:
                    break
                
                adj = random_graph(target_n, 0.5)
                adj = local_search(adj, target_n, deadline)
                
                if is_valid(target_n, adj):
                    best_graph = adj
                    best_n = target_n
                    found = True
                    break
            
            if found:
                break

    # Final polish if we have a graph
    if best_graph is not None and best_n >= 30:
        polished = polish(best_graph, best_n, deadline)
        if polished is not None:
            best_graph = polished

    if best_graph is None:
        adj = make_paley_graph(17)
        return {'n': 17, 'edges': get_edges(adj, 17)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
