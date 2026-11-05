import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges_total = len(all_edges)

    def get_edges_from_mask(mask):
        edges = []
        for i in range(num_edges_total):
            if (mask >> i) & 1:
                edges.append(list(all_edges[i]))
        return edges

    def init_adj(mask):
        adj = [0] * n
        for i in range(num_edges_total):
            if (mask >> i) & 1:
                u, v = all_edges[i]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def count_violations(adj):
        v = 0
        for u in range(n):
            for w in range(u + 1, n):
                if (adj[u] & adj[w]).bit_count() >= 3:
                    v += 1
        return v

    def can_add(u, v, adj):
        if (adj[u] & adj[v]).bit_count() >= 3:
            return False
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[w] >> v) & 1:
                if (adj[u] & adj[w]).bit_count() >= 2:
                    return False
            if (adj[w] >> u) & 1:
                if (adj[v] & adj[w]).bit_count() >= 2:
                    return False
        return True

    def greedy_fill(adj):
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if can_add(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True

    def adj_to_mask(adj):
        mask = 0
        for i in range(num_edges_total):
            u, v = all_edges[i]
            if (adj[u] >> v) & 1:
                mask |= (1 << i)
        return mask

    best_mask = 0
    best_edge_count = 0

    num_runs = 20
    for run in range(num_runs):
        if time.time() > deadline - 3.0:
            break

        p = random.uniform(0.15, 0.40)
        current_mask = 0
        for i in range(num_edges_total):
            if random.random() < p:
                current_mask |= (1 << i)

        adj = init_adj(current_mask)
        violations = count_violations(adj)
        edge_count = bin(current_mask).count('1')

        if violations == 0 and edge_count > best_edge_count:
            best_edge_count = edge_count
            best_mask = current_mask

        T = 2.0
        cooling = 0.995
        min_T = 0.05
        steps = 0
        max_steps = 30000

        while T > min_T and steps < max_steps:
            if time.time() > deadline - 3.0:
                break

            idx = random.randint(0, num_edges_total - 1)
            u, v = all_edges[idx]
            is_present = (current_mask >> idx) & 1

            if is_present:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_violations = count_violations(adj)
                new_edge_count = edge_count - 1
                old_energy = violations * 100 - edge_count
                new_energy = new_violations * 100 - new_edge_count
                delta_energy = new_energy - old_energy
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    current_mask ^= (1 << idx)
                    violations = new_violations
                    edge_count = new_edge_count
                    if violations == 0 and edge_count > best_edge_count:
                        best_edge_count = edge_count
                        best_mask = current_mask
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                if can_add(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_violations = violations
                    new_edge_count = edge_count + 1
                    old_energy = violations * 100 - edge_count
                    new_energy = new_violations * 100 - new_edge_count
                    delta_energy = new_energy - old_energy
                    if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                        current_mask |= (1 << idx)
                        violations = new_violations
                        edge_count = new_edge_count
                        if violations == 0 and edge_count > best_edge_count:
                            best_edge_count = edge_count
                            best_mask = current_mask
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_violations = violations + 1
                    new_edge_count = edge_count + 1
                    old_energy = violations * 100 - edge_count
                    new_energy = new_violations * 100 - new_edge_count
                    delta_energy = new_energy - old_energy
                    if random.random() < math.exp(-delta_energy / T):
                        current_mask |= (1 << idx)
                        violations = new_violations
                        edge_count = new_edge_count
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)

            T *= cooling
            steps += 1

        if violations == 0:
            greedy_fill(adj)
            fill_mask = adj_to_mask(adj)
            fill_count = bin(fill_mask).count('1')
            if fill_count > best_edge_count:
                best_edge_count = fill_count
                best_mask = fill_mask
        elif violations > 0:
            for _ in range(200):
                if time.time() > deadline - 2.0:
                    break
                found = False
                for a in range(n):
                    for b in range(a + 1, n):
                        if (adj[a] & adj[b]).bit_count() >= 3:
                            candidates = []
                            for i in range(num_edges_total):
                                eu, ev = all_edges[i]
                                if (current_mask >> i) & 1:
                                    if eu == a or eu == b or ev == a or ev == b:
                                        candidates.append(i)
                            if candidates:
                                rem_idx = random.choice(candidates)
                                ru, rv = all_edges[rem_idx]
                                adj[ru] &= ~(1 << rv)
                                adj[rv] &= ~(1 << ru)
                                current_mask ^= (1 << rem_idx)
                                violations = count_violations(adj)
                                edge_count = bin(current_mask).count('1')
                                found = True
                                break
                    if found:
                        break
                if violations == 0:
                    break
            if violations == 0:
                greedy_fill(adj)
                fill_mask = adj_to_mask(adj)
                fill_count = bin(fill_mask).count('1')
                if fill_count > best_edge_count:
                    best_edge_count = fill_count
                    best_mask = fill_mask

    if best_mask != 0:
        adj = init_adj(best_mask)
        greedy_fill(adj)
        best_mask = adj_to_mask(adj)

    return get_edges_from_mask(best_mask)

def _construct_greedy():
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.seed(42)
    shuffled = all_edges[:]
    random.shuffle(shuffled)
    adj = [0] * n
    edges = []
    for u, v in shuffled:
        if (adj[u] & adj[v]).bit_count() < 3:
            ok = True
            for w in range(n):
                if w == u or w == v:
                    continue
                if (adj[u] & adj[w]).bit_count() + (1 if (adj[w] & (1 << v)) else 0) >= 3:
                    ok = False
                    break
                if ok:
                    if (adj[v] & adj[w]).bit_count() + (1 if (adj[w] & (1 << u)) else 0) >= 3:
                        ok = False
                        break
            if ok:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
    return edges

def _greedy_construct(n, all_edges, seed):
    pass

def _can_add_edge_fast(adj, u, v):
    pass

def _local_search(n, edges, all_edges, deadline):
    pass
