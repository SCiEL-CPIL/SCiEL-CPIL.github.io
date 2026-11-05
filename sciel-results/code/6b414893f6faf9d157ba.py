import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges_total = len(all_edges)

    # Precompute edge index lookup
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i

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
        # Check if adding edge (u,v) would create a K2,3
        # 1. u and v must not already have 3 common neighbors
        if (adj[u] & adj[v]).bit_count() >= 3:
            return False
        # 2. For each common neighbor w of u and v, adding (u,v) increases
        #    common neighbors of (u,w) and (v,w) by 1
        #    So we need: for all w adjacent to both u and v,
        #    (adj[u] & adj[w]).bit_count() < 2 and (adj[v] & adj[w]).bit_count() < 2
        #    Wait, that's not quite right. Let me think more carefully.
        
        # After adding (u,v):
        # - For pair (u, w) where w is a neighbor of v: common neighbors increase by 1 (v becomes common)
        #   So we need (adj[u] & adj[w]).bit_count() < 2 for all w adjacent to v (and w != u)
        # - For pair (v, w) where w is a neighbor of u: common neighbors increase by 1 (u becomes common)
        #   So we need (adj[v] & adj[w]).bit_count() < 2 for all w adjacent to u (and w != v)
        
        # Actually more precisely:
        # Adding (u,v) means:
        # - v is now a neighbor of u, so for any w that is neighbor of both u and v (other than u,v),
        #   the pair (u,w) gains v as a common neighbor, and pair (v,w) gains u as a common neighbor.
        # - Also, for any w that is neighbor of u but not v, the pair (v,w) doesn't change.
        # - For any w that is neighbor of v but not u, the pair (u,w) doesn't change.
        
        # Wait, let me reconsider. The common neighbors of (u,w) are vertices x such that x is adjacent to both u and w.
        # Before adding (u,v): common neighbors of (u,w) = adj[u] & adj[w] (excluding u,w themselves but that's handled by bit ops)
        # After adding (u,v): if v is adjacent to w, then v becomes a new common neighbor of (u,w).
        #   So common neighbors of (u,w) after = (adj[u] & adj[w]) | ({v} if w adjacent to v)
        #   But wait, v was already in adj[u] after we add the edge. Let me think in terms of the new adj.
        
        # Let me just check all pairs that could be affected:
        # Pairs involving u: (u, w) for all w != u. The common neighbors change only if v is adjacent to w.
        #   New common neighbors of (u,w) = old common neighbors + {v} if v~w
        #   So we need: for all w adjacent to v (w != u), (adj[u] & adj[w]).bit_count() < 2
        #   (because adding v makes it +1, so we need current count < 2)
        
        # Pairs involving v: (v, w) for all w != v. Similarly:
        #   For all w adjacent to u (w != v), (adj[v] & adj[w]).bit_count() < 2
        
        # Pairs not involving u or v: unchanged.
        
        # But also the pair (u,v) itself: common neighbors = adj[u] & adj[v], need < 3.
        
        # So the check is:
        # 1. (adj[u] & adj[v]).bit_count() < 3
        # 2. For all w in adj[v] (w != u): (adj[u] & adj[w]).bit_count() < 2
        # 3. For all w in adj[u] (w != v): (adj[v] & adj[w]).bit_count() < 2
        
        # But this is O(n) per check. Let me optimize.
        
        # Check 1
        if (adj[u] & adj[v]).bit_count() >= 3:
            return False
        
        # Check 2: for each w adjacent to v (w != u), check (adj[u] & adj[w]).bit_count() < 2
        # This is expensive. Let me think of a faster way.
        
        # Alternative: the number of common neighbors of (u,w) for w in N(v)\{u}
        # We need max over such w of (adj[u] & adj[w]).bit_count() <= 1
        
        # Let's just do the O(n) check
        for w in range(n):
            if w == u or w == v:
                continue
            w_adj_v = (adj[v] >> w) & 1
            w_adj_u = (adj[u] >> w) & 1
            if w_adj_v:
                # w is adjacent to v, so after adding (u,v), v becomes common neighbor of (u,w)
                if (adj[u] & adj[w]).bit_count() >= 2:
                    return False
            if w_adj_u:
                # w is adjacent to u, so after adding (u,v), u becomes common neighbor of (v,w)
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

    # Try multiple strategies
    num_runs = 25
    for run in range(num_runs):
        if time.time() > deadline - 5.0:
            break

        # Varying initialization strategies
        if run < 5:
            # Dense random
            p = random.uniform(0.30, 0.50)
        elif run < 10:
            # Medium random
            p = random.uniform(0.20, 0.35)
        elif run < 15:
            # Structured: try block designs
            p = random.uniform(0.25, 0.40)
        else:
            # Very dense
            p = random.uniform(0.35, 0.55)

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

        # Simulated annealing
        T = 5.0
        cooling = 0.9985
        min_T = 0.05
        steps = 0
        max_steps = 80000

        while T > min_T and steps < max_steps:
            if time.time() > deadline - 5.0:
                break

            idx = random.randint(0, num_edges_total - 1)
            u, v = all_edges[idx]
            is_present = (current_mask >> idx) & 1

            if is_present:
                # Try removing
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_violations = count_violations(adj)
                new_edge_count = edge_count - 1
                old_energy = violations * 1000 - edge_count
                new_energy = new_violations * 1000 - new_edge_count
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
                # Try adding
                if can_add(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_violations = violations
                    new_edge_count = edge_count + 1
                    old_energy = violations * 1000 - edge_count
                    new_energy = new_violations * 1000 - new_edge_count
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
                    # Can't add without violation, but try anyway with high penalty
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_violations = violations + 1
                    new_edge_count = edge_count + 1
                    old_energy = violations * 1000 - edge_count
                    new_energy = new_violations * 1000 - new_edge_count
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

        # Fix violations
        if violations > 0:
            for _ in range(300):
                if time.time() > deadline - 3.0:
                    break
                found = False
                for a in range(n):
                    for b in range(a + 1, n):
                        if (adj[a] & adj[b]).bit_count() >= 3:
                            # Find edge to remove
                            candidates = []
                            for i in range(num_edges_total):
                                eu, ev = all_edges[i]
                                if (current_mask >> i) & 1:
                                    if eu == a or eu == b or ev == a or ev == b:
                                        candidates.append(i)
                            if candidates:
                                best_rem = None
                                best_score = float('inf')
                                for rem_idx in candidates:
                                    ru, rv = all_edges[rem_idx]
                                    adj[ru] &= ~(1 << rv)
                                    adj[rv] &= ~(1 << ru)
                                    test_violations = count_violations(adj)
                                    adj[ru] |= (1 << rv)
                                    adj[rv] |= (1 << ru)
                                    if test_violations < best_score:
                                        best_score = test_violations
                                        best_rem = rem_idx
                                if best_rem is not None:
                                    ru, rv = all_edges[best_rem]
                                    adj[ru] &= ~(1 << rv)
                                    adj[rv] &= ~(1 << ru)
                                    current_mask ^= (1 << best_rem)
                                    violations = count_violations(adj)
                                    edge_count = bin(current_mask).count('1')
                                    found = True
                                    break
                    if found:
                        break
                if violations == 0:
                    break

        # Greedy fill
        if violations == 0:
            greedy_fill(adj)
            fill_mask = adj_to_mask(adj)
            fill_count = bin(fill_mask).count('1')
            if fill_count > best_edge_count:
                best_edge_count = fill_count
                best_mask = fill_mask

    # Final polish: local search on best solution
    if best_mask != 0:
        adj = init_adj(best_mask)
        # Try swap moves: remove one edge, add another
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(num_edges_total):
                if time.time() > deadline - 2.0:
                    break
                u1, v1 = all_edges[i]
                if not ((adj[u1] >> v1) & 1):
                    continue  # edge not present, skip
                # Try removing this edge
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                for j in range(num_edges_total):
                    if j == i:
                        continue
                    u2, v2 = all_edges[j]
                    if (adj[u2] >> v2) & 1:
                        continue  # already present
                    if can_add(u2, v2, adj):
                        # Do the swap
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        improved = True
                        break
                # Undo removal
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                if improved:
                    break

        greedy_fill(adj)
        best_mask = adj_to_mask(adj)

    # Final greedy fill
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
