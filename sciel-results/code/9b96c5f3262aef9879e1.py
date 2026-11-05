import random
import time
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    max_time = 115.0  # Safety margin

    best_edges = []
    best_count = 0

    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    pair_set = set(all_pairs)

    def can_add(u, v, adj_mask, common):
        if (adj_mask[u] >> v) & 1:
            return False
        if common[u][v] > 2:
            return False
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if common[u][w] >= 2:
                return False
            m -= lsb
        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if common[v][w] >= 2:
                return False
            m -= lsb
        return True

    def add_edge(u, v, adj_mask, common, edges):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append((u, v))
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            m -= lsb
        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            m -= lsb

    def remove_edge(u, v, adj_mask, common, edges):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.remove((u, v))
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            m -= lsb
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            m -= lsb

    def greedy_fill(adj_mask, common, edges):
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if can_add(u, v, adj_mask, common):
                add_edge(u, v, adj_mask, common, edges)

    def local_search(initial_edges, max_iters=400):
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        for u, v in initial_edges:
            add_edge(u, v, adj_mask, common, edges)

        best_local_edges = list(edges)
        best_local_count = len(edges)

        for it in range(max_iters):
            if time.time() - start_time > max_time - 3:
                break
            pairs = all_pairs[:]
            random.shuffle(pairs)
            added_any = False
            for u, v in pairs:
                if can_add(u, v, adj_mask, common):
                    add_edge(u, v, adj_mask, common, edges)
                    added_any = True
                    if len(edges) > best_local_count:
                        best_local_count = len(edges)
                        best_local_edges = list(edges)
                    break
            if not added_any:
                if len(edges) == 0:
                    break
                kick_size = random.choice([1, 1, 2, 2, 3])
                to_remove = random.sample(edges, min(kick_size, len(edges)))
                for u, v in to_remove:
                    remove_edge(u, v, adj_mask, common, edges)
                for _ in range(40):
                    if time.time() - start_time > max_time - 3:
                        break
                    pairs = all_pairs[:]
                    random.shuffle(pairs)
                    added = False
                    for u, v in pairs:
                        if can_add(u, v, adj_mask, common):
                            add_edge(u, v, adj_mask, common, edges)
                            added = True
                            if len(edges) > best_local_count:
                                best_local_count = len(edges)
                                best_local_edges = list(edges)
                            break
                    if not added:
                        break
        return best_local_edges

    def verify_edges(edge_list):
        adj = [0] * n
        for u, v in edge_list:
            if u == v:
                return False
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            for v in range(u + 1, n):
                if bin(adj[u] & adj[v]).count('1') > 2:
                    return False
        return True

    def crossover(a, b):
        """Uniform crossover of two edge lists (sets)."""
        sa = set(a)
        sb = set(b)
        child = []
        for u, v in all_pairs:
            if (u, v) in sa or (u, v) in sb:
                # keep with prob 0.5 (uniform), else drop
                if random.random() < 0.5:
                    child.append((u, v))
        return child

    def mutate(edge_list, rate):
        """Bit-flip mutation: flip each edge's presence with prob rate."""
        s = set(edge_list)
        out = []
        for u, v in all_pairs:
            present = (u, v) in s
            if present and random.random() < rate:
                continue  # remove
            if (not present) and random.random() < rate:
                out.append((u, v))  # add
            elif present:
                out.append((u, v))
        return out

    def repair(edge_list):
        """Build a valid state from a (possibly invalid) edge list by keeping
        only edges that are valid given the edges already kept (order matters).
        Uses fast can_add/add machinery."""
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        # Try to keep as many as possible; process in given order
        for u, v in edge_list:
            if can_add(u, v, adj_mask, common):
                add_edge(u, v, adj_mask, common, edges)
        return edges

    # Main search loop
    while time.time() - start_time < max_time:
        # --- Local search trajectory ---
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        greedy_fill(adj_mask, common, edges)
        result_edges = local_search(edges)

        if verify_edges(result_edges) and len(result_edges) > best_count:
            best_count = len(result_edges)
            best_edges = [[u, v] for u, v in result_edges]

        if time.time() - start_time > max_time - 5:
            break

        # --- Genetic algorithm trajectory (uses current best as a parent) ---
        if time.time() - start_time < max_time - 8:
            # Build a small population: best + a couple of fresh greedy solutions
            pop = []
            if best_edges:
                pop.append([(u, v) for u, v in best_edges])
            for _ in range(2):
                if time.time() - start_time > max_time - 6:
                    break
                am = [0] * n
                cm = [[0] * n for _ in range(n)]
                ed = []
                greedy_fill(am, cm, ed)
                # quick local polish
                ed = local_search(ed, max_iters=80)
                if verify_edges(ed):
                    pop.append(ed)
            if not pop:
                pop.append([])

            # Evolve a few generations
            for gen in range(3):
                if time.time() - start_time > max_time - 6:
                    break
                new_pop = []
                # keep best
                best_in_pop = max(pop, key=lambda e: len(e))
                new_pop.append(best_in_pop)
                # generate offspring
                for _ in range(3):
                    if time.time() - start_time > max_time - 6:
                        break
                    # pick two parents
                    p1 = random.choice(pop)
                    p2 = random.choice(pop)
                    child = crossover(p1, p2)
                    child = mutate(child, rate=0.15)
                    # repair to valid, then local polish
                    child = repair(child)
                    child = local_search(child, max_iters=60)
                    if verify_edges(child):
                        new_pop.append(child)
                # selection: keep top by size
                new_pop.sort(key=lambda e: len(e), reverse=True)
                pop = new_pop[:4]
                # update global best
                for e in pop:
                    if verify_edges(e) and len(e) > best_count:
                        best_count = len(e)
                        best_edges = [[u, v] for u, v in e]

    # Fallback
    if not best_edges:
        best_edges = []
        for i in range(5):
            base = i * 4
            for u in range(4):
                for v in range(u + 1, 4):
                    best_edges.append([base + u, base + v])

    # Final verification
    if not verify_edges(best_edges):
        return [[0, 1]]

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
