import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290  # safety margin for 330s limit

    def bits_to_list(mask):
        result = []
        temp = mask
        while temp:
            lsb = temp & -temp
            result.append(lsb.bit_length() - 1)
            temp ^= lsb
        return result

    def has_k4(adj, mask):
        """True iff the subgraph induced by `mask` contains a K4 (bit-parallel)."""
        verts = bits_to_list(mask)
        for i in range(len(verts)):
            u = verts[i]
            neigh = adj[u] & mask & ~(1 << u)
            neigh_list = bits_to_list(neigh)
            for j in range(len(neigh_list)):
                v = neigh_list[j]
                common = adj[u] & adj[v] & mask & ~(1 << u) & ~(1 << v)
                if common == 0:
                    continue
                cl = bits_to_list(common)
                for a in cl:
                    # need b in common with b>a and edge a-b
                    rest = common & ~(1 << a) & (~((1 << (a + 1)) - 1))
                    if rest & adj[a]:
                        return True
        return False

    def has_i4(adj, mask):
        """True iff the subgraph induced by `mask` contains an independent 4-set."""
        verts = bits_to_list(mask)
        for i in range(len(verts)):
            u = verts[i]
            non_neigh = mask & ~adj[u] & ~(1 << u)
            nn_list = bits_to_list(non_neigh)
            for j in range(len(nn_list)):
                v = nn_list[j]
                common = mask & ~adj[u] & ~adj[v] & ~(1 << u) & ~(1 << v)
                if common == 0:
                    continue
                cl = bits_to_list(common)
                for a in cl:
                    rest = common & ~(1 << a) & (~((1 << (a + 1)) - 1))
                    if rest & ~adj[a]:
                        return True
        return False

    def is_valid(adj, n):
        """No K5 and no independent 5-set."""
        for u in range(n):
            if has_k4(adj, adj[u]):
                return False
        all_mask = (1 << n) - 1
        for u in range(n):
            non_neigh = all_mask & ~adj[u] & ~(1 << u)
            if has_i4(adj, non_neigh):
                return False
        return True

    def can_add(adj, n, neighbors_mask):
        """Can a new vertex with exactly `neighbors_mask` be added while staying (5,5)-free?"""
        if has_k4(adj, neighbors_mask):
            return False
        all_mask = (1 << n) - 1
        non_neighbors = all_mask & ~neighbors_mask
        if has_i4(adj, non_neighbors):
            return False
        return True

    def try_add(adj, n, neighbors_mask):
        if can_add(adj, n, neighbors_mask):
            new_n = n + 1
            new_adj = adj + [0]
            for v in range(n):
                if neighbors_mask & (1 << v):
                    new_adj[v] |= (1 << new_n)
                    new_adj[new_n] |= (1 << v)
            return new_n, new_adj
        return None

    # 1. Seed: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    if not is_valid(adj, n):
        return n, []

    best_n = n
    best_adj = adj[:]

    # 2. Aggressive extension: add vertices, flip edges, keep the best (5,5)-free graph
    cur_n = n
    cur_adj = adj[:]

    while cur_n < 46:
        if time.time() > deadline:
            break

        # --- Phase A: try to add one vertex with a good neighbor set ---
        all_mask = (1 << cur_n) - 1
        candidates = []
        # neighborhoods of existing vertices
        for v in range(cur_n):
            candidates.append(cur_adj[v])
        # non-neighborhoods (complements)
        for v in range(cur_n):
            candidates.append(all_mask & ~cur_adj[v] & ~(1 << v))
        # random neighbor sets of varying density
        for _ in range(60):
            size = random.randint(cur_n // 4, (3 * cur_n) // 4)
            if size > cur_n:
                size = cur_n
            neigh = random.sample(range(cur_n), size)
            m = 0
            for v in neigh:
                m |= (1 << v)
            candidates.append(m)
        random.shuffle(candidates)

        added = False
        for m in candidates:
            if time.time() > deadline:
                break
            res = try_add(cur_adj, cur_n, m)
            if res is not None:
                cur_n, cur_adj = res
                best_n = cur_n
                best_adj = cur_adj[:]
                added = True
                break

        if added:
            continue

        # --- Phase B: stuck on vertex count -> perturb via edge flips ---
        improved = False
        for _ in range(40):
            if time.time() > deadline:
                break
            i = random.randrange(cur_n)
            j = random.randrange(cur_n)
            if i == j:
                continue
            new_adj = cur_adj[:]
            if cur_adj[i] & (1 << j):
                new_adj[i] &= ~(1 << j)
                new_adj[j] &= ~(1 << i)
            else:
                new_adj[i] |= (1 << j)
                new_adj[j] |= (1 << i)
            if is_valid(new_adj, cur_n):
                cur_adj = new_adj
                # after a valid flip, retry adding a vertex
                all_mask = (1 << cur_n) - 1
                for _t in range(30):
                    if time.time() > deadline:
                        break
                    size = random.randint(cur_n // 4, (3 * cur_n) // 4)
                    if size > cur_n:
                        size = cur_n
                    neigh = random.sample(range(cur_n), size)
                    m = 0
                    for v in neigh:
                        m |= (1 << v)
                    res = try_add(cur_adj, cur_n, m)
                    if res is not None:
                        cur_n, cur_adj = res
                        best_n = cur_n
                        best_adj = cur_adj[:]
                        improved = True
                        break
                if improved:
                    break
        if improved:
            continue

        # --- Phase C: still stuck -> occasional restart from best ---
        if random.random() < 0.5:
            cur_n = best_n
            cur_adj = best_adj[:]

    # 3. Reconstruct edges from the best graph found
    final_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                final_edges.append((i, j))

    return best_n, final_edges
# EVOLVE-BLOCK-END
