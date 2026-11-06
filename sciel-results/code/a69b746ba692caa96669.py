import time
import random
import numpy as np

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
                        if count >= 100:
                            return count
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

    def spectral_init(n, deadline):
        """
        Spectral graph construction: start with random ±1 matrix,
        iteratively flip entries to reduce spectral radius,
        then threshold to get a graph.
        """
        # Random symmetric matrix with ±1 off-diagonal, 0 diagonal
        S = np.random.randint(-1, 2, size=(n, n))
        # Make symmetric
        S = (S + S.T) // 2
        np.fill_diagonal(S, 0)
        # Zero out diagonal again after symmetrization
        np.fill_diagonal(S, 0)

        # Target: minimize spectral radius while keeping off-diagonal in {-1, 1}
        # Use greedy sign-flipping: flip entry (i,j) if it reduces max abs eigenvalue
        max_iters = 200
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Compute eigenvalues
            try:
                eigvals = np.linalg.eigvalsh(S)
            except np.linalg.LinAlgError:
                break
            max_abs_eig = np.max(np.abs(eigvals))

            # Try flipping a random subset of entries
            improved = False
            num_tries = 30
            for _ in range(num_tries):
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                old_val = S[i, j]
                new_val = -old_val
                S[i, j] = new_val
                S[j, i] = new_val
                try:
                    eigvals_new = np.linalg.eigvalsh(S)
                except np.linalg.LinAlgError:
                    S[i, j] = old_val
                    S[j, i] = old_val
                    continue
                max_abs_eig_new = np.max(np.abs(eigvals_new))
                if max_abs_eig_new < max_abs_eig:
                    max_abs_eig = max_abs_eig_new
                    improved = True
                else:
                    # Revert
                    S[i, j] = old_val
                    S[j, i] = old_val

            if not improved:
                break

        # Threshold: edge if S[i,j] > 0
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if S[i, j] > 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def local_search(adj, n, deadline):
        """Perform local search to minimize violations with time-bounded iterations"""
        max_rounds = 200
        for _round in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            best_delta = 0
            best_u = -1
            best_v = -1

            for u in range(n):
                if time.time() > deadline:
                    return adj
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

    def simulated_annealing(adj, n, deadline):
        """SA-based local search for escaping local optima"""
        temp = 10.0
        cooling = 0.95
        max_rounds = 500
        for _round in range(max_rounds):
            if time.time() > deadline:
                break
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u

            is_edge = bool((adj[u] >> v) & 1)
            d = delta_violations(u, v, adj, n, is_edge)

            if d <= 0 or random.random() < np.exp(-d / max(temp, 0.01)):
                if is_edge:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            temp *= cooling
            if temp < 0.01:
                break
        return adj

    best_graph = None
    best_n = 0

    # Strategy 1: Try to find a valid 36-vertex graph
    for target_n in [36, 35]:
        if time.time() > deadline:
            break

        found = False
        max_restarts = 20

        for r in range(max_restarts):
            if time.time() > deadline:
                break

            if target_n == 36:
                if r == 0:
                    # Spectral initialization
                    adj = spectral_init(target_n, deadline)
                elif r == 1:
                    # Paley 29 + 7 random
                    base_p = 29
                    adj = make_paley_graph(base_p)
                    adj = adj + [0] * (target_n - base_p)
                    for u in range(base_p, target_n):
                        for v in range(target_n):
                            if v != u and random.random() < 0.45:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                elif r == 2:
                    # Two Paley 13 blocks + 10 random
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
                    # Random
                    adj = [0] * target_n
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if random.random() < 0.5:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
            else:
                if r == 0:
                    adj = spectral_init(target_n, deadline)
                elif r == 1:
                    base_p = 29
                    adj = make_paley_graph(base_p)
                    adj = adj + [0] * (target_n - base_p)
                    for u in range(base_p, target_n):
                        for v in range(target_n):
                            if v != u and random.random() < 0.45:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                else:
                    adj = [0] * target_n
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if random.random() < 0.5:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)

            # Local search first
            adj = local_search(adj, target_n, deadline)

            if is_valid(target_n, adj):
                best_graph = adj
                best_n = target_n
                found = True
                break

            # If not valid, try SA
            if time.time() < deadline - 10:
                adj_sa = list(adj)
                adj_sa = simulated_annealing(adj_sa, target_n, deadline)
                adj_sa = local_search(adj_sa, target_n, deadline)
                if is_valid(target_n, adj_sa):
                    best_graph = adj_sa
                    best_n = target_n
                    found = True
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

                adj = [0] * target_n
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if random.random() < 0.5:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

                adj = local_search(adj, target_n, deadline)

                if is_valid(target_n, adj):
                    best_graph = adj
                    best_n = target_n
                    found = True
                    break

            if found:
                break

    if best_graph is None:
        # Fallback to Paley 17
        adj = make_paley_graph(17)
        return {'n': 17, 'edges': get_edges(adj, 17)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
