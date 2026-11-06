"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import math

    def bron_kerbosch_bitmask(n, adj_mask, cliques):
        """Bron-Kerbosch with pivoting, using bitmasks. Collects cliques of size >= 2."""
        def bk(R, P, X):
            if P == 0 and X == 0:
                if R.bit_count() >= 2:
                    cliques.append(R)
                return
            # Pivot: choose v in P|X maximizing |P & adj[v]|
            PX = P | X
            if PX == 0:
                return
            best_v = -1
            best_count = -1
            temp = PX
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                cnt = (P & adj_mask[v]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
                temp ^= lsb
            # Candidates: P minus neighbors of pivot
            if best_v >= 0:
                candidates = P & ~adj_mask[best_v]
            else:
                candidates = P
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                bk(R | lsb, P & adj_mask[v], X & adj_mask[v])
                P ^= lsb
                X |= lsb
                temp ^= lsb

        bk(0, (1 << n) - 1, 0)

    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def greedy_transversal(cliques_masks, n, mode="count"):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques.
        mode: 'count' = number of cliques, 'size' = sum of clique sizes, 'inverse' = prefer smaller cliques"""
        if not cliques_masks:
            return 0

        # Build incidence: for each vertex, list of clique indices it covers
        num_cliques = len(cliques_masks)
        vertex_cliques = [[] for _ in range(n)]
        for ci, cm in enumerate(cliques_masks):
            temp = cm
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                vertex_cliques[v].append(ci)
                temp ^= lsb

        uncovered = [True] * num_cliques
        selected = 0
        uncovered_count = num_cliques

        while uncovered_count > 0:
            best_v = -1
            best_score = -1
            for v in range(n):
                if selected & (1 << v):
                    continue
                score = 0
                for ci in vertex_cliques[v]:
                    if uncovered[ci]:
                        if mode == "count":
                            score += 1
                        elif mode == "size":
                            score += cliques_masks[ci].bit_count()
                        elif mode == "inverse":
                            score += 1.0 / max(1, cliques_masks[ci].bit_count())
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v < 0 or best_score <= 0:
                break
            selected |= (1 << best_v)
            for ci in vertex_cliques[best_v]:
                if uncovered[ci]:
                    uncovered[ci] = False
                    uncovered_count -= 1

        return selected

    def local_search_reduce(selected_mask, cliques_masks, n, max_iters=500):
        """Try to remove vertices; if it breaks coverage, stop.
        Uses clique-coverage counts for efficient checking."""
        if not cliques_masks:
            return selected_mask

        # For each clique, count how many selected vertices cover it
        num_cliques = len(cliques_masks)
        cover_count = [0] * num_cliques
        for ci, cm in enumerate(cliques_masks):
            cnt = 0
            temp = cm & selected_mask
            while temp:
                temp &= temp - 1
                cnt += 1
            cover_count[ci] = cnt

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for v in range(n):
                if not (selected_mask & (1 << v)):
                    continue
                # Check if any clique has cover_count == 1 and is covered by v
                # If so, removing v would break coverage
                v_bit = 1 << v
                removable = True
                for ci, cm in enumerate(cliques_masks):
                    if cover_count[ci] == 1 and (cm & v_bit):
                        removable = False
                        break
                if removable:
                    selected_mask ^= v_bit
                    # Update cover counts
                    for ci, cm in enumerate(cliques_masks):
                        if cm & v_bit:
                            cover_count[ci] -= 1
                    improved = True
                    break
        return selected_mask

    def local_search_swap(selected_mask, cliques_masks, n, max_iters=200):
        """Try 1-swap: remove vertex u, add vertex v if net size doesn't increase
        and coverage is maintained. Useful when greedy left redundant pairs."""
        if not cliques_masks:
            return selected_mask

        selected_vertices = [v for v in range(n) if selected_mask & (1 << v)]
        best_mask = selected_mask
        best_size = selected_mask.bit_count()

        for it in range(max_iters):
            improved = False
            for u in selected_vertices:
                trial_without_u = best_mask ^ (1 << u)
                # Find cliques not covered by trial_without_u
                missing = []
                for cm in cliques_masks:
                    if not (cm & trial_without_u):
                        missing.append(cm)
                if not missing:
                    # Can remove u without adding anything
                    best_mask = trial_without_u
                    best_size -= 1
                    selected_vertices.remove(u)
                    improved = True
                    break
                # Find a vertex v (not in best_mask) that covers all missing cliques
                for v in range(n):
                    if best_mask & (1 << v):
                        continue
                    covers_all = True
                    v_bit = 1 << v
                    for cm in missing:
                        if not (cm & v_bit):
                            covers_all = False
                            break
                    if covers_all:
                        # Swap u for v: same size, but may enable further reductions
                        best_mask = (best_mask ^ (1 << u)) | (1 << v)
                        selected_vertices[selected_vertices.index(u)] = v
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best_mask

    def solve_instance(n, p, seed, deadline):
        adj_mask = build_graph(n, p, seed)
        cliques = []
        bron_kerbosch_bitmask(n, adj_mask, cliques)

        if not cliques:
            return []

        best_mask = None
        best_size = n + 1

        # Try multiple greedy strategies
        for mode in ["count", "size", "inverse"]:
            mask = greedy_transversal(cliques, n, mode)
            mask = local_search_reduce(mask, cliques, n)
            mask = local_search_swap(mask, cliques, n)
            mask = local_search_reduce(mask, cliques, n)
            s = mask.bit_count()
            if s < best_size:
                best_size = s
                best_mask = mask

        # Additional local search passes
        if best_mask is not None:
            for _ in range(3):
                if time.time() > deadline:
                    break
                best_mask = local_search_reduce(best_mask, cliques, n)
                best_mask = local_search_swap(best_mask, cliques, n)
                best_mask = local_search_reduce(best_mask, cliques, n)
                s = best_mask.bit_count()
                if s < best_size:
                    best_size = s

        return [i for i in range(n) if best_mask & (1 << i)]

    start_time = time.time()
    deadline = start_time + 130

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices (guaranteed valid but large)
            out.append(list(range(n)))
            continue

        result = solve_instance(n, p, seed, deadline)
        out.append(result)

    return out
# EVOLVE-BLOCK-END
