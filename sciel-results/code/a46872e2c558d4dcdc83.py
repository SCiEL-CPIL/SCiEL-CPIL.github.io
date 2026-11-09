import time
import random as _rnd
import math

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill

    def generate_edges(n, p, seed):
        rng = _rnd.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def bron_kerbosch(adj, n):
        cliques = []
        all_mask = (1 << n) - 1

        def bk(R, P, X):
            if not P and not X:
                if R:
                    cliques.append(frozenset(i for i in range(n) if (R >> i) & 1))
                return
            PX = P | X
            if PX:
                best_p = -1
                best_cnt = -1
                pp = PX
                while pp:
                    lb = pp & (-pp)
                    p = lb.bit_length() - 1
                    cnt = (adj[p] & P).bit_count()
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_p = p
                    pp ^= lb
                candidates = P & ~(adj[best_p])
            else:
                candidates = P

            cp = candidates
            while cp:
                lb = cp & (-cp)
                u = lb.bit_length() - 1
                cp ^= lb
                bk(R | lb, P & adj[u], X & adj[u])
                P ^= lb
                X |= lb

        bk(0, all_mask, 0)
        return cliques

    def filter_cliques(cliques, min_size=2):
        return [c for c in cliques if len(c) >= min_size]

    def greedy_hitting_set(cliques, n):
        uncovered = set(range(len(cliques)))
        vertex_cliques = [[] for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(ci)

        hit = set()
        while uncovered:
            best_v = -1
            best_cnt = -1
            for v in range(n):
                cnt = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_cnt <= 0:
                break
            hit.add(best_v)
            for ci in vertex_cliques[best_v]:
                uncovered.discard(ci)
        return hit

    def branch_and_bound(cliques, n, time_limit, initial_hit=None):
        t0 = time.time()
        if initial_hit is not None:
            best_hit = set(initial_hit)
        else:
            best_hit = greedy_hitting_set(cliques, n)
        best_size = len(best_hit)
        
        num_cliques = len(cliques)
        if num_cliques == 0:
            return best_hit
            
        # Precompute which cliques each vertex hits
        vertex_clique_indices = [[] for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_clique_indices[v].append(ci)
        
        # For each vertex, precompute a bitmask of cliques it covers
        vertex_clique_mask = [0] * n
        for ci in range(num_cliques):
            mask_bit = 1 << ci
            for v in cliques[ci]:
                vertex_clique_mask[v] |= mask_bit
        
        def lb_disjoint_cliques(remaining_mask):
            """Lower bound via maximum set of pairwise disjoint cliques in remaining set."""
            if remaining_mask == 0:
                return 0
            # Greedy: repeatedly pick a smallest clique, remove all overlapping
            count = 0
            m = remaining_mask
            while m:
                # Find smallest clique in m
                best_ci = -1
                best_size = 999
                mm = m
                while mm:
                    lb = mm & (-mm)
                    ci = lb.bit_length() - 1
                    sz = len(cliques[ci])
                    if sz < best_size:
                        best_size = sz
                        best_ci = ci
                    mm ^= lb
                if best_ci < 0:
                    break
                count += 1
                # Remove best_ci and all cliques that intersect it
                overlap_mask = 0
                for v in cliques[best_ci]:
                    overlap_mask |= vertex_clique_mask[v]
                remaining_mask &= ~overlap_mask
                m &= ~overlap_mask
            return count

        # State: (remaining_mask, hit_set)
        # We track the set of chosen vertices for the current path
        # To avoid deep recursion copying large sets, we use a bitset for chosen vertices if n is small, or just set.
        # Given n <= 40, a set of ints is fine, but bitmask for chosen vertices is faster for hashing/ops?
        # Let's stick to sets for chosen vertices for clarity, or bitmask.
        # Bitmask for chosen vertices:
        
        def solve(remaining_mask, chosen_mask, chosen_count):
            nonlocal best_hit, best_size, t0
            if time.time() - t0 > time_limit:
                return False
            
            if remaining_mask == 0:
                if chosen_count < best_size:
                    best_size = chosen_count
                    # Convert chosen_mask back to set of vertices
                    new_hit = set(i for i in range(n) if (chosen_mask >> i) & 1)
                    best_hit = new_hit
                return True
            
            # Lower bound
            lb = lb_disjoint_cliques(remaining_mask)
            if chosen_count + lb >= best_size:
                return True
            
            # Variable selection: pick a clique to branch on
            # Heuristic: pick the smallest clique, or one with high "constraint"
            m = remaining_mask
            best_ci = -1
            best_clique_size = 999
            while m:
                lb_bit = m & (-m)
                ci = lb_bit.bit_length() - 1
                sz = len(cliques[ci])
                if sz < best_clique_size:
                    best_clique_size = sz
                    best_ci = ci
                m ^= lb_bit
            
            if best_ci < 0:
                return True
                
            target_clique = cliques[best_ci]
            # Order vertices in this clique by how many remaining cliques they cover (descending)
            verts = list(target_clique)
            verts.sort(key=lambda v: (vertex_clique_mask[v] & remaining_mask).bit_count(), reverse=True)
            
            for v in verts:
                if (chosen_mask >> v) & 1:
                    continue
                new_chosen_mask = chosen_mask | (1 << v)
                # New remaining: remove all cliques hit by v
                hit_mask = vertex_clique_mask[v]
                new_remaining = remaining_mask & ~hit_mask
                if not solve(new_remaining, new_chosen_mask, chosen_count + 1):
                    return False
            return True

        solve((1 << num_cliques) - 1, 0, 0)
        return best_hit

    def local_search_polish(hit, cliques, n, time_limit):
        hit = set(hit)
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < time_limit:
            improved = False
            # 1. Try to remove any vertex
            for v in list(hit):
                if time.time() - t0 > time_limit:
                    break
                trial = hit - {v}
                valid = True
                for c in cliques:
                    if not (c & trial):
                        valid = False
                        break
                if valid:
                    hit = trial
                    improved = True
                    break
            if not improved:
                # 2. Try 1-swap: remove v, add w
                for v in list(hit):
                    if time.time() - t0 > time_limit:
                        break
                    trial_base = hit - {v}
                    for w in range(n):
                        if w in hit:
                            continue
                        trial = trial_base | {w}
                        # Check validity
                        valid = True
                        for c in cliques:
                            if not (c & trial):
                                valid = False
                                break
                        if valid:
                            hit = trial
                            improved = True
                            break
                    if improved:
                        break
        return hit

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        remaining = deadline - time.time()
        if remaining < 15:
            # Fallback: all vertices
            results.append(list(range(n)))
            continue

        edges = generate_edges(n, p, seed)
        adj = build_adjacency(n, edges)
        cliques = bron_kerbosch(adj, n)
        cliques = filter_cliques(cliques, min_size=2)

        if not cliques:
            results.append([])
            continue

        # Initial greedy solution
        hit = greedy_hitting_set(cliques, n)
        
        # Branch and bound with time budget
        bnb_time = min(remaining * 0.7, 40.0)
        hit_bnb = branch_and_bound(cliques, n, bnb_time, initial_hit=hit)
        
        if len(hit_bnb) < len(hit):
            hit = hit_bnb

        # Local search polish
        ls_time = min(remaining * 0.3, 10.0)
        hit = local_search_polish(hit, cliques, n, ls_time)

        # Final validation
        valid = True
        for c in cliques:
            if not (c & hit):
                valid = False
                break
        
        if not valid:
            # Fallback to all vertices involved
            all_v = set()
            for c in cliques:
                all_v |= c
            hit = all_v

        results.append(sorted(hit))

    return results
