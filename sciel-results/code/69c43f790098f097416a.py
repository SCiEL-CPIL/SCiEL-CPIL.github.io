import time
import random
import heapq

_GLOBAL_START = time.time()
_GLOBAL_BUDGET = 135.0  # safety margin under the 150 s wall-clock kill

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X):
        if not P and not X:
            if len(R) >= 2:
                yield tuple(sorted(R))
            return
        if P or X:
            u = max(P | X, key=lambda v: len(P & adj[v]))
            candidates = P - adj[u]
        else:
            candidates = set()
        for v in candidates:
            yield from bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    def get_maximal_cliques(adj):
        n = len(adj)
        return list(bron_kerbosch(adj, set(), set(range(n)), set()))

    def make_cover_matrix(cliques, n):
        # cover[v] = frozenset of clique indices that contain vertex v
        cover = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                cover[v].add(ci)
        return [frozenset(s) for s in cover]

    def greedy_cover(cliques, cover, uncovered, rng):
        # uncovered: set of clique indices
        selected = set()
        rem = set(uncovered)
        # counts: for each vertex, number of currently-uncovered cliques it hits
        count = [0] * len(cover)
        for ci in rem:
            for v in cliques[ci]:
                count[v] += 1
        while rem:
            best_v = -1
            best_c = -1
            ties = []
            for v in range(len(cover)):
                cv = count[v]
                if cv <= 0:
                    continue
                if cv > best_c:
                    best_c = cv
                    best_v = v
                    ties = [v]
                elif cv == best_c:
                    ties.append(v)
            if best_v == -1:
                break
            if len(ties) > 1:
                best_v = rng.choice(ties)
            selected.add(best_v)
            for ci in list(cover[best_v]):
                if ci in rem:
                    rem.discard(ci)
                    for v in cliques[ci]:
                        count[v] -= 1
        return selected

    def local_search_fast(selected, cliques, cover_bm, all_bm, n, rng, iters):
        """Fast local search using bitmask coverage. cover_bm[v] is a bitmask of cliques hit by v."""
        selected = set(selected)
        sel_bm = 0
        for v in selected:
            sel_bm |= cover_bm[v]
        for _ in range(iters):
            improved = False
            # 1) removal
            for v in list(selected):
                rest_bm = sel_bm & ~cover_bm[v]
                if rest_bm == all_bm:
                    selected.discard(v)
                    sel_bm = rest_bm
                    improved = True
                    break
            if improved:
                continue
            # 2) 1-exchange: remove v, add w (same size, but w covers more)
            best_w = -1
            best_v = -1
            best_gain = 0
            sel_list = list(selected)
            for v in sel_list:
                rest_bm = sel_bm & ~cover_bm[v]
                need = all_bm & ~rest_bm
                if need == 0:
                    continue  # would be a removal
                # Only consider w that cover all of need
                # Quick filter: w must have at least popcount(need) bits in common with need
                need_count = bin(need).count('1')
                for w in range(n):
                    if w in selected:
                        continue
                    if (cover_bm[w] & need) == need:
                        new_bm = rest_bm | cover_bm[w]
                        gain = bin(new_bm ^ sel_bm).count('1')
                        if gain > best_gain:
                            best_gain = gain
                            best_w = w
                            best_v = v
                            if best_gain >= 3:
                                break  # good enough, don't spend more time
                if best_gain >= 3:
                    break
            if best_w >= 0:
                selected.discard(best_v)
                selected.add(best_w)
                sel_bm = (sel_bm & ~cover_bm[best_v]) | cover_bm[best_w]
                continue
            break
        return selected

    def exact_branch_and_bound(cliques, cover_bm, all_bm, budget_fn, limit=5000000):
        """Exact B&B using bitmasks. Returns (best_set, solved_bool). best_set is always valid."""
        n = len(cover_bm)
        best = [set()]
        best_size = [10**9]
        solved = [False]
        nodes = [0]
        stop = [False]

        def branch(selected, sel_bm, depth):
            if stop[0]:
                return
            if sel_bm == all_bm:
                if len(selected) < best_size[0]:
                    best_size[0] = len(selected)
                    best[0] = set(selected)
                solved[0] = True
                return
            nodes[0] += 1
            if nodes[0] > limit:
                stop[0] = True
                return
            if not budget_fn():
                stop[0] = True
                return
            # Lower bound: greedy set cover on uncovered cliques
            unhit = all_bm & ~sel_bm
            lb = 0
            rem = unhit
            while rem:
                best_cov = 0
                best_v = -1
                for v in range(n):
                    c = bin(cover_bm[v] & rem).count('1')
                    if c > best_cov:
                        best_cov = c
                        best_v = v
                if best_cov == 0:
                    break
                lb += 1
                rem &= ~cover_bm[best_v]
            if len(selected) + lb >= best_size[0]:
                return

            # Choose an uncovered clique with fewest candidate vertices that cover it
            unhit_ci_list = [ci for ci in range(len(cliques)) if (all_bm & (1 << ci)) and not (sel_bm & (1 << ci))]
            if not unhit_ci_list:
                solved[0] = True
                return
            # Pick the clique that is "hardest" to cover: fewest vertices that hit it AND are not yet selected
            def clique_hardness(ci):
                cnt = 0
                for v in cliques[ci]:
                    if v not in selected:
                        cnt += 1
                return cnt
            rep_ci = min(unhit_ci_list, key=clique_hardness)
            # Candidate vertices in rep_ci
            verts = [v for v in cliques[rep_ci] if v not in selected]
            # Order by number of unhit cliques they cover (descending)
            verts.sort(key=lambda v: bin(cover_bm[v] & unhit).count('1'), reverse=True)
            for v in verts:
                new_bm = sel_bm | cover_bm[v]
                selected.add(v)
                branch(selected, new_bm, depth + 1)
                selected.discard(v)
                if stop[0]:
                    return

        branch(set(), 0, 0)
        return best[0], solved[0]

    def solve_instance(n, p, seed, time_budget):
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        if not cliques:
            return []
        cover = make_cover_matrix(cliques, n)
        all_uncovered = frozenset(range(len(cliques)))

        # Build bitmask coverage
        num_cliques = len(cliques)
        cover_bm = [0] * n
        for ci, c in enumerate(cliques):
            bit = 1 << ci
            for v in c:
                cover_bm[v] |= bit
        all_bm = (1 << num_cliques) - 1

        # Greedy + fast local search baseline
        rng = random.Random(777)
        g = greedy_cover(cliques, cover, all_uncovered, rng)
        g = local_search_fast(g, cliques, cover_bm, all_bm, n, rng, iters=50)
        best = set(g)

        # Time budget for exact + extra local search
        def budget_fn():
            return (time.time() - _GLOBAL_START) < _GLOBAL_BUDGET

        # Exact branch-and-bound (with node limit and time guard)
        eset, solved = exact_branch_and_bound(cliques, cover_bm, all_bm, budget_fn)
        if eset and len(eset) < len(best):
            best = set(eset)

        # If exact solved, we're done (optimal)
        if solved:
            return sorted(best)

        # Use remaining time for more greedy restarts + fast local search
        max_restarts = 200
        for it in range(max_restarts):
            if (time.time() - _GLOBAL_START) >= _GLOBAL_BUDGET:
                break
            rr = random.Random(100000 + it)
            cand = greedy_cover(cliques, cover, all_uncovered, rr)
            cand = local_search_fast(cand, cliques, cover_bm, all_bm, n, rr, iters=20)
            if len(cand) < len(best):
                best = set(cand)
            # also try local search starting from current best with fresh rng
            cand2 = local_search_fast(set(best), cliques, cover_bm, all_bm, n, rr, iters=15)
            if len(cand2) < len(best):
                best = set(cand2)
        return sorted(best)

    out = []
    instances = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))
    per_instance = (_GLOBAL_BUDGET - (time.time() - _GLOBAL_START)) / len(instances)
    for i, (n, p, seed) in enumerate(instances):
        # hard cap: leave at least a small margin for the rest
        remaining = _GLOBAL_BUDGET - (time.time() - _GLOBAL_START)
        tb = max(1.0, min(per_instance, remaining - 2.0))
        res = solve_instance(n, p, seed, tb)
        out.append(res)
        # if we're running out, just fill the rest cheaply
        if (time.time() - _GLOBAL_START) > _GLOBAL_BUDGET - 1.0:
            for (n2, p2, s2) in instances[i + 1:]:
                out.append(sorted(range(n2)))  # safe (valid) but large
            break
    return out
