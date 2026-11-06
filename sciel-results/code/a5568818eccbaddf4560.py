import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin below 150s wall-clock kill

    def elapsed():
        return time.time() - start_time

    def deadline_hit():
        return elapsed() > DEADLINE

    # ---------- Graph construction ----------
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n  # bitmask adjacency
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    # ---------- Bron-Kerbosch (bitmask) ----------
    def maximal_cliques(adj):
        n = len(adj)
        full = (1 << n) - 1
        cliques = []

        def bk(R, P, X):
            if not P and not X:
                if R:  # size >= 1 (we'll filter >= 2 outside)
                    cliques.append(R)
                return
            if not P:
                return
            # pivot
            if P:
                pivot = max(P, key=lambda v: (P & adj[v]).bit_count())
            else:
                pivot = max(X, key=lambda v: (P & adj[v]).bit_count())
            candidates = P & ~adj[pivot]
            while candidates:
                v = (candidates & -candidates).bit_length() - 1
                bit = 1 << v
                candidates ^= bit
                R2 = R | bit
                P2 = P & adj[v]
                X2 = X & adj[v]
                bk(R2, P2, X2)
                P &= ~bit
                X |= bit

        bk(0, full, 0)
        return [c for c in cliques if c.bit_count() >= 2]

    # ---------- Greedy set cover (hitting set on cliques) ----------
    def greedy_cover(cliques, n):
        if not cliques:
            return 0
        # uncovered bitset of cliques
        m = len(cliques)
        uncovered = (1 << m) - 1
        # count[v] = number of uncovered cliques containing v
        count = [0] * n
        for ci in range(m):
            c = cliques[ci]
            # iterate set bits
            cc = c
            while cc:
                v = (cc & -cc).bit_length() - 1
                cc &= cc - 1
                count[v] += 1
        S = 0
        while uncovered:
            best_v = -1
            best_c = -1
            for v in range(n):
                if count[v] > best_c:
                    best_c = count[v]
                    best_v = v
            if best_v < 0:
                break
            S |= (1 << best_v)
            # remove covered cliques
            cc = cliques
            # for each clique containing best_v, clear it
            for ci in range(m):
                if (uncovered >> ci) & 1:
                    if (cliques[ci] >> best_v) & 1:
                        uncovered &= ~(1 << ci)
                        # decrement counts for other vertices in this clique
                        c = cliques[ci]
                        while c:
                            v = (c & -c).bit_length() - 1
                            c &= c - 1
                            if v != best_v:
                                count[v] -= 1
            count[best_v] = 0
        return S

    # ---------- Maximum matching based lower bound / initial cover ----------
    # For a general graph, use a simple greedy matching to get a lower bound
    # and use Konig's theorem on the bipartite double cover is complex.
    # Instead: greedy maximal matching gives a lower bound on vertex cover size.
    # We use a simple greedy approach for initial solution refinement.

    def greedy_vertex_cover(adj, n):
        """Simple greedy: repeatedly pick vertex of max degree among uncovered edges."""
        remaining_edges = 0
        adj_mask = list(adj)
        S = 0
        # Track uncovered edges via adjacency
        for _ in range(n):
            if deadline_hit():
                break
            # Find vertex with max number of uncovered incident edges
            best_v = -1
            best_deg = -1
            for v in range(n):
                if (S >> v) & 1:
                    continue
                # count neighbors not in S
                neigh = adj_mask[v] & ~S
                deg = neigh.bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v < 0 or best_deg == 0:
                break
            S |= (1 << best_v)
        return S

    # ---------- Branch and Bound ----------
    def branch_and_bound(cliques, n, initial_S, time_limit):
        """
        Exact B&B for minimum hitting set on cliques.
        Uses disjoint clique lower bound.
        Returns best solution found (bitmask) or initial_S if time runs out.
        """
        if not cliques:
            return initial_S

        m = len(cliques)
        # Precompute for each vertex the bitmask of cliques it hits
        vertex_cliques = [0] * n
        for ci in range(m):
            c = cliques[ci]
            cc = c
            while cc:
                v = (cc & -cc).bit_length() - 1
                cc &= cc - 1
                vertex_cliques[v] |= (1 << ci)

        best_S = initial_S
        best_size = initial_S.bit_count()

        # Sort cliques by size descending for better pruning
        # Actually, sort vertices by frequency descending
        # Build list of cliques as lists of vertices for fast access
        clique_lists = []
        for c in cliques:
            verts = []
            cc = c
            while cc:
                v = (cc & -cc).bit_length() - 1
                cc &= cc - 1
                verts.append(v)
            clique_lists.append(verts)

        # uncovered: bitmask of cliques not yet hit
        # We'll do a recursive search
        # State: (current_S, uncovered_cliques, lower_bound)

        # For lower bound: find max number of pairwise disjoint cliques among uncovered
        # This is a maximum independent set on the intersection graph of cliques
        # which is NP-hard, so we use a greedy approximation

        def disjoint_lower_bound(uncovered):
            """Greedy: repeatedly pick smallest uncovered clique, remove all intersecting."""
            if not uncovered:
                return 0
            # Get list of uncovered clique indices
            ub = uncovered
            count = 0
            while ub:
                # Pick clique with fewest remaining (smallest)
                best_ci = -1
                best_size = n + 1
                uu = ub
                while uu:
                    ci = (uu & -uu).bit_length() - 1
                    uu &= uu - 1
                    sz = len(clique_lists[ci])
                    if sz < best_size:
                        best_size = sz
                        best_ci = ci
                if best_ci < 0:
                    break
                count += 1
                # Remove all cliques that intersect with clique_lists[best_ci]
                inter_mask = 0
                for v in clique_lists[best_ci]:
                    inter_mask |= vertex_cliques[v]
                ub &= ~inter_mask
            return count

        # Recursive B&B
        # To avoid deep recursion, use iterative approach with a stack
        # But Python recursion with limit is fine for small problems
        sys.setrecursionlimit(10000)

        def search(current_S, uncovered, depth):
            nonlocal best_S, best_size
            if deadline_hit():
                return

            if not uncovered:
                sz = current_S.bit_count()
                if sz < best_size:
                    best_size = sz
                    best_S = current_S
                return

            # Lower bound
            lb = current_S.bit_count() + disjoint_lower_bound(uncovered)
            if lb >= best_size:
                return

            # Pick a clique to branch on (smallest uncovered)
            # Find smallest uncovered clique
            best_ci = -1
            best_sz = n + 1
            uu = uncovered
            while uu:
                ci = (uu & -uu).bit_length() - 1
                uu &= uu - 1
                sz = len(clique_lists[ci])
                if sz < best_sz:
                    best_sz = sz
                    best_ci = ci
            if best_ci < 0:
                return

            clique = clique_lists[best_ci]
            # Branch on each vertex in this clique
            for v in clique:
                if (current_S >> v) & 1:
                    continue
                new_S = current_S | (1 << v)
                # New uncovered: remove all cliques hit by v
                new_uncovered = uncovered & ~vertex_cliques[v]
                search(new_S, new_uncovered, depth + 1)
                if deadline_hit():
                    return

        search(initial_S, (1 << m) - 1, 0)
        return best_S

    # ---------- Local search polish ----------
    def local_search_polish(S, cliques, n):
        """Try to remove redundant vertices and swap to smaller ones."""
        if not cliques:
            return S
        changed = True
        while changed and not deadline_hit():
            changed = False
            # Remove redundant
            for v in range(n):
                if (S >> v) & 1:
                    # Check if v is essential
                    essential = False
                    for c in cliques:
                        if (c >> v) & 1:
                            # Check if any other vertex in S is in c
                            others = c & S & ~(1 << v)
                            if others == 0:
                                essential = True
                                break
                    if not essential:
                        S &= ~(1 << v)
                        changed = True
        return S

    # ---------- Solve one instance ----------
    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        cliques = maximal_cliques(adj)

        if not cliques:
            return []

        # Initial solution: greedy cover
        S_greedy = greedy_cover(cliques, n)
        # Also try greedy vertex cover
        S_vc = greedy_vertex_cover(adj, n)

        # Pick better initial
        if S_vc.bit_count() < S_greedy.bit_count():
            S_init = S_vc
        else:
            S_init = S_greedy

        # Local search polish
        S_init = local_search_polish(S_init, cliques, n)

        # Branch and bound
        S_best = branch_and_bound(cliques, n, S_init, time_limit=10.0)

        # Final polish
        S_best = local_search_polish(S_best, cliques, n)

        # Convert to list
        result = []
        sb = S_best
        while sb:
            v = (sb & -sb).bit_length() - 1
            sb &= sb - 1
            result.append(v)
        return sorted(result)

    # ---------- Main ----------
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if deadline_hit():
            # Fallback: just return greedy for remaining
            adj = build_graph(n, p, seed)
            cliques = maximal_cliques(adj)
            if cliques:
                S = greedy_cover(cliques, n)
                S = local_search_polish(S, cliques, n)
            else:
                S = 0
            result = []
            sb = S
            while sb:
                v = (sb & -sb).bit_length() - 1
                sb &= sb - 1
                result.append(v)
            results.append(sorted(result))
        else:
            results.append(solve_instance(n, p, seed))

    return results
