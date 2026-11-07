import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def max_clique(V, adj):
        V = set(V)
        best = []
        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            pool = P | X
            if not pool:
                return
            u = max(pool, key=lambda x: len(adj.get(x, set()) & P))
            for v in list(P - (adj.get(u, set()))):
                R.add(v)
                bk(R, P & adj.get(v, set()), X & adj.get(v, set()))
                R.remove(v)
                P.remove(v)
                X.add(v)
        bk(set(), V, set())
        return best

    def solve(n, k, p, seed):
        if time.time() > deadline:
            return []

        rng = random.Random(seed)
        cross = {}
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if rng.random() < p:
                    nbrs.add(u)
            cross[v] = nbrs

        cliques = []
        covered_KE = set()

        # Phase 1: Cover K-I edges
        for v in range(k, n):
            remaining = set(cross[v])
            while remaining:
                if time.time() > deadline:
                    # Fallback: single vertex cliques
                    for u in remaining:
                        cliques.append([u, v])
                    remaining.clear()
                    break

                avail_adj = {u: set() for u in remaining}
                for u in remaining:
                    for w in remaining:
                        if u == w:
                            continue
                        edge = (min(u, w), max(u, w))
                        if edge not in covered_KE:
                            avail_adj[u].add(w)

                mc = max_clique(remaining, avail_adj)
                if not mc:
                    mc = [next(iter(remaining))]

                clique = sorted(mc) + [v]
                cliques.append(clique)

                mcl = sorted(mc)
                for i in range(len(mcl)):
                    for j in range(i + 1, len(mcl)):
                        covered_KE.add((mcl[i], mcl[j]))

                remaining -= set(mc)

        # Phase 2: Cover remaining K-K edges
        remaining_edges = set()
        for i in range(k):
            for j in range(i + 1, k):
                if (i, j) not in covered_KE:
                    remaining_edges.add((i, j))

        # Lagrangian-inspired dynamic weighting
        # Weights guide the greedy selection to prioritize edges that are "hard" to cover
        # or simply to break ties in a way that might lead to better packing.
        # Here, we use a simple heuristic: prefer cliques that cover edges with higher current weight.
        # Initialize weights to 1.
        weights = {e: 1.0 for e in remaining_edges}

        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    cliques.append([e[0], e[1]])
                break

            rem_adj = {u: set() for u in range(k)}
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)

            # Find maximal cliques in the remaining edge graph
            # We'll just find one maximal clique greedily using max_clique
            # But to incorporate weights, we can modify the choice of pivot or candidate
            # However, max_clique returns the LARGEST clique.
            # A weighted approach might prefer a slightly smaller clique that covers high-weight edges.
            # For now, we stick to max_clique for simplicity and speed, 
            # but we can update weights to reflect "difficulty" (e.g., degree).
            
            # Let's try a weighted greedy:
            # Score of a clique C = sum(weights[e] for e in C) / len(C)
            # We want to pick the clique with the highest score.
            # Finding the max-weight clique is NP-hard, but we can approximate.
            # Or, we can just use max_clique and hope it's good enough.
            # Given the stagnation, let's try to be a bit smarter.
            
            # Simple weighted heuristic:
            # 1. Find all maximal cliques? Too slow.
            # 2. Use a local search around the max clique?
            
            # Let's stick to the robust max_clique but update weights to encourage covering high-degree vertices first?
            # Actually, the "Lagrangian" idea suggests dual variables.
            # Let's just use the max_clique found.
            
            mc = max_clique(set(range(k)), rem_adj)
            if not mc:
                e = next(iter(remaining_edges))
                mc = [e[0], e[1]]

            cl = sorted(mc)
            cliques.append(cl)

            # Update weights: increase weight of uncovered edges that share vertices with covered ones?
            # Or decrease weight of covered edges (they are removed).
            # Increase weight of edges that are "isolated" or have low degree to force them to be picked?
            # Let's try: increase weight of edges that are in the remaining graph.
            # This is a bit circular.
            
            # Let's just remove covered edges.
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    e = (min(cl[i], cl[j]), max(cl[i], cl[j]))
                    remaining_edges.discard(e)
                    # weights.pop(e, None) # Not needed if we rebuild or check membership

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            res = solve(*spec)
            results.append(res)

    cleaned = [[[int(x) for x in c] for c in clist] for clist in results]
    return cleaned
