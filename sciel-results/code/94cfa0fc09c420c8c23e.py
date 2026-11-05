import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(R, P, X, adj, n):
        if not P and not X:
            if len(R) >= 2:
                yield frozenset(R)
            return
        if P:
            pivot = max(P, key=lambda p: len(P & adj[p]))
        elif X:
            pivot = max(X, key=lambda p: len(P & adj[p]))
        else:
            return
        for v in list(P - adj[pivot]):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            yield from bron_kerbosch(new_R, new_P, new_X, adj, n)
            P.remove(v)
            X.add(v)

    def solve_ilp(n, valid_cliques):
        """Solve the hitting set problem exactly using scipy.optimize.milp."""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
        except ImportError:
            return None

        C = len(valid_cliques)
        if C == 0:
            return []

        # Variables: x_v for each vertex v (binary)
        # Minimize: sum x_v
        # Subject to: for each clique i, sum_{v in clique_i} x_v >= 1
        # 0 <= x_v <= 1

        c = np.ones(n)  # objective: minimize sum of x_v

        # Constraints: A @ x >= 1 for each clique
        # A[i, v] = 1 if v in clique i
        A = np.zeros((C, n))
        for i, clique in enumerate(valid_cliques):
            for v in clique:
                A[i, v] = 1

        # A @ x >= 1  =>  -A @ x <= -1
        # LinearConstraint expects: lb <= A_eq @ x <= ub
        # We can express as: A @ x >= 1 -> LinearConstraint(A, lb=np.ones(C), ub=np.inf)
        constraints = LinearConstraint(A, lb=np.ones(C), ub=np.inf)

        # Variable bounds: 0 <= x_v <= 1
        integrality = np.ones(n)  # all binary
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        if res.status == 0 and res.x is not None:
            selected = [v for v in range(n) if res.x[v] > 0.5]
            return sorted(selected)
        return None

    def greedy_local_search(n, valid_cliques, vertex_cliques, rng, deadline):
        """Fast greedy + local search as fallback."""
        C = len(valid_cliques)
        best_set = None
        best_size = float('inf')

        for run in range(200):
            if time.time() > deadline:
                break

            hit_cliques = set()
            current_set = set()

            while len(hit_cliques) < C:
                best_v = -1
                max_hits = -1
                candidates = []

                for v in range(n):
                    if v in current_set:
                        continue
                    hits = 0
                    for ci in vertex_cliques[v]:
                        if ci not in hit_cliques:
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                        candidates = [v]
                    elif hits == max_hits and hits > 0:
                        candidates.append(v)

                if max_hits == 0:
                    break

                v = rng.choice(candidates)
                current_set.add(v)
                for ci in vertex_cliques[v]:
                    hit_cliques.add(ci)

            # Local search: remove redundant vertices
            improved = True
            while improved:
                improved = False
                vs = list(current_set)
                rng.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        break
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        c = valid_cliques[ci]
                        found_other = False
                        for u in c:
                            if u != v and u in current_set:
                                found_other = True
                                break
                        if not found_other:
                            can_remove = False
                            break
                    if can_remove:
                        current_set.remove(v)
                        improved = True

            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)

        return sorted(best_set) if best_set else []

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        all_cliques = list(bron_kerbosch(frozenset(), set(range(n)), set(), adj, n))
        valid_cliques = [c for c in all_cliques if len(c) >= 2]

        if not valid_cliques:
            return []

        C = len(valid_cliques)

        vertex_cliques = defaultdict(set)
        for i, c in enumerate(valid_cliques):
            for v in c:
                vertex_cliques[v].add(i)

        # Try ILP first - should be near-instant for n<=40
        ilp_result = solve_ilp(n, valid_cliques)
        if ilp_result is not None:
            return ilp_result

        # Fallback: greedy + local search
        rng_solver = random.Random(seed + 123)
        return greedy_local_search(n, valid_cliques, vertex_cliques, rng_solver, deadline)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue

        res = solve_instance(n, p, seed)
        results.append(res)

    return results
