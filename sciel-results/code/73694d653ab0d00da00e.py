import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
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
        return sorted(edges)

    results = []
    start_time = time.time()
    deadline = start_time + 110.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        rng = random.Random(seed)
        neighbors = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        # Build edge list
        all_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                all_edges.append((u, w))
        for v in range(k, n):
            for u in range(k):
                if u in neighbors[v]:
                    all_edges.append((min(u, v), max(u, v)))
        
        edge_to_idx = {e: i for i, e in enumerate(all_edges)}
        num_edges = len(all_edges)

        # Find all maximal cliques of size >= 2
        # For split graphs, maximal cliques are either:
        # 1. Subsets of C (clique vertices)
        # 2. {v} union S where v in I and S subset of N(v) in C
        # We need to enumerate candidate cliques efficiently.
        
        # Strategy: For each vertex v, find maximal cliques containing v.
        # Use Bron-Kerbosch with pivoting.
        
        def bron_kerbosch(R, P, X):
            """Find all maximal cliques using Bron-Kerbosch with pivoting."""
            if not P and not X:
                if len(R) >= 2:
                    yield frozenset(R)
                return
            
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            pivot = None
            max_deg = -1
            for v in P:
                deg = len(P & neighbors[v])
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
            if pivot is None:
                for v in X:
                    deg = len(P & neighbors[v])
                    if deg > max_deg:
                        max_deg = deg
                        pivot = v
            
            if pivot is not None:
                candidates = [v for v in P if pivot not in neighbors[v]]
            else:
                candidates = list(P)
            
            for v in candidates:
                R.add(v)
                P_new = P & neighbors[v]
                X_new = X & neighbors[v]
                yield from bron_kerbosch(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)
        
        # For efficiency, we'll use a smarter approach for split graphs.
        # In a split graph, every maximal clique is of the form:
        # - A subset of C that is a clique (all subsets of C are cliques since C is a clique)
        # - {v} ∪ S where v ∈ I and S ⊆ N_C(v)
        
        # Since C is a clique, any subset of C is a clique.
        # For vertices in I, a clique containing v must be {v} ∪ S where S ⊆ N_C(v).
        # The maximal cliques containing v are {v} ∪ S where S is maximal in N_C(v) under inclusion...
        # but since C is a clique, N_C(v) itself is a clique, so {v} ∪ N_C(v) is the unique maximal clique containing v.
        
        # Wait, that's not right. In a split graph, C is a clique and I is an independent set.
        # A clique can contain at most one vertex from I.
        # So maximal cliques are:
        # 1. C itself (and subsets, but C is the only maximal one in C)
        # 2. For each v in I with N_C(v) non-empty: {v} ∪ N_C(v)
        
        # But we might want smaller cliques too for the partition.
        # Let's enumerate all cliques of size >= 2 as candidates.
        
        # For practical purposes, let's enumerate:
        # - All pairs (edges)
        # - All triples that form cliques
        # - Larger cliques from the maximal structure
        
        # Actually, for the ILP, we only need to consider maximal cliques or a useful subset.
        # Let's think about this differently.
        
        # In a split graph, any clique has at most one vertex from I.
        # So cliques are either:
        # - Subsets of C (size >= 2)
        # - {v} ∪ S where v in I, S subset of N_C(v), |S| >= 1
        
        # The key insight: we can decompose the problem.
        # Edges within C must be covered by cliques that are subsets of C.
        # Edges between C and I must be covered by cliques of the form {v} ∪ S.
        
        # Let's use a different structural approach:
        # For each v in I, the edges (v, u) for u in N_C(v) must all be in the same clique or different cliques.
        # If we put them all in one clique {v} ∪ N_C(v), that covers all those edges in one clique.
        # But that clique also covers all edges within N_C(v).
        
        # So the trade-off is:
        # - Using {v} ∪ N_C(v) covers |N_C(v)| edges incident to v plus C(C(N_C(v))) edges within N_C(v)
        # - Using separate edges covers |N_C(v)| edges but doesn't cover C-C edges
        
        # Let's build the ILP properly.
        # Variables: x_S for each candidate clique S (binary)
        # Constraint: for each edge e, sum_{S: e in S} x_S = 1
        # Objective: minimize sum x_S
        
        # Candidate cliques:
        # 1. For each v in I with N_C(v) non-empty: {v} ∪ S for all non-empty S ⊆ N_C(v)
        #    But this is exponential. We need to be smarter.
        
        # Key observation: In the optimal solution, for each v in I, all edges (v,u) for u in N_C(v)
        # are either in one clique {v} ∪ S_v, or split across multiple cliques.
        # If split, each clique containing v has form {v} ∪ S where S ⊆ N_C(v).
        
        # For the ILP to be tractable, let's limit candidates:
        # - For each v in I: the full clique {v} ∪ N_C(v)
        # - For each v in I and each u in N_C(v): the edge {v, u}
        # - For subsets of C: all edges (u,w) in C, and larger cliques
        
        # Actually, let's think about what cliques are useful.
        # A clique S ⊆ C is useful if it covers edges within C.
        # A clique {v} ∪ S is useful if it covers edges (v,u) for u in S.
        
        # The constraint is that each edge is covered exactly once.
        
        # Let's use a simpler but effective approach:
        # Phase 1: For each v in I, decide whether to use {v} ∪ N_C(v) or break it up.
        # Phase 2: Cover remaining C-C edges with cliques in C.
        
        # This is still complex. Let's use the ILP with a limited set of candidates.
        
        # Candidate set:
        # 1. For each v in I with |N_C(v)| >= 2: clique {v} ∪ N_C(v)
        # 2. For each v in I and u in N_C(v): clique {v, u}
        # 3. For each edge (u,w) in C: clique {u, w}
        # 4. For each subset S of C with |S| >= 3 that forms a "useful" clique
        
        # For (4), in a split graph C is a clique, so any subset works.
        # But we don't want all subsets. Let's add maximal cliques in C.
        # Since C is a clique, the only maximal clique in C is C itself.
        
        # Let's try: candidates = all edges + all {v}∪N_C(v) + C itself
        # This might not be optimal but let's see.
        
        # Actually, let's be more systematic. The ILP formulation:
        # We have edges E. For each edge e, we need exactly one clique containing it.
        
        # Let's generate a reasonable set of candidate cliques:
        candidates = set()
        
        # Edges (size 2 cliques)
        for e in all_edges:
            candidates.add(frozenset(e))
        
        # For each v in I, add {v} ∪ N_C(v) if size >= 2
        for v in range(k, n):
            N_v = neighbors[v] & set(range(k))
            if len(N_v) >= 1:
                cl = frozenset([v] + list(N_v))
                candidates.add(cl)
        
        # Add C itself if size >= 2
        if k >= 2:
            candidates.add(frozenset(range(k)))
        
        # Add some larger cliques in C: for each pair in C, try extending
        # Actually, since C is a clique, any subset is a clique.
        # Let's add triples in C that might be useful
        for u in range(k):
            for w in range(u+1, k):
                for x in range(w+1, k):
                    if (u,w) in edge_to_idx and (u,x) in edge_to_idx and (w,x) in edge_to_idx:
                        candidates.add(frozenset([u, w, x]))
        
        # Convert to list
        candidate_list = list(candidates)
        num_candidates = len(candidate_list)
        
        # Build constraint matrix: A[i,j] = 1 if edge i is in clique j
        # For efficiency, use sparse representation
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Build incidence: for each candidate clique, which edges it covers
        clique_edges = []
        for cl in candidate_list:
            verts = sorted(cl)
            edges_in_clique = []
            for a, b in combinations(verts, 2):
                if (a, b) in edge_to_idx:
                    edges_in_clique.append(edge_to_idx[(a, b)])
            clique_edges.append(edges_in_clique)
        
        # Build A matrix (sparse)
        # A[e, c] = 1 if edge e is in clique c
        rows = []
        cols = []
        vals = []
        for c_idx, edge_list in enumerate(clique_edges):
            for e_idx in edge_list:
                rows.append(e_idx)
                cols.append(c_idx)
                vals.append(1)
        
        from scipy.sparse import csr_matrix
        A = csr_matrix((vals, (rows, cols)), shape=(num_edges, num_candidates))
        
        # Constraints: A @ x = 1 for all edges
        # x >= 0, x integer
        lb = np.ones(num_candidates)  # Actually x >= 0, but let's use 0
        ub = np.ones(num_candidates)
        
        # Use milp
        c = np.ones(num_candidates)  # minimize sum x
        constraints = LinearConstraint(A, lb=np.ones(num_edges), ub=np.ones(num_edges))
        integrality = np.ones(num_candidates)  # all binary
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality,
                       bounds=Bounds(lb=np.zeros(num_candidates), ub=np.ones(num_candidates)),
                       options={"time_limit": max(10, (deadline - time.time()) / 3)})
        except Exception:
            res = None
        
        if res is not None and res.status == 0:
            selected = [i for i in range(num_candidates) if res.x[i] > 0.5]
            cliques = [sorted(list(candidate_list[i])) for i in selected]
        else:
            # Fallback: greedy
            cliques = []
            remaining_edges = set(all_edges)
            
            # Sort candidates by size descending
            cand_sorted = sorted(candidate_list, key=lambda cl: -len(cl))
            
            for cl in cand_sorted:
                if time.time() > deadline:
                    break
                verts = sorted(cl)
                cl_edges = set()
                for a, b in combinations(verts, 2):
                    if (a, b) in remaining_edges:
                        cl_edges.add((a, b))
                if len(cl_edges) >= 1 and len(verts) >= 2:
                    # Check all edges in this clique are still remaining
                    all_in_remaining = True
                    for a, b in combinations(verts, 2):
                        if (a, b) not in remaining_edges:
                            all_in_remaining = False
                            break
                    if all_in_remaining:
                        cliques.append(verts)
                        for a, b in combinations(verts, 2):
                            remaining_edges.discard((a, b))
            
            # Cover remaining edges
            while remaining_edges:
                e = next(iter(remaining_edges))
                u, w = e
                cliques.append(sorted([u, w]))
                remaining_edges.discard(e)
        
        # Verify and fix
        covered = set()
        for cl in cliques:
            for a, b in combinations(cl, 2):
                covered.add((min(a, b), max(a, b)))
        
        all_edges_set = set(all_edges)
        uncovered = all_edges_set - covered
        while uncovered:
            e = next(iter(uncovered))
            u, w = e
            cliques.append(sorted([u, w]))
            uncovered.discard(e)
        
        # Remove duplicate edges check - ensure each edge covered exactly once
        # Re-verify
        edge_count = defaultdict(int)
        for cl in cliques:
            for a, b in combinations(cl, 2):
                edge_count[(min(a, b), max(a, b))] += 1
        
        # If any edge covered more than once, we have a problem
        # For now, trust the ILP or greedy
        
        results.append([[int(x) for x in cl] for cl in cliques])

    return results
