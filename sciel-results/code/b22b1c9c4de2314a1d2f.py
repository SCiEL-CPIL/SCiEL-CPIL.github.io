import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def enumerate_maximal_cliques(adj, n, deadline):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        max_cliques = []
        
        def bk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if R:
                    max_cliques.append(frozenset(R))
                return
            if not P:
                return
            
            # Choose pivot: vertex in P ∪ X with max degree in P
            pivot = None
            max_deg = -1
            for v in list(P) + list(X):
                deg = len(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
            
            # Branch on P \ N(pivot)
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)
                if time.time() > deadline:
                    return
        
        P = set(range(n))
        X = set()
        bk(set(), P, X)
        
        return max_cliques

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency sets
        adj = {u: set() for u in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Enumerate all maximal cliques
        max_cliques = enumerate_maximal_cliques(adj, n, deadline - 20)
        
        if not max_cliques:
            # Fallback: use all edges as individual cliques
            return [[u, v] for u, v in edges]
        
        # Create edge index
        edge_list = sorted(edges)
        edge_index = {e: i for i, e in enumerate(edge_list)}
        m = len(edge_list)
        k = len(max_cliques)
        
        # Build incidence matrix: A[edge_idx, clique_idx] = 1 if edge is in clique
        A = np.zeros((m, k), dtype=np.int8)
        for ci, clique in enumerate(max_cliques):
            verts = list(clique)
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    e = (min(verts[i], verts[j]), max(verts[i], verts[j]))
                    if e in edge_index:
                        A[edge_index[e], ci] = 1
        
        # Solve ILP: minimize sum(x_i) subject to A @ x >= 1, x binary
        c = np.ones(k)
        # Constraint: A @ x >= 1 (each edge covered at least once)
        # Convert to: -A @ x <= -1
        constraint = LinearConstraint(-A, -1, 0)
        bounds = Bounds(0, 1)
        
        try:
            result = milp(c, constraints=constraint, integrality=np.ones(k), bounds=bounds)
            
            if result.success:
                selected = [ci for ci in range(k) if result.x[ci] > 0.5]
                partition = [sorted(list(max_cliques[ci])) for ci in selected]
                
                # Verify coverage
                covered = set()
                for clique in partition:
                    verts = clique
                    for i in range(len(verts)):
                        for j in range(i + 1, len(verts)):
                            e = (min(verts[i], verts[j]), max(verts[i], verts[j]))
                            covered.add(e)
                
                if covered == set(edges):
                    return partition
                else:
                    # Some edges not covered, add them as individual cliques
                    uncovered = set(edges) - covered
                    for e in uncovered:
                        partition.append(list(e))
                    return partition
            else:
                raise RuntimeError(f"MILP failed: {result.message}")
        except Exception:
            pass
        
        # Fallback: greedy set cover
        remaining_edges = set(edges)
        partition = []
        
        while remaining_edges and time.time() < deadline:
            # Find clique covering most remaining edges
            best_clique = None
            best_count = 0
            for clique in max_cliques:
                verts = list(clique)
                count = 0
                for i in range(len(verts)):
                    for j in range(i + 1, len(verts)):
                        e = (min(verts[i], verts[j]), max(verts[i], verts[j]))
                        if e in remaining_edges:
                            count += 1
                if count > best_count:
                    best_count = count
                    best_clique = clique
                if time.time() > deadline:
                    break
            
            if best_clique is None or best_count == 0:
                break
            
            partition.append(sorted(list(best_clique)))
            
            # Remove covered edges
            verts = list(best_clique)
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    e = (min(verts[i], verts[j]), max(verts[i], verts[j]))
                    remaining_edges.discard(e)
        
        # Fallback for remaining edges
        if remaining_edges:
            for u, v in remaining_edges:
                partition.append([u, v])
        
        return partition

    result1 = solve_instance(32, 0.6, 601, 50)
    result2 = solve_instance(64, 0.55, 602, 60)
    
    return [result1, result2]
