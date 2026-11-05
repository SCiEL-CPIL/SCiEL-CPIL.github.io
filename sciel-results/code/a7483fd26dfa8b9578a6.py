"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch(adj, n):
    """Find all maximal cliques of size >= 2."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Pivot selection: choose vertex in P ∪ X with max degree in P
        pivot_candidates = P | X
        if pivot_candidates:
            pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
            candidates = P - adj[pivot]
        else:
            candidates = P
        
        for v in list(candidates):
            bk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _solve_hitting_set(cliques, n, time_budget):
    """Find minimum hitting set for collection of cliques."""
    if not cliques:
        return set()
    
    start_time = time.time()
    
    # Build vertex -> cliques mapping
    vertex_to_cliques = defaultdict(set)
    for i, c in enumerate(cliques):
        for v in c:
            vertex_to_cliques[v].add(i)
    
    # Greedy: repeatedly pick vertex covering most uncovered cliques
    uncovered = set(range(len(cliques)))
    solution = set()
    
    while uncovered:
        if time.time() - start_time > time_budget * 0.3:
            break
        best_v = None
        best_count = 0
        for v in range(n):
            count = len(vertex_to_cliques[v] & uncovered)
            if count > best_count:
                best_count = count
                best_v = v
        if best_v is None or best_count == 0:
            break
        solution.add(best_v)
        uncovered -= vertex_to_cliques[best_v]
    
    if not uncovered:
        # Verify and return
        pass
    
    # Now do local search to improve
    # Build coverage: for each vertex in solution, which cliques it uniquely covers
    def get_coverage(sol):
        """Return dict: clique_idx -> set of solution vertices covering it"""
        cov = defaultdict(set)
        for v in sol:
            for ci in vertex_to_cliques[v]:
                cov[ci].add(v)
        return cov
    
    def is_valid(sol):
        for ci in range(len(cliques)):
            if not any(v in sol for v in cliques[ci]):
                return False
        return True
    
    def try_remove(sol, v):
        """Try removing vertex v from solution."""
        # Check if all cliques containing v are still covered by other solution vertices
        for ci in vertex_to_cliques[v]:
            clique = cliques[ci]
            if not any(u in sol for u in clique if u != v):
                return None  # Cannot remove
        new_sol = sol - {v}
        return new_sol
    
    def try_2_for_1(sol, v1, v2, candidate):
        """Try replacing v1 and v2 with candidate."""
        if candidate in sol:
            return None
        # Check: all cliques that v1 or v2 cover must be covered by (sol - {v1,v2}) ∪ {candidate}
        new_sol = (sol - {v1, v2}) | {candidate}
        for ci in (vertex_to_cliques[v1] | vertex_to_cliques[v2]):
            clique = cliques[ci]
            if not any(u in new_sol for u in clique):
                return None
        return new_sol
    
    # Local search phase
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        
        # Try removals
        for v in list(solution):
            if time.time() - start_time > time_budget:
                break
            new_sol = try_remove(solution, v)
            if new_sol is not None:
                solution = new_sol
                improved = True
        
        # Try 2-for-1
        sol_list = list(solution)
        for i in range(len(sol_list)):
            for j in range(i + 1, len(sol_list)):
                if time.time() - start_time > time_budget:
                    break
                v1, v2 = sol_list[i], sol_list[j]
                # Find candidates: vertices that cover all cliques uniquely covered by v1 or v2
                # A vertex w can replace {v1,v2} if for every clique containing v1 or v2,
                # w is in that clique OR some other solution vertex is in that clique
                # Simpler: just try all vertices not in solution
                for w in range(n):
                    if w in solution:
                        continue
                    new_sol = try_2_for_1(solution, v1, v2, w)
                    if new_sol is not None:
                        solution = new_sol
                        improved = True
                        break  # restart inner loops
                if improved:
                    break
            if improved:
                break
    
    return solution

def construct():
    total_start = time.time()
    deadline = total_start + 110  # Safety margin before 120s limit
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    # Allocate time proportional to n^2
    weights = [n * n for n, p, s in instances]
    total_weight = sum(weights)
    time_budget = 100  # seconds total for solving
    
    for idx, (n, p, seed) in enumerate(instances):
        if time.time() > deadline:
            # Emergency: return all vertices as fallback
            adj = _generate_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            results.append(sorted(touched))
            continue
        
        # Generate graph
        adj = _generate_graph(n, p, seed)
        
        # Find maximal cliques
        cliques = _bron_kerbosch(adj, n)
        
        # Solve hitting set
        t_budget = time_budget * weights[idx] / total_weight
        solution = _solve_hitting_set(cliques, n, t_budget)
        
        # Verify
        if not solution:
            # Fallback
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            solution = touched
        
        results.append(sorted(solution))
    
    return results
# EVOLVE-BLOCK-END
