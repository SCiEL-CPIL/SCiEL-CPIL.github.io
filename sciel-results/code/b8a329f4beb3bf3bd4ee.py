"""Combined approach: greedy set cover + local search + randomized restarts for minimal clique transversal."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

_DEADLINE = 145.0
_START_TIME = time.time()

def _remaining():
    return _DEADLINE - time.time()

def construct():
    start = time.time()
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start > _DEADLINE - 5:
            # Fallback: return all touched vertices for remaining instances
            rng = random.Random(seed)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            touched = sorted({x for e in edges for x in e})
            out.append(touched)
            continue
        sol = _solve_instance(n, p, seed)
        out.append(sol)
    return out

def _solve_instance(n, p, seed):
    # Build adjacency
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    
    # Enumerate all maximal cliques via Bron-Kerbosch with pivot
    cliques = []
    def bronk(R, P, X):
        if time.time() - _START_TIME > _DEADLINE - 10:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot
        best_pivot = None
        max_count = -1
        for v in P:
            c = len(adj[v] & (P | X))
            if c > max_count:
                max_count = c
                best_pivot = v
        if best_pivot is not None:
            candidates = P - adj[best_pivot]
        else:
            candidates = set(P)
        for v in list(candidates):
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bronk(set(), set(range(n)), set())
    
    # Filter: only cliques of size >= 2 matter
    cliques = [c for c in cliques if len(c) >= 2]
    
    if not cliques:
        return []
    
    m = len(cliques)
    
    # Precompute vertex-clique incidence
    vertex_to_cliques = defaultdict(list)
    for i, c in enumerate(cliques):
        for v in c:
            vertex_to_cliques[v].append(i)
    
    # Precompute clique membership as sets for fast lookup
    clique_sets = [set(c) for c in cliques]
    
    # Precompute for each vertex, the set of cliques it covers
    vertex_clique_set = [set(vertex_to_cliques[v]) for v in range(n)]
    
    def _local_search(selected, time_limit):
        """Iteratively remove redundant vertices using coverage counts."""
        improved = True
        while improved and time.time() - _START_TIME < time_limit:
            improved = False
            for v in list(selected):
                if v not in selected:
                    continue
                # Check if v is needed: is there a clique where v is the only selected vertex?
                v_needed = False
                for ci in vertex_to_cliques[v]:
                    c = clique_sets[ci]
                    # Check if any other selected vertex is in this clique
                    other_hit = False
                    for v2 in selected:
                        if v2 != v and v2 in c:
                            other_hit = True
                            break
                    if not other_hit:
                        v_needed = True
                        break
                if not v_needed:
                    selected.discard(v)
                    improved = True
        return selected
    
    def _local_search_2exchange(selected, time_limit):
        """Try removing pairs of vertices that are jointly redundant."""
        improved = True
        while improved and time.time() - _START_TIME < time_limit:
            improved = False
            sel_list = sorted(selected)
            for i in range(len(sel_list)):
                if time.time() - _START_TIME > time_limit:
                    return selected
                for j in range(i + 1, len(sel_list)):
                    v1, v2 = sel_list[i], sel_list[j]
                    if v1 not in selected or v2 not in selected:
                        continue
                    # Check if removing both v1 and v2 still covers all cliques
                    test_sel = selected - {v1, v2}
                    valid = True
                    for ci in range(m):
                        c = clique_sets[ci]
                        if not (c & test_sel):
                            valid = False
                            break
                    if valid:
                        selected = test_sel
                        improved = True
                        break
                if improved:
                    break
        return selected
    
    def _greedy_cover(randomize=False, rng=None, weight=1.0):
        """Greedy set cover with optional randomization and weighting."""
        selected = set()
        uncovered = set(range(m))
        
        while uncovered and time.time() - _START_TIME < _DEADLINE - 15:
            best_v = None
            best_score = 0
            for v in range(n):
                if v in selected:
                    continue
                count = 0
                for ci in vertex_to_cliques[v]:
                    if ci in uncovered:
                        count += 1
                if count <= best_score:
                    continue
                score = count * weight
                if randomize and rng is not None:
                    score += rng.random() * 0.5
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v is None:
                break
            selected.add(best_v)
            for ci in vertex_to_cliques[best_v]:
                uncovered.discard(ci)
        
        # Cover any remaining uncovered cliques
        for ci in uncovered:
            v = next(iter(cliques[ci]))
            selected.add(v)
        
        return selected
    
    # First greedy pass
    selected = _greedy_cover(randomize=False)
    selected = _local_search(selected, _DEADLINE - 15)
    selected = _local_search_2exchange(selected, _DEADLINE - 15)
    
    best_size = len(selected)
    best_sol = set(selected)
    
    # Randomized restarts with different greedy orderings
    if time.time() - _START_TIME < _DEADLINE - 25:
        for trial in range(15):
            if time.time() - _START_TIME > _DEADLINE - 20:
                break
            trial_rng = random.Random(seed + trial * 1000)
            trial_sel = _greedy_cover(randomize=True, rng=trial_rng)
            trial_sel = _local_search(trial_sel, _DEADLINE - 20)
            trial_sel = _local_search_2exchange(trial_sel, _DEADLINE - 20)
            if len(trial_sel) < best_size:
                best_size = len(trial_sel)
                best_sol = set(trial_sel)
    
    # Swap-based local search with faster validity check
    if time.time() - _START_TIME < _DEADLINE - 10 and best_size > 1:
        improved = True
        while improved and time.time() - _START_TIME < _DEADLINE - 10:
            improved = False
            sel_list = list(best_sol)
            non_sel = set(range(n)) - best_sol
            for v_out in sel_list:
                if time.time() - _START_TIME > _DEADLINE - 10:
                    break
                for v_in in non_sel:
                    # Try swap: remove v_out, add v_in
                    test_sol = set(best_sol)
                    test_sol.discard(v_out)
                    test_sol.add(v_in)
                    # Check validity using coverage
                    valid = True
                    for ci in range(m):
                        c = clique_sets[ci]
                        if not (c & test_sol):
                            valid = False
                            break
                    if valid:
                        test_sol = _local_search(test_sol, _DEADLINE - 10)
                        test_sol = _local_search_2exchange(test_sol, _DEADLINE - 10)
                        if len(test_sol) < best_size:
                            best_size = len(test_sol)
                            best_sol = test_sol
                            improved = True
                            break
                if improved:
                    break
    
    return sorted(best_sol)

# EVOLVE-BLOCK-END
