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
    
    # Also precompute clique membership as sets for fast lookup
    clique_sets = [set(c) for c in cliques]
    
    def _local_search(selected, time_limit):
        """Iteratively remove redundant vertices."""
        improved = True
        while improved and time.time() - _START_TIME < time_limit:
            improved = False
            for v in list(selected):
                if v not in selected:
                    continue
                # Check if v is needed: is there a clique that only v (among selected) hits?
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
    
    def _greedy_cover(randomize=False, rng=None):
        """Greedy set cover with optional randomization for ties."""
        selected = set()
        uncovered = set(range(m))
        
        while uncovered and time.time() - _START_TIME < _DEADLINE - 15:
            best_v = None
            best_count = 0
            for v in range(n):
                count = 0
                for ci in vertex_to_cliques[v]:
                    if ci in uncovered:
                        count += 1
                        if count <= best_count:
                            continue
                if count > best_count:
                    best_count = count
                    best_v = v
                elif count == best_count and count > 0 and randomize and rng is not None:
                    if best_v is None or rng.random() < 0.5:
                        best_v = v
                if count == len(uncovered):
                    break
            if best_v is None or best_count == 0:
                break
            selected.add(best_v)
            for ci in vertex_to_cliques[best_v]:
                uncovered.discard(ci)
        
        # Cover any remaining uncovered cliques
        for ci in uncovered:
            v = next(iter(cliques[ci]))
            selected.add(v)
        
        return selected, uncovered
    
    # First greedy pass
    selected, _ = _greedy_cover(randomize=False)
    
    # Local search to remove redundant vertices
    selected = _local_search(selected, _DEADLINE - 15)
    
    best_size = len(selected)
    best_sol = set(selected)
    
    # Randomized restarts with different greedy orderings
    if time.time() - _START_TIME < _DEADLINE - 25:
        for trial in range(10):
            if time.time() - _START_TIME > _DEADLINE - 20:
                break
            trial_rng = random.Random(seed + trial * 1000)
            trial_sel, trial_uncov = _greedy_cover(randomize=True, rng=trial_rng)
            trial_sel = _local_search(trial_sel, _DEADLINE - 20)
            if len(trial_sel) < best_size:
                best_size = len(trial_sel)
                best_sol = set(trial_sel)
    
    # Additional local search: try swapping vertices to find smaller solutions
    # For each pair (v_out in selected, v_in not in selected), check if swap maintains validity
    # This is expensive, so only do it if we have time
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
                    # Try swap
                    test_sol = set(best_sol)
                    test_sol.discard(v_out)
                    test_sol.add(v_in)
                    # Check validity: every clique must be hit
                    valid = True
                    for ci in range(m):
                        c = clique_sets[ci]
                        if not (c & test_sol):
                            valid = False
                            break
                    if valid:
                        # Check if this actually reduces size (it doesn't, it's a swap)
                        # But it might enable further removals
                        test_sol = _local_search(test_sol, _DEADLINE - 10)
                        if len(test_sol) < best_size:
                            best_size = len(test_sol)
                            best_sol = test_sol
                            improved = True
                            break
                if improved:
                    break
    
    return sorted(best_sol)

# EVOLVE-BLOCK-END
