"""Bootstrap direct approach: meet every maximal clique (size>=2) with a minimal-ish transversal."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

_DEADLINE = 150.0
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
    
    # Precompute vertex-clique incidence
    m = len(cliques)
    vertex_to_cliques = defaultdict(list)
    for i, c in enumerate(cliques):
        for v in c:
            vertex_to_cliques[v].append(i)
    
    # Greedy set cover: select vertices to hit all cliques
    selected = set()
    uncovered = set(range(m))
    
    while uncovered and time.time() - _START_TIME < _DEADLINE - 20:
        # Find vertex that hits the most uncovered cliques
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
                if count == len(uncovered):
                    break
        if best_v is None or best_count == 0:
            break
        selected.add(best_v)
        for ci in vertex_to_cliques[best_v]:
            uncovered.discard(ci)
    
    # If we still have uncovered cliques, add one vertex from each
    for ci in uncovered:
        v = next(iter(cliques[ci]))
        selected.add(v)
    
    # Local search: try to remove vertices from the selected set
    # A vertex can be removed if every clique it hits is also hit by another selected vertex
    # We iterate: try removing each vertex, check validity, keep removals that work
    improved = True
    while improved and time.time() - _START_TIME < _DEADLINE - 15:
        improved = False
        # For each selected vertex, check if it's redundant
        # A vertex v is redundant if for every clique c containing v, there exists
        # another vertex v' in selected \ {v} that is also in c
        sel_list = list(selected)
        for v in sel_list:
            if v not in selected:
                continue
            # Check if v is needed: is there a clique that only v (among selected) hits?
            v_needed = False
            for ci in vertex_to_cliques[v]:
                c = cliques[ci]
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
    
    # Second pass: try swaps - remove one vertex, add another to see if we can reduce
    # Actually, since we're minimizing size, swaps that keep same size don't help.
    # Try: for each pair (v_out, v_in) where v_out in selected, v_in not in selected,
    # check if (selected - {v_out} + {v_in}) is still a valid hitting set.
    # This won't reduce size, so skip.
    
    # Try to find smaller solution via randomized restarts with different greedy orderings
    if time.time() - _START_TIME < _DEADLINE - 30:
        best_size = len(selected)
        best_sol = set(selected)
        # Try a few random orderings of the greedy
        for trial in range(5):
            if time.time() - _START_TIME > _DEADLINE - 25:
                break
            trial_sel = set()
            trial_uncov = set(range(m))
            # Randomize vertex ordering for ties
            while trial_uncov and time.time() - _START_TIME < _DEADLINE - 25:
                best_v = None
                best_count = 0
                for v in range(n):
                    count = 0
                    for ci in vertex_to_cliques[v]:
                        if ci in trial_uncov:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_v = v
                    elif count == best_count and count > 0:
                        # Tie-breaking: random
                        if best_v is None or random.random() < 0.5:
                            best_v = v
                if best_v is None or best_count == 0:
                    break
                trial_sel.add(best_v)
                for ci in vertex_to_cliques[best_v]:
                    trial_uncov.discard(ci)
            for ci in trial_uncov:
                v = next(iter(cliques[ci]))
                trial_sel.add(v)
            # Local search on trial solution
            trial_improved = True
            while trial_improved and time.time() - _START_TIME < _DEADLINE - 25:
                trial_improved = False
                for v in list(trial_sel):
                    if v not in trial_sel:
                        continue
                    v_needed = False
                    for ci in vertex_to_cliques[v]:
                        c = cliques[ci]
                        other_hit = False
                        for v2 in trial_sel:
                            if v2 != v and v2 in c:
                                other_hit = True
                                break
                        if not other_hit:
                            v_needed = True
                            break
                    if not v_needed:
                        trial_sel.discard(v)
                        trial_improved = True
            if len(trial_sel) < best_size:
                best_size = len(trial_sel)
                best_sol = set(trial_sel)
        selected = best_sol
    
    return sorted(selected)

# EVOLVE-BLOCK-END
