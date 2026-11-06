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
    # Allocate time budget per instance: ~45s each with safety margin
    for idx, (n, p, seed) in enumerate(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))):
        remaining_total = _DEADLINE - (time.time() - start)
        if remaining_total < 5:
            # Fallback: return all touched vertices
            rng = random.Random(seed)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            touched = sorted({x for e in edges for x in e})
            out.append(touched)
            continue
        # Give each remaining instance at most 1/3 of remaining time minus buffer
        instances_left = 3 - idx
        time_budget = max(3.0, remaining_total / instances_left - 2.0)
        sol = _solve_instance(n, p, seed, time_budget)
        out.append(sol)
    return out

def _solve_instance(n, p, seed, time_budget):
    start_t = time.time()
    def _elapsed():
        return time.time() - start_t
    def _time_left():
        return time_budget - _elapsed()
    
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    
    # Enumerate maximal cliques with Bron-Kerbosch using sets
    cliques = []
    def bronk(R, P, X):
        if _elapsed() > time_budget - 3:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Pivot rule: pick vertex in P∪X with max |adj ∩ (P∪X)|
        PX = P | X
        best_pivot = None
        max_count = -1
        for v in PX:
            c = len(adj[v] & PX)
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
    cliques = [c for c in cliques if len(c) >= 2]
    if not cliques:
        return []
    
    m = len(cliques)
    
    # Build incidence structures
    vertex_to_cliques = [list() for _ in range(n)]
    clique_vertices = [list() for _ in range(m)]
    for i, c in enumerate(cliques):
        for v in c:
            vertex_to_cliques[v].append(i)
            clique_vertices[i].append(v)
    
    # Precompute: for each clique, which vertices are in it (as set for fast intersection)
    clique_sets = [set(c) for c in cliques]
    
    # For fast validity check: for each vertex, bitmask of cliques it covers
    # Since m can be large, use list of booleans or a set
    vertex_clique_set = [set(vertex_to_cliques[v]) for v in range(n)]
    
    def _is_valid(selected):
        """Check if selected covers all cliques using union of covered cliques."""
        covered = set()
        for v in selected:
            covered.update(vertex_clique_set[v])
        return len(covered) == m
    
    def _remove_redundant(selected, limit_frac=0.95):
        """Remove vertices that are not needed (no clique is solely covered by them)."""
        improved = True
        while improved and _elapsed() < time_budget * limit_frac:
            improved = False
            # Compute coverage count: for each clique, how many selected vertices cover it
            clique_cover_count = [0] * m
            for v in selected:
                for ci in vertex_to_cliques[v]:
                    clique_cover_count[ci] += 1
            
            # Find vertices that are the unique cover for some clique
            for v in list(selected):
                if v not in selected:
                    continue
                needed = False
                for ci in vertex_to_cliques[v]:
                    if clique_cover_count[ci] == 1:
                        needed = True
                        break
                if not needed:
                    selected.discard(v)
                    # Update coverage counts
                    for ci in vertex_to_cliques[v]:
                        clique_cover_count[ci] -= 1
                    improved = True
        return selected
    
    def _try_remove_pair(selected, limit_frac=0.95):
        """Try to find a pair of vertices that can both be removed."""
        improved = True
        while improved and _elapsed() < time_budget * limit_frac:
            improved = False
            sel_list = list(selected)
            if len(sel_list) < 2:
                break
            # Compute coverage counts
            clique_cover_count = [0] * m
            for v in selected:
                for ci in vertex_to_cliques[v]:
                    clique_cover_count[ci] += 1
            
            # Find cliques with coverage count exactly 2
            weak_cliques = [ci for ci in range(m) if clique_cover_count[ci] == 2]
            if not weak_cliques:
                break
            
            for ci in weak_cliques:
                if _elapsed() > time_budget * limit_frac:
                    break
                # Get the two vertices covering this clique
                covering = [v for v in selected if ci in vertex_to_cliques[v]]
                if len(covering) != 2:
                    continue
                v1, v2 = covering[0], covering[1]
                # Check if removing both still covers everything
                # A clique is unprotected if its only selected vertices are v1 or v2
                valid = True
                for ci2 in range(m):
                    cnt = clique_cover_count[ci2]
                    if cnt == 0:
                        valid = False
                        break
                    if cnt == 1:
                        # Check if the sole cover is v1 or v2
                        sole = None
                        for v in selected:
                            if v != v1 and v != v2 and ci2 in vertex_to_cliques[v]:
                                sole = v
                                break
                        if sole is None:
                            valid = False
                            break
                if valid:
                    selected.discard(v1)
                    selected.discard(v2)
                    improved = True
                    break
        return selected
    
    def _greedy_cover(randomize=False, rng=None):
        """Greedy set cover."""
        selected = set()
        uncovered = set(range(m))
        # Track coverage counts for efficiency
        while uncovered and _elapsed() < time_budget - 5:
            best_v = None
            best_score = -1
            for v in range(n):
                if v in selected:
                    continue
                count = sum(1 for ci in vertex_to_cliques[v] if ci in uncovered)
                if count <= 0:
                    continue
                score = float(count)
                if randomize and rng is not None:
                    score += rng.random() * 0.3
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v is None:
                break
            selected.add(best_v)
            for ci in vertex_to_cliques[best_v]:
                uncovered.discard(ci)
        
        for ci in uncovered:
            v = next(iter(cliques[ci]))
            selected.add(v)
        return selected
    
    def _local_search(selected, limit_frac=0.95):
        """Swap-based local search: try replacing one vertex with another (efficient)."""
        improved = True
        while improved and _elapsed() < time_budget * limit_frac:
            improved = False
            sel_list = list(selected)
            non_sel = [v for v in range(n) if v not in selected]
            if not non_sel or not sel_list:
                break
            # Precompute coverage counts
            clique_cover_count = [0] * m
            for v in selected:
                for ci in vertex_to_cliques[v]:
                    clique_cover_count[ci] += 1
            # For each clique, track which selected vertices cover it
            clique_selected_coverers = [set() for _ in range(m)]
            for v in selected:
                for ci in vertex_to_cliques[v]:
                    clique_selected_coverers[ci].add(v)
            
            for v_out in sel_list:
                if _elapsed() > time_budget * limit_frac:
                    break
                # Compute what would be uncovered if we remove v_out
                # A clique becomes "at risk" if v_out is its only selected coverer
                at_risk = []
                for ci in vertex_to_cliques[v_out]:
                    if len(clique_selected_coverers[ci]) == 1:
                        at_risk.append(ci)
                if not at_risk:
                    continue
                for v_in in non_sel:
                    # Check if v_in covers all at-risk cliques
                    covers_all = all(ci in vertex_to_cliques[v_in] for ci in at_risk)
                    if covers_all:
                        # Try the swap
                        test_sol = set(selected)
                        test_sol.discard(v_out)
                        test_sol.add(v_in)
                        # Check full validity
                        if _is_valid(test_sol):
                            test_sol = _remove_redundant(test_sol, limit_frac)
                            test_sol = _try_remove_pair(test_sol, limit_frac)
                            if len(test_sol) < len(selected):
                                selected = test_sol
                                improved = True
                                break
                if improved:
                    break
        return selected
    
    def _two_exchange(selected, limit_frac=0.95):
        """Try replacing two selected vertices with one non-selected vertex."""
        improved = True
        while improved and _elapsed() < time_budget * limit_frac:
            improved = False
            sel_list = list(selected)
            non_sel = [v for v in range(n) if v not in selected]
            if len(sel_list) < 2 or not non_sel:
                break
            # For each non-selected vertex, check if it can replace two selected
            for v_in in non_sel:
                if _elapsed() > time_budget * limit_frac:
                    break
                # Find cliques that v_in covers
                in_cliques = set(vertex_to_cliques[v_in])
                if not in_cliques:
                    continue
                # Find selected vertices that are only needed for cliques v_in covers
                # i.e., selected vertices v_out where all cliques they uniquely cover are in in_cliques
                candidates = []
                for v_out in sel_list:
                    if v_out in selected:
                        candidates.append(v_out)
                # Try pairs of candidates
                for i in range(len(candidates)):
                    if _elapsed() > time_budget * limit_frac:
                        break
                    v_out1 = candidates[i]
                    if v_out1 not in selected:
                        continue
                    for j in range(i+1, len(candidates)):
                        v_out2 = candidates[j]
                        if v_out2 not in selected:
                            continue
                        test_sol = set(selected)
                        test_sol.discard(v_out1)
                        test_sol.discard(v_out2)
                        test_sol.add(v_in)
                        if len(test_sol) < len(selected):
                            if _is_valid(test_sol):
                                test_sol = _remove_redundant(test_sol, limit_frac)
                                selected = test_sol
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return selected
    
    # Initial solution via greedy
    selected = _greedy_cover(randomize=False)
    selected = _remove_redundant(selected)
    selected = _try_remove_pair(selected)
    
    best_size = len(selected)
    best_sol = set(selected)
    
    # Randomized restarts
    if _elapsed() < time_budget * 0.4:
        max_trials = 30
        for trial in range(max_trials):
            if _elapsed() > time_budget * 0.55:
                break
            trial_rng = random.Random(seed + trial * 1000 + 42)
            trial_sel = _greedy_cover(randomize=True, rng=trial_rng)
            trial_sel = _remove_redundant(trial_sel, 0.55)
            trial_sel = _try_remove_pair(trial_sel, 0.55)
            if len(trial_sel) < best_size:
                best_size = len(trial_sel)
                best_sol = set(trial_sel)
    
    # Swap-based local search
    if _elapsed() < time_budget * 0.70 and best_size > 1:
        best_sol = _local_search(best_sol, 0.75)
        best_size = len(best_sol)
    
    # Two-exchange: replace two selected with one non-selected
    if _elapsed() < time_budget * 0.85 and best_size > 2:
        best_sol = _two_exchange(best_sol, 0.88)
        best_size = len(best_sol)
    
    # Final polish: try more aggressive pair removals
    if _elapsed() < time_budget * 0.95:
        best_sol = _remove_redundant(best_sol, 0.98)
        best_sol = _try_remove_pair(best_sol, 0.98)
        if _elapsed() < time_budget * 0.97:
            best_sol = _local_search(best_sol, 0.98)
    
    return sorted(best_sol)

# EVOLVE-BLOCK-END
