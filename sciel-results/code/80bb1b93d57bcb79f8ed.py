import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    D = 11
    
    # Generate all weight-4 support sets (combinations of 4 positions)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)  # 330
    
    # Precompute support conflict: two supports conflict if they intersect in >= 3 positions
    # (because any sign assignment will have dot >= 3 > 2)
    # If intersection <= 2, there EXISTS a sign assignment with dot <= 2
    support_conflict = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i+1, n_supports):
            sj = set(supports[j])
            if len(si & sj) >= 3:
                support_conflict[i, j] = True
                support_conflict[j, i] = True
    
    # For each pair of non-conflicting supports, determine if sign assignment is always compatible
    # Actually, for intersection <= 2, we can always find signs with dot <= 2
    # For intersection == 2: dot can be -2, 0, or 2. We need dot <= 2, so all work except... wait
    # dot = sum over shared positions of s1[i]*s2[i]. With 2 shared positions, dot = s1a*s2a + s1b*s2b
    # This can be -2, 0, or 2. All are <= 2, so any signs work!
    # For intersection == 1: dot = s1a*s2a which is +/-1, always <= 2
    # For intersection == 0: dot = 0
    # So if supports intersect in <= 2, ANY sign assignment works!
    
    # This means: we just need an independent set in the support conflict graph
    # Then we can assign arbitrary signs (e.g., all +1) to each selected support
    
    # Build conflict list for supports
    support_conflict_list = []
    for i in range(n_supports):
        conflicts = np.where(support_conflict[i])[0]
        conflicts = [j for j in conflicts if j != i]
        support_conflict_list.append(conflicts)
    
    def greedy_support_set(order):
        """Greedy independent set on support graph."""
        selected = []
        available = set(order)
        
        for v in order:
            if v not in available:
                continue
            selected.append(v)
            available.remove(v)
            for c in support_conflict_list[v]:
                if c in available:
                    available.remove(c)
        
        return selected
    
    def local_search_1for1(selected_set, max_iters=10000):
        """Try to improve by adding one support at a time (greedy fill)."""
        selected = set(selected_set)
        available = set(range(n_supports)) - selected
        for c in list(selected):
            for conf in support_conflict_list[c]:
                if conf in available:
                    available.discard(conf)
        
        # Try to add any available support
        for s in list(available):
            # Check if s conflicts with any selected
            conflict_found = False
            for sel in selected:
                if support_conflict[s, sel]:
                    conflict_found = True
                    break
            if not conflict_found:
                selected.add(s)
                available.discard(s)
                for conf in support_conflict_list[s]:
                    if conf in available:
                        available.discard(conf)
        
        return selected
    
    def local_search_2for2(selected_set, max_iters=5000):
        """Try 2-for-2 swaps: remove 2, add 2 different supports."""
        selected = set(selected_set)
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            sel_list = list(selected)
            if len(sel_list) < 2:
                break
            for i in range(len(sel_list)):
                for j in range(i+1, len(sel_list)):
                    # Try removing sel_list[i] and sel_list[j]
                    remaining = selected - {sel_list[i], sel_list[j]}
                    # Find best pair to add
                    candidates_to_add = []
                    for s1 in range(n_supports):
                        if s1 in remaining:
                            continue
                        # Check s1 doesn't conflict with remaining
                        ok1 = True
                        for r in remaining:
                            if support_conflict[s1, r]:
                                ok1 = False
                                break
                        if not ok1:
                            continue
                        for s2 in range(s1+1, n_supports):
                            if s2 in remaining or s2 == s1:
                                continue
                            # Check s2 doesn't conflict with remaining
                            ok2 = True
                            for r in remaining:
                                if support_conflict[s2, r]:
                                    ok2 = False
                                    break
                            if not ok2:
                                continue
                            # Check s1 and s2 don't conflict
                            if support_conflict[s1, s2]:
                                continue
                            candidates_to_add.append((s1, s2))
                    
                    if len(candidates_to_add) >= 2:
                        # Add the first valid pair
                        s1, s2 = candidates_to_add[0]
                        selected.discard(sel_list[i])
                        selected.discard(sel_list[j])
                        selected.add(s1)
                        selected.add(s2)
                        improved = True
                        break
                if improved:
                    break
        return selected
    
    best_supports = set()
    
    # Phase 1: Random restarts with greedy
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        order = list(range(n_supports))
        random.shuffle(order)
        sol = greedy_support_set(order)
        # Local search 1-for-1
        sol = local_search_1for1(sol)
        if len(sol) > len(best_supports):
            best_supports = sol
    
    # Phase 2: Min-degree heuristic
    degrees = np.array([len(cl) for cl in support_conflict_list])
    order = np.argsort(degrees).tolist()
    sol = greedy_support_set(order)
    sol = local_search_1for1(sol)
    if len(sol) > len(best_supports):
        best_supports = sol
    
    # Phase 3: More random restarts with local search
    if time.time() < deadline - 15:
        for restart in range(200):
            if time.time() > deadline - 5:
                break
            order = list(range(n_supports))
            random.shuffle(order)
            sol = greedy_support_set(order)
            sol = local_search_1for1(sol)
            if len(sol) > len(best_supports):
                best_supports = sol
    
    # Phase 4: 2-for-2 polish on best solution
    if time.time() < deadline - 5 and len(best_supports) > 0:
        polished = local_search_2for2(best_supports, max_iters=2000)
        if len(polished) > len(best_supports):
            best_supports = polished
    
    if not best_supports:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
    
    # Convert selected supports to vectors (all +1 signs)
    result = np.zeros((len(best_supports), D), dtype=np.int64)
    for idx, s in enumerate(sorted(best_supports)):
        for pos in supports[s]:
            result[idx, pos] = 1
    
    return result
