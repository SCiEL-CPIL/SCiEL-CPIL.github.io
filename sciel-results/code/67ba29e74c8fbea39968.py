import time
import math
import random
import heapq
from fractions import Fraction

def construct():
    deadline = time.time() + 140  # safety margin under 150s

    # Even moduli in [20, 200] -- structured gcds enable better packing
    moduli = list(range(20, 201, 2))  # 100 even moduli
    n = len(moduli)
    idx = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table (symmetric)
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def try_greedy(order):
        """Build a solution by adding moduli in the given order, first valid residue."""
        classes = []
        present = set()
        for m in order:
            mi = idx.get(m)
            if mi is None or m in present:
                continue
            found = False
            for r in range(m):
                valid = True
                for m2, r2 in classes:
                    g = gcd_table[mi][idx[m2]]
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    classes.append([m, r])
                    present.add(m)
                    found = True
                    break
            if not found:
                pass
        return classes

    def score_of(classes):
        return sum(1.0 / m for m, _ in classes)

    def is_valid(classes):
        nc = len(classes)
        for i in range(nc):
            m1, r1 = classes[i]
            for j in range(i + 1, nc):
                m2, r2 = classes[j]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    def local_search(classes, max_iters=20000):
        """Improve via: add missing moduli, reassign single conflicting residue, random kick."""
        classes = [c[:] for c in classes]
        score = score_of(classes)
        present = set(m for m, _ in classes)
        missing = [m for m in moduli if m not in present]
        missing.sort()

        rng = random.Random(123)

        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False

            # Move A: direct add of smallest missing modulus
            added = False
            for m in missing:
                mi = idx[m]
                for r in range(m):
                    valid = True
                    for m2, r2 in classes:
                        g = gcd_table[mi][idx[m2]]
                        if r % g == r2 % g:
                            valid = False
                            break
                    if valid:
                        classes.append([m, r])
                        present.add(m)
                        missing.remove(m)
                        score += 1.0 / m
                        added = True
                        break
                if added:
                    break

            if added:
                continue

            # Move B: Reassign one existing class to free a slot for a missing modulus
            if not missing:
                break

            m_target = missing[0]
            mi_target = idx[m_target]

            best_r_target = None
            min_conf = float('inf')

            for r in range(m_target):
                conf = 0
                for m2, r2 in classes:
                    g = gcd_table[mi_target][idx[m2]]
                    if r % g == r2 % g:
                        conf += 1
                if conf < min_conf:
                    min_conf = conf
                    best_r_target = r
                    if conf == 0:
                        break

            if min_conf <= 1:
                conf_js = []
                for j, (m2, r2) in enumerate(classes):
                    g = gcd_table[mi_target][idx[m2]]
                    if best_r_target % g == r2 % g:
                        conf_js.append(j)

                for j in conf_js:
                    mj, rj = classes[j]
                    mj_i = idx[mj]
                    g_jt = gcd_table[mj_i][mi_target]
                    target_bad = best_r_target % g_jt
                    found_move = False
                    for new_r in range(mj):
                        if new_r % g_jt == target_bad:
                            continue
                        ok = True
                        for k, (m2, r2) in enumerate(classes):
                            if k == j:
                                continue
                            g = gcd_table[mj_i][idx[m2]]
                            if new_r % g == r2 % g:
                                ok = False
                                break
                        if ok:
                            classes[j][1] = new_r
                            classes.append([m_target, best_r_target])
                            present.add(m_target)
                            missing.remove(m_target)
                            score += 1.0 / m_target
                            found_move = True
                            break
                    if found_move:
                        improved = True
                        break

            if improved:
                continue

            # Move C: Random Kick
            if classes:
                j = rng.randint(0, len(classes) - 1)
                mj, rj = classes[j]
                mj_i = idx[mj]
                for _ in range(10):
                    new_r = rng.randint(0, mj - 1)
                    if new_r == rj:
                        continue
                    ok = True
                    for k, (m2, r2) in enumerate(classes):
                        if k == j:
                            continue
                        g = gcd_table[mj_i][idx[m2]]
                        if new_r % g == r2 % g:
                            ok = False
                            break
                    if ok:
                        classes[j][1] = new_r
                        break

        return classes, score

    # --- Multi-start greedy ---
    rng = random.Random(42)
    best_classes = try_greedy(sorted(moduli))
    best_score = score_of(best_classes)

    for attempt in range(500):
        if time.time() > deadline - 25:
            break
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        weighted = sorted(shuffled, key=lambda m: (m * 0.3 + rng.random() * 40))
        cand = try_greedy(weighted)
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_classes = cand

    # Local search on best
    best_classes, best_score = local_search(best_classes)
    if not is_valid(best_classes):
        best_classes = try_greedy(sorted(moduli))
        best_score = score_of(best_classes)

    # More randomized restarts + local search
    for attempt in range(400):
        if time.time() > deadline - 15:
            break
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        weighted = sorted(shuffled, key=lambda m: (m * 0.5 + rng.random() * 50))
        cand = try_greedy(weighted)
        s = score_of(cand)
        if s > best_score * 0.98:
            cand, s2 = local_search(cand)
            if s2 > best_score:
                best_score = s2
                best_classes = cand

    # Final pass: try to add any remaining moduli
    present = set(m for m, _ in best_classes)
    missing = [m for m in moduli if m not in present]
    missing.sort()
    for m in missing:
        if time.time() > deadline - 5:
            break
        mi = idx[m]
        for r in range(m):
            valid = True
            for m2, r2 in best_classes:
                g = gcd_table[mi][idx[m2]]
                if r % g == r2 % g:
                    valid = False
                    break
            if valid:
                best_classes.append([m, r])
                best_score += 1.0 / m
                break

    # --- Branch-and-bound on conflict graph for remaining moduli ---
    # Formulate as MWIS: nodes are (m, r) pairs, edges are modular clashes.
    # We want to maximize sum(1/m) over an independent set.
    # Use fractional knapsack upper bound for pruning.
    
    present_set = set(m for m, _ in best_classes)
    remaining = [m for m in moduli if m not in present_set]
    remaining.sort()  # smallest first (highest weight)
    
    if remaining and time.time() < deadline - 10:
        # Build conflict structure for remaining moduli against existing + each other
        # For B&B, we process moduli in order of decreasing 1/m (i.e., increasing m is wrong)
        # Actually, for knapsack bound, we sort by weight/cost ratio.
        # Here "cost" is 1 (each modulus takes 1 slot in the independent set sense? No.)
        # The constraint is not a simple knapsack. It's a graph independent set.
        # The "fractional knapsack" bound for MWIS is tricky.
        # A common bound: sum of weights of all remaining nodes that are not adjacent to any selected node.
        # But that's loose.
        
        # Let's try a simpler B&B:
        # State: (current_selection, next_index_in_remaining)
        # But the conflict depends on residues, not just moduli.
        # So we need to track residues.
        
        # Alternative: For each remaining modulus, precompute the set of valid residues
        # given the current best_classes. Then do B&B on which moduli to include.
        
        # Precompute valid residues for each remaining modulus against current classes
        valid_residues = {}
        for m in remaining:
            mi = idx[m]
            valid_rs = []
            for r in range(m):
                ok = True
                for m2, r2 in best_classes:
                    g = gcd_table[mi][idx[m2]]
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    valid_rs.append(r)
            valid_residues[m] = valid_rs
        
        # Remove moduli with no valid residues
        remaining = [m for m in remaining if valid_residues[m]]
        
        if remaining:
            # B&B: try to add as many remaining moduli as possible
            # Order by 1/m descending (smallest m first)
            remaining.sort(key=lambda m: -1.0/m)
            
            # For each modulus, we need to pick a residue.
            # The conflict between two remaining moduli m1, m2 depends on chosen residues.
            # This is complex. Let's do a greedy B&B:
            # At each step, try adding the next modulus with each valid residue.
            # Bound: current_score + sum(1/m for m in remaining_from_here)
            
            def bb_search(classes, rem_list, current_score):
                nonlocal best_score, best_classes
                
                if not rem_list:
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in classes]
                    return
                
                if time.time() > deadline - 2:
                    return
                
                # Upper bound: current_score + sum of 1/m for all remaining
                upper = current_score + sum(1.0/m for m in rem_list)
                if upper <= best_score:
                    return
                
                m = rem_list[0]
                rest = rem_list[1:]
                mi = idx[m]
                
                # Try each valid residue for m
                for r in valid_residues[m]:
                    # Check if this residue conflicts with any in classes
                    # (It shouldn't, since we precomputed against best_classes,
                    #  but classes may have changed if we added other moduli)
                    ok = True
                    for m2, r2 in classes:
                        g = gcd_table[mi][idx[m2]]
                        if r % g == r2 % g:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    new_classes = classes + [[m, r]]
                    bb_search(new_classes, rest, current_score + 1.0/m)
                
                # Also try skipping m
                bb_search(classes, rest, current_score)
            
            # Start B&B
            bb_search([c[:] for c in best_classes], remaining, best_score)
    
    return best_classes
