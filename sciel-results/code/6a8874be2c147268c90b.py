import time
import math
import random

def construct():
    deadline = time.time() + 110  # safety margin under 150s

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
            # Heuristic: try residues in a pseudo-random order to avoid systematic bias?
            # Or just 0 to m-1.
            # Let's try residues that are coprime to m? No, r < m.
            # Let's try to spread residues.
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
                # Try harder? No, greedy fails.
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
        missing.sort()  # smallest first (highest 1/m)
        
        rng = random.Random(123)

        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False

            # Move A: direct add of smallest missing modulus
            added = False
            for m in missing:
                mi = idx[m]
                # Try to find a valid residue
                # Optimization: check residues in random order? 
                # No, deterministic is better for reproducibility, but random might find one faster if exists?
                # Let's stick to 0..m-1 for speed, it's O(m * N).
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
            # Pick the smallest missing modulus
            if not missing:
                break
            
            m_target = missing[0]
            mi_target = idx[m_target]
            
            # Find the residue r_target with minimum conflicts
            best_r_target = None
            min_conf = float('inf')
            
            # Sample some residues if m is large to save time? No, m <= 200.
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
                # Identify conflicting classes
                conf_js = []
                for j, (m2, r2) in enumerate(classes):
                    g = gcd_table[mi_target][idx[m2]]
                    if best_r_target % g == r2 % g:
                        conf_js.append(j)
                
                # Try to move one of the conflicting classes
                for j in conf_js:
                    mj, rj = classes[j]
                    mj_i = idx[mj]
                    g_jt = gcd_table[mj_i][mi_target]
                    # We need to move rj to new_r such that new_r != best_r_target % g_jt
                    # AND new_r is valid against all other classes
                    target_bad = best_r_target % g_jt
                    found_move = False
                    # Try all new_r
                    for new_r in range(mj):
                        if new_r % g_jt == target_bad:
                            continue
                        # Check validity against others
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
            # Pick a random class, change its residue to a random valid one (if any)
            # This might open up space for a missing modulus.
            if classes:
                j = rng.randint(0, len(classes) - 1)
                mj, rj = classes[j]
                mj_i = idx[mj]
                # Find a new valid residue for class j
                # Try random residues
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
                        # Try to add a missing modulus again? 
                        # The main loop will try Move A next iteration.
                        break

        return classes, score

    # --- Multi-start greedy ---
    rng = random.Random(42)
    best_classes = try_greedy(sorted(moduli))
    best_score = score_of(best_classes)

    # Randomized greedy restarts
    for attempt in range(500):
        if time.time() > deadline - 15:
            break
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        # Weight towards smaller moduli
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
        if time.time() > deadline - 8:
            break
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        weighted = sorted(shuffled, key=lambda m: (m * 0.5 + rng.random() * 50))
        cand = try_greedy(weighted)
        s = score_of(cand)
        if s > best_score * 0.98:  # only local search promising candidates
            cand, s2 = local_search(cand)
            if s2 > best_score:
                best_score = s2
                best_classes = cand

    # Final pass: try to add any remaining moduli
    present = set(m for m, _ in best_classes)
    missing = [m for m in moduli if m not in present]
    missing.sort()
    for m in missing:
        if time.time() > deadline - 2:
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

    return best_classes
