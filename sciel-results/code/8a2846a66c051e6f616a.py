import time
import math
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    best_score = 0.0
    best_classes = []

    # Precompute gcds for all pairs in [20,200]
    max_m = 200
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(20, max_m + 1):
        for j in range(20, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)

    def find_best_residue(classes, m):
        """Find a residue for m that minimizes future conflicts.
        Returns (r, conflict_count) or (None, None) if no valid residue."""
        # For each candidate residue r in [0, m), check compatibility
        # Also count how many 'slots' it uses (i.e., how many other moduli it conflicts with)
        best_r = None
        best_conflicts = None
        for r in range(m):
            compatible = True
            conflicts = 0
            for m2, r2 in classes:
                g = gcd_table[m][m2]
                if r % g == r2 % g:
                    compatible = False
                    break
                else:
                    conflicts += 1
            if compatible:
                if best_r is None or conflicts < best_conflicts:
                    best_r = r
                    best_conflicts = conflicts
        if best_r is None:
            return None, None
        return best_r, best_conflicts

    def find_any_residue(classes, m):
        """Quickly find any valid residue for m."""
        for r in range(m):
            ok = True
            for m2, r2 in classes:
                g = gcd_table[m][m2]
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                return r
        return None

    def compute_score(classes):
        return sum(1.0 / m for m, r in classes)

    def greedy_order(order):
        """Run greedy construction with given order of moduli."""
        classes = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_any_residue(classes, m)
            if r is not None:
                classes.append([m, r])
        return classes

    def local_search(classes):
        """Improve a solution via add, swap, and residue re-optimization."""
        classes = [list(c) for c in classes]
        selected = set(m for m, r in classes)
        unselected = [m for m in range(20, 201) if m not in selected]
        score = compute_score(classes)
        improved = True
        max_iters = 200
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break

            # Phase 1: Try to add unselected moduli
            added = True
            while added and time.time() < deadline:
                added = False
                for m in unselected[:]:
                    r = find_any_residue(classes, m)
                    if r is not None:
                        classes.append([m, r])
                        selected.add(m)
                        unselected.remove(m)
                        score += 1.0 / m
                        improved = True
                        added = True
                        break  # Restart add phase with new state

            # Phase 2: Try swaps (remove large, add small)
            if not improved:
                sorted_classes = sorted(classes, key=lambda x: x[0], reverse=True)
                for m_remove in [c[0] for c in sorted_classes]:
                    if time.time() > deadline:
                        break
                    trial = [c for c in classes if c[0] != m_remove]
                    best_gain = 0.0
                    best_add = None
                    for m_add in unselected:
                        if m_add >= m_remove:
                            continue
                        r_add = find_any_residue(trial, m_add)
                        if r_add is not None:
                            gain = 1.0 / m_add - 1.0 / m_remove
                            if gain > best_gain:
                                best_gain = gain
                                best_add = (m_add, r_add)
                    if best_add is not None and best_gain > 0:
                        classes = trial + [list(best_add)]
                        selected.remove(m_remove)
                        selected.add(best_add[0])
                        unselected.remove(best_add[0])
                        unselected.append(m_remove)
                        score += best_gain
                        improved = True
                        break  # Restart with new state

            # Phase 3: Residue re-optimization - try changing residues to free up space
            if not improved:
                for idx in range(len(classes)):
                    if time.time() > deadline:
                        break
                    m, r = classes[idx]
                    others = [c for i, c in enumerate(classes) if i != idx]
                    # Try to find a residue that's compatible and might help others
                    for r_new in range(m):
                        if r_new == r:
                            continue
                        ok = True
                        for m2, r2 in others:
                            g = gcd_table[m][m2]
                            if r_new % g == r2 % g:
                                ok = False
                                break
                        if ok:
                            # Check if this helps: can we now add an unselected modulus?
                            trial = others + [[m, r_new]]
                            for m_add in unselected:
                                r_add = find_any_residue(trial, m_add)
                                if r_add is not None:
                                    classes = trial + [[m_add, r_add]]
                                    selected.add(m_add)
                                    unselected.remove(m_add)
                                    score += 1.0 / m_add
                                    improved = True
                                    break
                    if improved:
                        break

        return classes, score

    # Strategy 1: Greedy ascending (small moduli first - high weight)
    order_asc = list(range(20, 201))
    classes = greedy_order(order_asc)
    if classes:
        score = compute_score(classes)
        if score > best_score:
            best_score = score
            best_classes = [list(c) for c in classes]

    # Strategy 2: Greedy descending (large moduli first - may leave room for small ones)
    order_desc = list(range(200, 19, -1))
    classes = greedy_order(order_desc)
    if classes:
        score = compute_score(classes)
        if score > best_score:
            best_score = score
            best_classes = [list(c) for c in classes]

    # Strategy 3: Randomized greedy with many restarts
    num_restarts = 3000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        moduli = list(range(20, 201))
        random.shuffle(moduli)
        classes = greedy_order(moduli)
        if classes:
            score = compute_score(classes)
            if score > best_score:
                best_score = score
                best_classes = [list(c) for c in classes]

    # Strategy 4: Local search on the best solution
    if best_classes:
        improved_classes, improved_score = local_search(best_classes)
        if improved_score > best_score:
            best_score = improved_score
            best_classes = improved_classes

    # Strategy 5: More local search iterations on top results
    # Run a few more local search passes
    for _ in range(3):
        if time.time() > deadline:
            break
        improved_classes, improved_score = local_search(best_classes)
        if improved_score > best_score:
            best_score = improved_score
            best_classes = improved_classes

    return best_classes
