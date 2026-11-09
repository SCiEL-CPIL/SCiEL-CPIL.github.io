# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Moduli 20..200
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def compute_valid_mask(m, existing_classes):
        """Bitmask of valid residues for modulus m given existing classes.
        Returns (valid_mask, count) where valid_mask has bits set for valid residues."""
        m_val = m
        all_mask = (1 << m_val) - 1
        valid = all_mask

        for (m2, r2) in existing_classes:
            i = mod_idx[m]
            j = mod_idx[m2]
            g = gcd_table[i][j]
            if g == 1:
                # gcd 1 means no constraint (any residue is fine)
                continue
            # Forbidden: r ≡ r2 (mod g)
            r0 = r2 % g
            forbidden = 0
            for r in range(r0, m_val, g):
                forbidden |= (1 << r)
            valid &= ~forbidden

        count = bin(valid).count('1')
        return valid, count

    def pick_best_residue(valid_mask, m, existing_classes):
        """Pick a residue from valid_mask that maximizes flexibility for future additions."""
        if valid_mask == 0:
            return None

        # Get all valid residues
        valid_residues = []
        for r in range(m):
            if (valid_mask >> r) & 1:
                valid_residues.append(r)

        if len(valid_residues) == 1:
            return valid_residues[0]

        used_moduli = set(m2 for m2, r2 in existing_classes)
        best_r = valid_residues[0]
        best_count = -1

        # Limit evaluation to a subset if too many candidates
        if len(valid_residues) > 25:
            step = max(1, len(valid_residues) // 25)
            candidates = valid_residues[::step]
        else:
            candidates = valid_residues

        for r_cand in candidates:
            # Evaluate: total valid residues available for all unused moduli given this choice
            score_cand = 0
            for m2 in moduli:
                if m2 in used_moduli or m2 == m:
                    continue
                j2 = mod_idx[m2]
                g2 = gcd_table[mod_idx[m]][j2]
                if g2 == 1:
                    score_cand += m2
                    continue
                r0_cand = r_cand % g2
                n_forbidden = m2 // g2 + (1 if m2 % g2 > r0_cand else 0)
                n_valid = m2 - n_forbidden
                score_cand += n_valid
            if score_cand > best_count:
                best_count = score_cand
                best_r = r_cand

        return best_r

    def score(classes):
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total

    def greedy_construct(order=None):
        if order is None:
            order = sorted(moduli)
        classes = []
        for m in order:
            valid_mask, count = compute_valid_mask(m, classes)
            if count > 0:
                r = pick_best_residue(valid_mask, m, classes)
                classes.append([m, r])
        return classes

    def try_add_class(classes, m):
        """Try to add modulus m to classes; return (new_classes, r) or (None, None)."""
        valid_mask, count = compute_valid_mask(m, classes)
        if count > 0:
            r = pick_best_residue(valid_mask, m, classes)
            return classes + [[m, r]], r
        return None, None

    def residue_polish(classes, max_iter=500):
        """Residue reassignment polish: try changing residues of existing classes
        to free up space for additional small moduli."""
        best = [list(c) for c in classes]
        best_score = score(best)
        used_moduli = set(m for m, r in best)

        for it in range(max_iter):
            if time.time() > deadline:
                break

            improved = False

            # Try to add any unused modulus
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                valid_mask, count = compute_valid_mask(m, best)
                if count > 0:
                    r = pick_best_residue(valid_mask, m, best)
                    new_classes = best + [[m, r]]
                    new_score = score(new_classes)
                    if new_score > best_score:
                        best = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break

            if not improved:
                # Try reassigning a residue to free space
                for i in range(len(best)):
                    if time.time() > deadline:
                        break
                    m_i, r_i = best[i]
                    # Get valid residues for this modulus given all OTHER classes
                    others = [best[j] for j in range(len(best)) if j != i]
                    valid_mask, count = compute_valid_mask(m_i, others)
                    if count <= 1:
                        continue

                    # Try a few alternative residues
                    alt_residues = []
                    for r in range(m_i):
                        if (valid_mask >> r) & 1 and r != r_i:
                            alt_residues.append(r)

                    if not alt_residues:
                        continue

                    # Sample up to 10 alternatives
                    if len(alt_residues) > 10:
                        random.shuffle(alt_residues)
                        alt_residues = alt_residues[:10]

                    for r_new in alt_residues:
                        trial = [list(c) for c in best]
                        trial[i] = [m_i, r_new]
                        # Try to add a missing modulus
                        added = False
                        for m_add in sorted(moduli):
                            if m_add in used_moduli:
                                continue
                            vm, cnt = compute_valid_mask(m_add, trial)
                            if cnt > 0:
                                r_add = pick_best_residue(vm, m_add, trial)
                                trial.append([m_add, r_add])
                                new_score = score(trial)
                                if new_score > best_score:
                                    best = trial
                                    best_score = new_score
                                    used_moduli.add(m_add)
                                    improved = True
                                    added = True
                                    break
                        if added:
                            break
                    if improved:
                        break

            if not improved:
                break

        return best

    def local_search(classes, max_iter=2000):
        """Local search: add missing small moduli, 1-for-2 swaps, and residue reassignment."""
        best = [list(c) for c in classes]
        best_score = score(best)
        used_moduli = set(m for m, r in best)

        for it in range(max_iter):
            if time.time() > deadline:
                break

            improved = False

            # Pass A: try to add any unused modulus
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                valid_mask, count = compute_valid_mask(m, best)
                if count > 0:
                    r = pick_best_residue(valid_mask, m, best)
                    new_classes = best + [[m, r]]
                    new_score = score(new_classes)
                    if new_score > best_score:
                        best = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break

            if not improved:
                # Pass B: 1-for-2 swap
                for i in range(len(best)):
                    if time.time() > deadline:
                        break
                    m_rem, r_rem = best[i]
                    temp = [best[j] for j in range(len(best)) if j != i]
                    temp_used = set(m for m, r in temp)
                    added = 0
                    for m in sorted(moduli):
                        if m in temp_used or m == m_rem:
                            continue
                        valid_mask, count = compute_valid_mask(m, temp)
                        if count > 0:
                            r = pick_best_residue(valid_mask, m, temp)
                            temp = temp + [[m, r]]
                            temp_used.add(m)
                            added += 1
                            if added == 2:
                                break
                    if added > 0:
                        new_score = score(temp)
                        if new_score > best_score:
                            best = temp
                            best_score = new_score
                            used_moduli = set(m for m, r in best)
                            improved = True
                            break

            if not improved:
                break

        # Final residue polish
        best = residue_polish(best, max_iter=300)

        return best

    # Multi-start search
    best_classes = None
    best_score = Fraction(0, 1)

    # Start 1: pure greedy, smallest moduli first
    classes = greedy_construct()
    classes = local_search(classes, max_iter=800)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]

    # Multiple randomized starts with diverse strategies
    random.seed(42)
    num_starts = 60
    for start in range(num_starts):
        if time.time() > deadline:
            break

        if start % 3 == 0:
            # Weighted random ordering: favor small moduli
            alpha = 1.0 + 0.5 * random.random()
            weights = [(m, 1.0 / (m ** alpha)) for m in moduli]
            order = []
            remaining = list(weights)
            while remaining:
                total_w = sum(w for _, w in remaining)
                if total_w <= 0:
                    order.append(remaining[0][0])
                    remaining.pop(0)
                    continue
                r = random.random() * total_w
                cum = 0
                chosen_idx = 0
                for i, (m, w) in enumerate(remaining):
                    cum += w
                    if r <= cum:
                        chosen_idx = i
                        break
                order.append(remaining[chosen_idx][0])
                remaining.pop(chosen_idx)
        elif start % 3 == 1:
            # Reverse weighted: slightly favor larger moduli (to explore different structures)
            alpha = 0.5 + 0.5 * random.random()
            weights = [(m, 1.0 / (m ** alpha)) for m in moduli]
            order = []
            remaining = list(weights)
            while remaining:
                total_w = sum(w for _, w in remaining)
                if total_w <= 0:
                    order.append(remaining[0][0])
                    remaining.pop(0)
                    continue
                r = random.random() * total_w
                cum = 0
                chosen_idx = 0
                for i, (m, w) in enumerate(remaining):
                    cum += w
                    if r <= cum:
                        chosen_idx = i
                        break
                order.append(remaining[chosen_idx][0])
                remaining.pop(chosen_idx)
        else:
            # Random permutation with slight bias toward small
            order = list(moduli)
            random.shuffle(order)
            # Sort with noise
            order.sort(key=lambda m: m + random.uniform(0, 30))

        classes = greedy_construct(order)
        classes = local_search(classes, max_iter=400)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]

    # Final intensive local search and polish on best
    if best_classes is not None:
        best_classes = local_search(best_classes, max_iter=1500)

    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
