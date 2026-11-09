# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
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
    
    # Precompute for each modulus m, for each g that is a gcd with another modulus,
    # the list of residues r in [0,m) grouped by r % g
    # residue_groups[m][g][k] = list of r in [0,m) where r % g == k
    residue_groups = {}
    for m in moduli:
        residue_groups[m] = {}
        # Find all g values that appear as gcd(m, m2) for some m2
        g_set = set()
        for m2 in moduli:
            if m2 != m:
                g_set.add(gcd(m, m2))
        for g in g_set:
            if g == 1:
                continue  # no constraint
            groups = {}
            for r in range(m):
                k = r % g
                if k not in groups:
                    groups[k] = []
                groups[k].append(r)
            residue_groups[m][g] = groups
    
    def compute_forbidden_mask(m, r2, g):
        """Bitmask of residues r in [0, m) where r ≡ r2 (mod g)."""
        mask = 0
        r0 = r2 % g
        for r in range(r0, m, g):
            mask |= (1 << r)
        return mask
    
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
                # gcd 1 means all residues are forbidden
                return 0, 0
            # Forbidden: r ≡ r2 (mod g)
            r0 = r2 % g
            forbidden = 0
            for r in range(r0, m_val, g):
                forbidden |= (1 << r)
            valid &= ~forbidden
        
        count = bin(valid).count('1')
        return valid, count
    
    def pick_best_residue(valid_mask, m, existing_classes):
        """Pick a residue from valid_mask that maximizes flexibility for future additions.
        Strategy: for each candidate residue, compute how many future moduli would still be addable.
        Approximate: pick residue that minimizes the number of 'tight' constraints imposed."""
        if valid_mask == 0:
            return None
        
        # Get all valid residues
        valid_residues = []
        r = 0
        while (valid_mask >> r) & 1 == 0:
            r += 1
        while r < m:
            if (valid_mask >> r) & 1:
                valid_residues.append(r)
            r += 1
        
        if len(valid_residues) == 1:
            return valid_residues[0]
        
        # For each candidate, estimate how many unused moduli remain addable
        used_moduli = set(m2 for m2, r2 in existing_classes)
        best_r = valid_residues[0]
        best_count = -1
        
        # Limit evaluation to a subset if too many candidates
        if len(valid_residues) > 20:
            # Sample up to 20 candidates
            step = max(1, len(valid_residues) // 20)
            candidates = valid_residues[::step]
        else:
            candidates = valid_residues
        
        for r_cand in candidates:
            # Count how many unused moduli could still be added
            addable = 0
            for m2 in moduli:
                if m2 in used_moduli or m2 == m:
                    continue
                i2 = mod_idx[m]
                j2 = mod_idx[m2]
                g2 = gcd_table[i2][j2]
                if g2 == 1:
                    continue
                # Check if r_cand is compatible with m2 having some residue
                # We need to check if there exists r2 in [0, m2) such that r2 != r_cand (mod g2)
                # This is always possible unless g2 == 1 (already handled) or m2 < g2 (impossible since g2 <= m2)
                # Actually, the constraint is: for the EXISTING class (m, r_cand), when adding (m2, r2),
                # we need r2 != r_cand (mod g2). There are m2 - (m2//g2 or so) valid residues.
                # So it's always addable if g2 > 1.
                # The real constraint is between existing classes, not between new and existing in this direction.
                # For counting flexibility, we just check if g2 > 1
                addable += 1
            
            # Simpler heuristic: pick residue that is 'central' in its residue class mod small gcds
            # Or just pick the one that minimizes sum of r_cand % g for small g
            # Let's use: minimize the number of conflicts with future small moduli
            # Approximate: prefer residues that are 0 mod small numbers (more symmetric)
            # Actually, let's just pick the middle one for speed
            if len(candidates) <= 10:
                # Evaluate more carefully
                score_cand = 0
                for m2 in moduli:
                    if m2 in used_moduli or m2 == m:
                        continue
                    j2 = mod_idx[m2]
                    g2 = gcd_table[mod_idx[m]][j2]
                    if g2 > 1:
                        # The number of valid residues for m2 given (m, r_cand) is m2 - len([r for r in range(m2) if r % g2 == r_cand % g2])
                        # = m2 - (m2 // g2 + (1 if m2 % g2 > r_cand % g2 else 0))
                        r0_cand = r_cand % g2
                        n_forbidden = m2 // g2 + (1 if m2 % g2 > r0_cand else 0)
                        n_valid = m2 - n_forbidden
                        score_cand += n_valid
                if score_cand > best_count:
                    best_count = score_cand
                    best_r = r_cand
            else:
                # Just pick middle
                best_r = candidates[len(candidates) // 2]
        
        return best_r
    
    def score(classes):
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total
    
    # Greedy construction with priority on small moduli
    def greedy_construct(order=None):
        if order is None:
            order = sorted(moduli)  # smallest first
        classes = []
        for m in order:
            valid_mask, count = compute_valid_mask(m, classes)
            if count > 0:
                r = pick_best_residue(valid_mask, m, classes)
                classes.append([m, r])
        return classes
    
    def try_add_class(classes, used, m):
        """Try to add modulus m to classes; return (new_classes, r) or (None, None)."""
        valid_mask, count = compute_valid_mask(m, classes)
        if count > 0:
            r = pick_best_residue(valid_mask, m, classes)
            return classes + [[m, r]], r
        return None, None

    def local_search(classes, max_iter=2000):
        """Local search: add missing small moduli, 1-for-2 swaps, and residue reassignment."""
        best = [list(c) for c in classes]
        best_score = score(best)
        used_moduli = set(m for m, r in best)

        for it in range(max_iter):
            if time.time() > deadline:
                break

            improved = False

            # Pass A: try to add any unused modulus (smallest m first = largest 1/m)
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                new_classes, r = try_add_class(best, used_moduli, m)
                if new_classes is not None:
                    best = new_classes
                    best_score = score(best)
                    used_moduli.add(m)
                    improved = True
                    break

            if not improved:
                # Pass B: 1-for-2 swap (remove one, add up to two)
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
                        new_temp, r = try_add_class(temp, temp_used, m)
                        if new_temp is not None:
                            temp = new_temp
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
                # Pass C: residue reassignment to free space for a missing small modulus
                # For each missing small modulus, try reassigning residues of existing
                # classes to make room, accepting only if net score improves.
                missing = [m for m in sorted(moduli) if m not in used_moduli]
                if missing:
                    target_m = missing[0]
                    # Try reassigning one existing class's residue to free the target
                    for i in range(len(best)):
                        if time.time() > deadline:
                            break
                        m_i, r_i = best[i]
                        # Try all valid residues for class i (given others)
                        others = [best[j] for j in range(len(best)) if j != i]
                        valid_mask, count = compute_valid_mask(m_i, others)
                        if count == 0:
                            continue
                        # Enumerate valid residues for m_i
                        r = 0
                        found = False
                        while r < m_i and not found:
                            if (valid_mask >> r) & 1:
                                candidate = [best[j] if j != i else [m_i, r]
                                             for j in range(len(best))]
                                # Now try to add target_m
                                new_classes, r_t = try_add_class(candidate, used_moduli, target_m)
                                if new_classes is not None:
                                    new_score = score(new_classes)
                                    if new_score > best_score:
                                        best = new_classes
                                        best_score = new_score
                                        used_moduli = set(m for m, r in best)
                                        improved = True
                                        found = True
                            r += 1
                        if improved:
                            break

            if not improved:
                break

        return best
    
    # Multi-start search
    best_classes = None
    best_score = Fraction(0, 1)
    
    # Start 1: pure greedy, smallest moduli first
    classes = greedy_construct()
    classes = local_search(classes, max_iter=500)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Multiple randomized starts
    random.seed(42)
    num_starts = 80
    for start in range(num_starts):
        if time.time() > deadline - 20.0:  # reserve time for final polish
            break
        
        # Create weighted random ordering: strongly favor small moduli
        alpha = 1.0 + 0.5 * random.random()
        weights = [(m, 1.0 / (m ** alpha)) for m in moduli]
        
        # Weighted sampling without replacement
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
        
        classes = greedy_construct(order)
        classes = local_search(classes, max_iter=300)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final intensive local search on best (polish pass with more iterations)
    if best_classes is not None:
        best_classes = local_search(best_classes, max_iter=3000)

    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
