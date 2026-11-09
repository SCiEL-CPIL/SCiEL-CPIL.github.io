import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
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

    # Precompute conflict masks for each modulus m and residue r
    # conflict_mask[m_idx][r] is a bitmask of other moduli that are blocked?
    # No, the conflict depends on the residue of the other modulus too.
    # So we can't precompute a static mask for "which moduli are blocked".
    # We can precompute, for each pair (m, m'), the condition:
    # (r - r') % gcd(m, m') == 0.
    
    # Let's use the conflict count array approach.
    # conflict_count[m_idx][r] = number of selected classes (m', r') such that (r - r') % gcd(m, m') == 0.
    
    # We will maintain this array dynamically.
    
    def init_conflict_counts():
        return [[0] * moduli[i] for i in range(n_mod)]

    def update_conflict_counts(conflict_counts, m_idx, r, delta):
        m = moduli[m_idx]
        for j in range(n_mod):
            if j == m_idx:
                continue
            g = gcd_table[m_idx][j]
            if g == 1:
                continue
            m_j = moduli[j]
            # For all r_j in [0, m_j), check if (r - r_j) % g == 0
            # This is equivalent to r_j % g == r % g
            # So we can just update the counts for r_j that satisfy this.
            r_mod_g = r % g
            for r_j in range(m_j):
                if r_j % g == r_mod_g:
                    conflict_counts[j][r_j] += delta

    def find_best_residue(conflict_counts, m_idx):
        """Find residue r for m_idx such that conflict_counts[m_idx][r] == 0.
        If multiple, pick the one that minimizes future conflicts (heuristic).
        For now, just pick the first one, or the one with min 'potential'.
        """
        m = moduli[m_idx]
        valid_rs = [r for r in range(m) if conflict_counts[m_idx][r] == 0]
        if not valid_rs:
            return None
        
        # Heuristic: pick r that minimizes the sum of 1/gcd(m, m') for m' not yet selected?
        # Or just pick the smallest r?
        # Smallest r is often good as it leaves "large" residues for others?
        # Actually, the constraint is modulo g. The specific value of r doesn't matter as much
        # as which residue class mod g it falls into.
        # All valid r are equivalent in terms of current conflicts (count is 0).
        # They differ in how they constrain FUTURE moduli.
        # A residue r constrains modulus m' by forbidding all r' such that r' % gcd(m, m') == r % gcd(m, m').
        # To maximize flexibility, we should pick r such that it forbids the "least" residues for other moduli?
        # No, the number of forbidden residues for m' is always gcd(m, m').
        # So all valid r are symmetric with respect to the *count* of blocked residues for any m'.
        # Thus, any valid r is equally good for future feasibility in terms of counts.
        # However, if we consider multiple future moduli, the specific residues might matter.
        # But for now, picking the smallest valid r is a reasonable deterministic choice.
        
        return valid_rs[0]

    def score(classes):
        total = 0.0
        for (m, r) in classes:
            total += 1.0 / m
        return total

    def greedy_construct(order=None):
        if order is None:
            order = sorted(moduli, key=lambda m: 1.0/m)
        
        classes = []
        used_moduli = set()
        conflict_counts = init_conflict_counts()
        
        for m in order:
            m_idx = mod_idx[m]
            r = find_best_residue(conflict_counts, m_idx)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
                update_conflict_counts(conflict_counts, m_idx, r, 1)
        
        return classes

    def local_search(classes, max_iter=1000):
        best_classes = [list(c) for c in classes]
        best_score = score(best_classes)
        used_moduli = set(m for m, r in best_classes)
        conflict_counts = init_conflict_counts()
        
        # Initialize conflict counts for the starting set
        for (m, r) in best_classes:
            m_idx = mod_idx[m]
            update_conflict_counts(conflict_counts, m_idx, r, 1)

        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            
            improved = False
            n_classes = len(best_classes)
            
            # 1. Try adding a new modulus
            # Sort missing moduli by 1/m descending
            missing = [m for m in sorted(moduli) if m not in used_moduli]
            
            for m in missing:
                m_idx = mod_idx[m]
                r = find_best_residue(conflict_counts, m_idx)
                if r is not None:
                    # Accept
                    best_classes.append([m, r])
                    used_moduli.add(m)
                    update_conflict_counts(conflict_counts, m_idx, r, 1)
                    best_score += 1.0 / m
                    improved = True
                    break
            
            # 2. Try Swap: Remove m_old, Add m_new
            if not improved:
                # We want to increase score, so 1/m_new > 1/m_old
                # Iterate missing m_new (small m) and existing m_old (large m)
                
                # To speed up, we can pre-compute a list of (m_old, index) sorted by m_old descending
                existing_sorted = sorted([(m, i) for i, (m, r) in enumerate(best_classes)], key=lambda x: x[0], reverse=True)
                
                found_swap = False
                for m_new in missing:
                    if time.time() - start_time > deadline - 3:
                        break
                    
                    m_new_idx = mod_idx[m_new]
                    
                    # Try removing m_old from the end of existing_sorted (largest m_old first)
                    # We need to check if removing m_old allows adding m_new.
                    # This requires updating conflict counts temporarily.
                    # To avoid expensive updates for every candidate, we can just check
                    # if there is ANY valid residue for m_new after removing m_old.
                    
                    # Optimization: Only try removing m_old if m_old > m_new
                    for m_old, i_old in existing_sorted:
                        if m_old <= m_new:
                            continue
                        
                        # Temporarily remove m_old
                        r_old = best_classes[i_old][1]
                        m_old_idx = mod_idx[m_old]
                        update_conflict_counts(conflict_counts, m_old_idx, r_old, -1)
                        
                        r_new = find_best_residue(conflict_counts, m_new_idx)
                        
                        if r_new is not None:
                            # Accept swap
                            # Remove old
                            best_classes.pop(i_old)
                            used_moduli.discard(m_old)
                            
                            # Add new
                            best_classes.append([m_new, r_new])
                            used_moduli.add(m_new)
                            update_conflict_counts(conflict_counts, m_new_idx, r_new, 1)
                            
                            best_score += 1.0 / m_new - 1.0 / m_old
                            
                            # Re-sort existing_sorted? Or just break and restart loop?
                            # Breaking and restarting is safer for correctness of sorted list.
                            found_swap = True
                            break
                        else:
                            # Restore m_old
                            update_conflict_counts(conflict_counts, m_old_idx, r_old, 1)
                    
                    if found_swap:
                        improved = True
                        break
            
            if not improved:
                # Try 2-for-1 swap? Or just break?
                # For now, break if no improvement in 1-for-1.
                # We can add a random restart or perturbation here.
                break
        
        return best_classes

    # Main execution
    best_classes = None
    best_score = 0.0
    
    # Start 1: Greedy
    classes = greedy_construct()
    classes = local_search(classes, max_iter=500)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Start 2: Randomized Greedy
    random.seed(42)
    for start_idx in range(20):
        if time.time() - start_time > deadline - 10:
            break
        
        # Random order biased towards small moduli
        remaining = list(moduli)
        order = []
        while remaining:
            # Weight by 1/m^alpha
            alpha = 1.0 + start_idx * 0.1
            weights = [1.0 / (m ** alpha) for m in remaining]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            order.append(remaining[chosen_idx])
            remaining.pop(chosen_idx)
        
        classes = greedy_construct(order)
        classes = local_search(classes, max_iter=200)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
            
    # Final polish
    if best_classes:
        best_classes = local_search(best_classes, max_iter=1000)
        best_score = score(best_classes)
        
    return best_classes if best_classes else []
