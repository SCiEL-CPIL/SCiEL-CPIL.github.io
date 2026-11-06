import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find pairwise disjoint residue classes with distinct moduli in [20, 200]
    maximizing the sum of 1/m.
    
    Two classes (m1, r1) and (m2, r2) are disjoint iff r1 != r2 mod gcd(m1, m2).
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Precompute gcd for all pairs of moduli in [20, 200]
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # We'll use a greedy approach with backtracking limited by time.
    # Sort moduli by 1/m descending (i.e., m ascending) to prioritize high-value moduli.
    # For each modulus, we need to find a residue r in [0, m) that doesn't conflict
    # with already selected classes.
    
    # Greedy: process moduli in order of increasing m (highest 1/m first).
    # For each m, find any valid residue. If none exists, skip.
    
    # But greedy might not be optimal. Let's try a smarter approach:
    # Since the constraint is r_i != r_j mod gcd(m_i, m_j), and gcd can be small,
    # there are usually many valid residues. The main bottleneck is when gcd is large.
    
    # Strategy: 
    # 1. Try greedy in order of increasing m.
    # 2. If we have time, try local search or randomized restarts.
    
    def is_valid_residue(m_idx, r, selected):
        """Check if residue r for modulus at index m_idx is valid given selected classes."""
        m = moduli[m_idx]
        for (sel_idx, sel_r) in selected:
            g = gcd_table[m_idx][sel_idx]
            if r % g == sel_r % g:
                return False
        return True
    
    def greedy_construct():
        """Greedy: for each m in increasing order, pick the smallest valid residue."""
        selected = []
        score = Fraction(0)
        for m_idx in range(n_mod):
            m = moduli[m_idx]
            found = False
            for r in range(m):
                if is_valid_residue(m_idx, r, selected):
                    selected.append((m_idx, r))
                    score += Fraction(1, m)
                    found = True
                    break
            if not found:
                # Try other residues? Actually if no residue works, skip this m.
                pass
        return selected, score
    
    # First, try the simple greedy
    selected, score = greedy_construct()
    
    best_selected = list(selected)
    best_score = score
    
    # Now try to improve with local search / hill climbing
    # Idea: try swapping residues or adding/removing classes
    
    # Let's also try a different greedy: process moduli but choose residue more carefully
    # or try multiple orderings
    
    # Alternative: for each modulus, the number of forbidden residues from a selected class
    # is gcd(m_i, m_j) residues out of m_i. So the fraction of valid residues is roughly
    # prod(1 - gcd/m_i) over selected. This is usually close to 1 for small gcds.
    
    # The greedy should work well. Let's verify and try to improve.
    
    # Try randomized restarts with different residue choices
    import random
    rng = random.Random(42)
    
    start_time = time.time()
    
    # Try several randomized variants
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Randomized greedy: for each m in increasing order, pick a random valid residue
        # (or the first one, or a random one from valid ones)
        selected_rand = []
        score_rand = Fraction(0)
        
        # Shuffle the order of processing? No, we should still process small m first
        # but maybe we can be smarter about which residues to pick.
        
        # Actually, let's try: process moduli in increasing m, but for each, 
        # collect all valid residues and pick randomly.
        for m_idx in range(n_mod):
            m = moduli[m_idx]
            valid_residues = []
            for r in range(m):
                if is_valid_residue(m_idx, r, selected_rand):
                    valid_residues.append(r)
            if valid_residues:
                # Pick randomly to explore different solutions
                r = rng.choice(valid_residues)
                selected_rand.append((m_idx, r))
                score_rand += Fraction(1, m)
        
        if score_rand > best_score:
            best_score = score_rand
            best_selected = list(selected_rand)
    
    # Try local search: try to change one residue at a time to enable adding a skipped modulus
    # First, identify which moduli were skipped in the best solution
    selected_mods = set(idx for idx, _ in best_selected)
    skipped = [i for i in range(n_mod) if i not in selected_mods]
    
    # Try to add skipped moduli by adjusting existing residues
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for skip_idx in skipped:
            if time.time() > deadline:
                break
            m = moduli[skip_idx]
            # Try to find a valid residue for this m given current selection
            # If not possible, try to modify one existing class's residue to make room
            for r in range(m):
                if is_valid_residue(skip_idx, r, best_selected):
                    # Can add directly
                    best_selected.append((skip_idx, r))
                    best_score += Fraction(1, m)
                    selected_mods.add(skip_idx)
                    skipped.remove(skip_idx)
                    improved = True
                    break
            else:
                # Try modifying one existing class
                for i, (sel_idx, sel_r) in enumerate(best_selected):
                    g = gcd_table[skip_idx][sel_idx]
                    # We need sel_r' such that sel_r' % g != r % g for all r in [0,m)
                    # Actually we need: for the new r we pick for skip_idx, 
                    # sel_r' % g != r % g
                    # And sel_r' must still be valid with all other selected classes.
                    
                    # Try a few alternative residues for sel_idx
                    for new_r in range(moduli[sel_idx]):
                        if new_r == sel_r:
                            continue
                        # Check if new_r is valid for sel_idx with all other selected
                        valid = True
                        for j, (other_idx, other_r) in enumerate(best_selected):
                            if j == i:
                                continue
                            if new_r % gcd_table[sel_idx][other_idx] == other_r % gcd_table[sel_idx][other_idx]:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Now check if we can find an r for skip_idx that works with new_r
                        found_r = False
                        for r in range(m):
                            if r % g != new_r % g:
                                # Check against all other selected
                                ok = True
                                for j, (other_idx, other_r) in enumerate(best_selected):
                                    if j == i:
                                        continue
                                    if r % gcd_table[skip_idx][other_idx] == other_r % gcd_table[skip_idx][other_idx]:
                                        ok = False
                                        break
                                if ok:
                                    # Apply the change
                                    best_selected[i] = (sel_idx, new_r)
                                    best_selected.append((skip_idx, r))
                                    best_score += Fraction(1, m)
                                    selected_mods.add(skip_idx)
                                    skipped.remove(skip_idx)
                                    improved = True
                                    found_r = True
                                    break
                        if found_r:
                            break
                    if improved:
                        break
            if improved:
                break
    
    # Convert to output format
    result = [[moduli[idx], r] for idx, r in best_selected]
    return result
# EVOLVE-BLOCK-END
