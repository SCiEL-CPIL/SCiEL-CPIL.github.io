"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

BASE = 51975
DEADLINE = 145  # safety margin before 150s kill

def get_divisors(n):
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return sorted(divs)

MODULI = get_divisors(BASE)

def count_uncovered(classes):
    """Count residues not covered by any class. classes: list of (m, r)."""
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.sum(covered == 0))

def greedy_init(order=None):
    """Greedy: for each modulus in order, pick residue covering most uncovered."""
    if order is None:
        order = sorted(MODULI)
    covered = np.zeros(BASE, dtype=np.uint8)
    classes = []
    for m in order:
        best_r = 0
        best_count = -1
        for r in range(m):
            cnt = int(np.sum(covered[r::m] == 0))
            if cnt > best_count:
                best_count = cnt
                best_r = r
        classes.append((m, best_r))
        covered[best_r::m] = 1
    return classes

def local_search(classes, time_budget):
    """Local search: try changing residue of each class, accept if improves."""
    classes = list(classes)
    uncovered = count_uncovered(classes)
    start = time.time()
    
    while time.time() - start < time_budget:
        improved = False
        # Shuffle order to avoid cycles
        idx_order = list(range(len(classes)))
        random.shuffle(idx_order)
        
        for i in idx_order:
            if time.time() - start >= time_budget:
                break
            m, r = classes[i]
            # Try all residues for this modulus
            best_r = r
            best_unc = uncovered
            # Current contribution: how many newly covered if we change
            # We need to recompute: remove class i, try each residue
            # Efficient: for each candidate r', compute uncovered
            # This is O(m * BASE/m) per candidate = O(BASE) per candidate
            # Too slow for large m. Instead, compute delta.
            
            # Current covered by class i: positions r, r+m, r+2m, ...
            # For each candidate r', the change is:
            #   newly covered: positions r', r'+m, ... that were NOT covered by other classes
            #   newly uncovered: positions r, r+m, ... that were ONLY covered by class i
            # 
            # To do this efficiently, we need a count array.
            
            # Let's use a different approach: just try a few random residues
            # plus the current one, and compute uncovered from scratch for small m
            # For large m, the number of positions is small (BASE/m), so it's fast.
            
            # Actually, let's compute the coverage count for each position
            # This is expensive to maintain, so let's just do full recompute for
            # promising candidates.
            
            # Try current residue and a few random ones
            candidates = [r]
            for _ in range(min(5, m - 1)):
                candidates.append(random.randrange(m))
            
            for r_new in candidates:
                if r_new == r:
                    continue
                new_classes = list(classes)
                new_classes[i] = (m, r_new)
                unc = count_uncovered(new_classes)
                if unc < best_unc:
                    best_unc = unc
                    best_r = r_new
            
            if best_r != r:
                classes[i] = (m, best_r)
                uncovered = best_unc
                improved = True
        
        if not improved:
            # Try a random perturbation (kick)
            i = random.randrange(len(classes))
            m, r = classes[i]
            r_new = random.randrange(m)
            if r_new != r:
                new_classes = list(classes)
                new_classes[i] = (m, r_new)
                unc = count_uncovered(new_classes)
                if unc <= uncovered + 2:  # accept slightly worse
                    classes = new_classes
                    uncovered = unc
                    improved = True
            if not improved:
                break
    
    return classes, uncovered

def construct():
    start_time = time.time()
    best_classes = None
    best_score = BASE  # worst case
    
    # Try multiple greedy initializations with different orderings
    rng = random.Random(42)
    
    # 1. Greedy with sorted moduli
    classes = greedy_init(sorted(MODULI))
    score = count_uncovered(classes)
    if score < best_score:
        best_score = score
        best_classes = list(classes)
    
    # 2. Greedy with reverse sorted
    if time.time() - start_time < DEADLINE:
        classes = greedy_init(sorted(MODULI, reverse=True))
        score = count_uncovered(classes)
        if score < best_score:
            best_score = score
            best_classes = list(classes)
    
    # 3. Greedy with random orderings
    for _ in range(5):
        if time.time() - start_time >= DEADLINE:
            break
        order = MODULI[:]
        rng.shuffle(order)
        classes = greedy_init(order)
        score = count_uncovered(classes)
        if score < best_score:
            best_score = score
            best_classes = list(classes)
    
    # 4. Local search from best
    if time.time() - start_time < DEADLINE:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 10:
            classes, score = local_search(best_classes, remaining * 0.7)
            if score < best_score:
                best_score = score
                best_classes = classes
    
    # 5. More restarts if time remains
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        for trial in range(20):
            if time.time() - start_time >= DEADLINE - 5:
                break
            order = MODULI[:]
            rng.shuffle(order)
            classes = greedy_init(order)
            score = count_uncovered(classes)
            if score < best_score:
                best_score = score
                best_classes = list(classes)
            # Quick local search
            remaining = DEADLINE - (time.time() - start_time)
            if remaining > 15 and score <= best_score + 50:
                classes, score = local_search(classes, min(remaining * 0.5, 10))
                if score < best_score:
                    best_score = score
                    best_classes = classes
    
    # Final local search on best
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        remaining = DEADLINE - elapsed - 1
        if remaining > 5:
            classes, score = local_search(best_classes, remaining)
            if score < best_score:
                best_score = score
                best_classes = classes
    
    return [[m, r] for m, r in best_classes]
# EVOLVE-BLOCK-END
