import time
import numpy as np
import random

BASE = 51975
# Precompute all odd divisors > 1 of 51975
def _get_divisors():
    divs = []
    for d in range(3, BASE + 1, 2):
        if BASE % d == 0:
            divs.append(d)
    return divs

_DIVS = _get_divisors()

def _greedy_init(order, covered=None):
    """Greedy: given an order of moduli, pick best residue for each."""
    if covered is None:
        covered = np.zeros(BASE, dtype=np.uint8)
    classes = []
    for m in order:
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = np.count_nonzero(covered[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes, covered

def _score(classes):
    """Count uncovered residues."""
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.count_nonzero(covered == 0))

def _local_search(classes, time_limit=10.0):
    """Local search: try changing residues to reduce uncovered count.
    
    Uses proper coverage accounting: a position is covered if ANY class covers it.
    We track a count array to correctly compute marginal gains/losses.
    """
    start = time.time()
    n = len(classes)
    
    # Build coverage count: how many classes cover each position
    cov_count = np.zeros(BASE, dtype=np.int16)
    for m, r in classes:
        cov_count[r::m] += 1
    
    best_score = int(np.count_nonzero(cov_count == 0))
    best_classes = [c[:] for c in classes]
    
    # Precompute slice indices for each class
    # For class i with modulus m and residue r, positions are r, r+m, r+2m, ...
    
    for iteration in range(5000):
        if time.time() - start > time_limit:
            break
        
        # Try changing residue of a random class
        idx = random.randint(0, n - 1)
        m, r_old = classes[idx]
        
        # Try a few random residues (more for small moduli)
        num_trials = min(m, max(5, m // 3))
        for _ in range(num_trials):
            r_new = random.randint(0, m - 1)
            if r_new == r_old:
                continue
            
            # Compute marginal change correctly using cov_count
            # Positions in old class: r_old::m
            # Positions in new class: r_new::m
            
            old_positions = np.arange(r_old, BASE, m)
            new_positions = np.arange(r_new, BASE, m)
            
            # Gained: positions in new class that are currently uncovered (cov_count == 0)
            gained = np.count_nonzero(cov_count[new_positions] == 0)
            
            # Lost: positions in old class that are currently only covered by this class (cov_count == 1)
            # AND are not in the new class
            # Actually, we need to be more careful:
            # After removing old class and adding new class:
            # - A position in old_positions but not in new_positions: cov_count decreases by 1
            #   It becomes uncovered if cov_count was 1
            # - A position in new_positions but not in old_positions: cov_count increases by 1
            #   It becomes covered if cov_count was 0
            # - A position in both: no net change
            
            # Simpler approach: compute the exact delta
            # Create a set of new positions for fast lookup
            new_set = set(new_positions.tolist())
            
            # Lost: positions in old_positions where cov_count == 1 and position not in new_positions
            lost = 0
            for pos in old_positions:
                if cov_count[pos] == 1 and pos not in new_set:
                    lost += 1
            
            delta = gained - lost
            if delta > 0:
                # Apply the change
                cov_count[old_positions] -= 1
                cov_count[new_positions] += 1
                classes[idx][1] = r_new
                new_score = best_score - delta
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [c[:] for c in classes]
                break
    
    return best_classes, best_score

def _local_search_v2(classes, time_limit=10.0):
    """Faster local search using numpy operations without Python loops for position iteration."""
    start = time.time()
    n = len(classes)
    
    # Build coverage count
    cov_count = np.zeros(BASE, dtype=np.int16)
    for m, r in classes:
        cov_count[r::m] += 1
    
    best_score = int(np.count_nonzero(cov_count == 0))
    best_classes = [c[:] for c in classes]
    
    # For efficiency, precompute which positions each class covers
    # and use numpy vectorized operations
    
    for iteration in range(5000):
        if time.time() - start > time_limit:
            break
        
        idx = random.randint(0, n - 1)
        m, r_old = classes[idx]
        
        # Number of trials: try more for small moduli
        num_trials = min(m, max(3, m // 5))
        
        for _ in range(num_trials):
            r_new = random.randint(0, m - 1)
            if r_new == r_old:
                continue
            
            old_slice = slice(r_old, None, m)
            new_slice = slice(r_new, None, m)
            
            old_cov = cov_count[old_slice]
            new_cov = cov_count[new_slice]
            
            # Gained: positions in new slice with cov_count == 0
            gained = np.count_nonzero(new_cov == 0)
            
            # Lost: positions in old slice with cov_count == 1
            # But we need to exclude positions that are in BOTH old and new slices
            # A position x is in both if x ≡ r_old (mod m) AND x ≡ r_new (mod m)
            # This means r_old ≡ r_new (mod m), i.e., r_old == r_new, which we already excluded
            # So old and new slices are disjoint!
            
            lost = np.count_nonzero(old_cov == 1)
            
            delta = gained - lost
            if delta > 0:
                cov_count[old_slice] -= 1
                cov_count[new_slice] += 1
                classes[idx][1] = r_new
                new_score = best_score - delta
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [c[:] for c in classes]
                break
    
    return best_classes, best_score

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    best_classes = None
    best_score = BASE  # Worst case: nothing covered
    
    # Multiple greedy initializations with different orderings
    strategies = []
    
    # Ascending
    strategies.append(sorted(_DIVS))
    # Descending  
    strategies.append(sorted(_DIVS, reverse=True))
    # A few random orderings
    for _ in range(5):
        random_order = _DIVS[:]
        random.shuffle(random_order)
        strategies.append(random_order)
    
    for order in strategies:
        if time.time() > deadline - 20:
            break
        
        classes, covered = _greedy_init(order)
        score = int(np.count_nonzero(covered == 0))
        
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
        
        # Local search if we have time
        if time.time() < deadline - 15:
            ls_classes, ls_score = _local_search_v2(
                classes,
                time_limit=min(10.0, deadline - time.time() - 5)
            )
            if ls_score < best_score:
                best_score = ls_score
                best_classes = [c[:] for c in ls_classes]
        
        if best_score == 0:
            break
    
    if best_classes is None:
        # Fallback: simple ascending greedy
        best_classes, _ = _greedy_init(sorted(_DIVS))
    
    return best_classes
