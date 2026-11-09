"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_BASE = 51975
# Precompute all divisors of 51975 = 3^3 * 5^2 * 7 * 11, excluding 1
_DIVISORS = [d for d in range(2, _BASE + 1) if _BASE % d == 0]
# Sort divisors: we want to try larger moduli first (they cover fewer residues per class,
# but have more residue choices; smaller moduli cover more residues per class)
# Actually for greedy, we want to pick the class that covers the most UNCOVERED residues.
# A class (m, r) covers exactly _BASE // m residues.
# So smaller m covers more residues per class.
# We'll sort by m ascending so smaller moduli are considered first.
_DIVISORS_SORTED = sorted(_DIVISORS)

def _score(classes):
    """Return the number of uncovered residues modulo _BASE."""
    covered = bytearray(_BASE)
    for m, r in classes:
        # Mark all residues congruent to r mod m
        for x in range(r, _BASE, m):
            covered[x] = 1
    return _BASE - sum(covered)

def _greedy_construct(divisors, seed=0):
    """Greedy construction: repeatedly pick the class (m, r) that covers the most uncovered residues."""
    rng = random.Random(seed)
    covered = bytearray(_BASE)
    classes = []
    available = list(divisors)
    
    while available:
        best_m = None
        best_r = None
        best_gain = -1
        # Shuffle for tie-breaking variety
        rng.shuffle(available)
        for m in available:
            # For each residue r in [0, m), compute how many currently uncovered residues it covers
            # To speed this up, we can precompute or just iterate
            # Since m can be up to 51975, and we have up to ~40 divisors, this is manageable
            # But for large m, iterating over all r is expensive. Let's be smart.
            # For a given m, the gain for residue r is the number of x in [r, _BASE, m] where covered[x] == 0
            # We can compute this by checking the slice
            # Optimization: if m is large, _BASE // m is small, so each class covers few residues
            # Let's just compute it directly but break early if possible
            
            # Quick upper bound: max possible gain is _BASE // m
            # If even the max possible gain is less than best_gain, skip this m
            max_possible = _BASE // m
            if max_possible < best_gain:
                continue
            
            # Find the best residue for this m
            local_best_r = None
            local_best_gain = -1
            for r in range(m):
                # Count uncovered in range(r, _BASE, m)
                gain = 0
                for x in range(r, _BASE, m):
                    if not covered[x]:
                        gain += 1
                        if gain > local_best_gain + (max_possible - local_best_gain):
                            # Can't beat local best anymore for this r, but we still need to finish
                            pass
                if gain > local_best_gain:
                    local_best_gain = gain
                    local_best_r = r
                # Early termination: if we've found a residue covering all possible, break
                if local_best_gain == max_possible:
                    break
            
            if local_best_gain > best_gain:
                best_gain = local_best_gain
                best_m = m
                best_r = local_best_r
        
        if best_m is None or best_gain <= 0:
            break
        
        classes.append([best_m, best_r])
        # Update covered
        for x in range(best_r, _BASE, best_m):
            covered[x] = 1
        available.remove(best_m)
        
        # Check if fully covered
        if sum(covered) == _BASE:
            break
    
    return classes

def _local_search(classes, seed=0, max_iters=50):
    """Local search: try swapping or replacing classes to improve coverage."""
    rng = random.Random(seed)
    current_classes = [list(c) for c in classes]
    current_score = _score(current_classes)
    best_classes = [list(c) for c in current_classes]
    best_score = current_score
    
    for _ in range(max_iters):
        improved = False
        # Try replacing one class with a different modulus/residue
        for i in range(len(current_classes)):
            m_old, r_old = current_classes[i]
            # Try a random alternative modulus
            alt_m = rng.choice(_DIVISORS_SORTED)
            if alt_m == m_old:
                continue
            # Try a random residue for alt_m
            alt_r = rng.randint(0, alt_m - 1)
            
            # Test the replacement
            test_classes = [list(c) for c in current_classes]
            test_classes[i] = [alt_m, alt_r]
            test_score = _score(test_classes)
            
            if test_score < current_score:
                current_classes = test_classes
                current_score = test_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [list(c) for c in current_classes]
                break  # Restart the loop after an improvement
        
        if not improved:
            # Try some random perturbations
            for _ in range(10):
                i = rng.randint(0, len(current_classes) - 1)
                m_old = current_classes[i][0]
                alt_m = rng.choice(_DIVISORS_SORTED)
                if alt_m == m_old:
                    continue
                alt_r = rng.randint(0, alt_m - 1)
                test_classes = [list(c) for c in current_classes]
                test_classes[i] = [alt_m, alt_r]
                test_score = _score(test_classes)
                if test_score < current_score:
                    current_classes = test_classes
                    current_score = test_score
                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [list(c) for c in current_classes]
                    break
    
    return best_classes, best_score

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with greedy construction
    best_classes = _greedy_construct(_DIVISORS_SORTED, seed=0)
    best_score = _score(best_classes)
    
    if best_score == 0:
        return best_classes
    
    # Try local search to improve
    if time.time() < deadline:
        ls_classes, ls_score = _local_search(best_classes, seed=0, max_iters=30)
        if ls_score < best_score:
            best_score = ls_score
            best_classes = ls_classes
        if best_score == 0:
            return best_classes
    
    # Try multiple greedy restarts with different orderings
    for seed in range(1, 50):
        if time.time() >= deadline:
            break
        gc = _greedy_construct(_DIVISORS_SORTED, seed=seed)
        gs = _score(gc)
        if gs < best_score:
            best_score = gs
            best_classes = gc
            if best_score == 0:
                return best_classes
    
    # Final local search on the best found
    if time.time() < deadline - 10:
        ls_classes, ls_score = _local_search(best_classes, seed=42, max_iters=50)
        if ls_score < best_score:
            best_score = ls_score
            best_classes = ls_classes
    
    return best_classes
# EVOLVE-BLOCK-END
