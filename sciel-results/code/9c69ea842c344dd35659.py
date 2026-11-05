"""Odd covering systems with moduli dividing 51975 (fewest uncovered residues)."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _divisors_of(n):
    """Return all odd divisors > 1 of n, sorted ascending."""
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

def _score(classes, base):
    """Count residues covered by no class."""
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int((covered == 0).sum())

def _compute_gains(covered, m, base):
    """Compute gain for each residue r in [0, m) given current coverage."""
    gains = np.zeros(m, dtype=np.int32)
    for r in range(m):
        gains[r] = int((covered[r::m] == 0).sum())
    return gains

def _greedy_construct(divisors, base, order, rng=None, top_k=1, use_random=False):
    """Greedy construction with specified order and optional randomness."""
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    
    for m in order:
        if use_random and rng is not None:
            gains = _compute_gains(covered, m, base)
            if gains.max() == 0:
                continue
            # Pick from top-k residues with randomness
            k = min(top_k, m)
            top_indices = np.argsort(gains)[::-1][:k]
            weights = gains[top_indices].astype(np.float64)
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()
            r = int(top_indices[rng.choice(len(top_indices), p=weights)])
        else:
            # Deterministic: pick best residue
            gains = _compute_gains(covered, m, base)
            if gains.max() == 0:
                continue
            r = int(np.argmax(gains))
        
        classes.append([m, r])
        covered[r::m] = 1
    
    s = int((covered == 0).sum())
    return classes, s

def _local_search(classes, base, deadline, max_passes=10):
    """Local search: try swapping residues for each modulus to improve score."""
    if not classes:
        return classes, base
    
    # Precompute coverage
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    current_score = int((covered == 0).sum())
    
    improved = True
    passes = 0
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        
        for i, (m, r) in enumerate(classes):
            if time.time() > deadline:
                return classes, current_score
            
            # Compute gains for current residue and all alternatives
            gains = _compute_gains(covered, m, base)
            current_gain = gains[r]
            
            # Find best alternative
            best_alt_r = r
            best_gain = current_gain
            
            # Try all residues (optimized: only compute for residues that might improve)
            for alt_r in range(m):
                if alt_r == r:
                    continue
                alt_gain = gains[alt_r]
                if alt_gain > best_gain:
                    best_gain = alt_gain
                    best_alt_r = alt_r
            
            if best_alt_r != r:
                # Update coverage: remove old, add new
                # Count how many positions are uniquely covered by old residue
                old_positions = np.arange(r, base, m)
                new_positions = np.arange(best_alt_r, base, m)
                
                # Remove old coverage
                covered[r::m] = 0
                # Add new coverage  
                covered[best_alt_r::m] = 1
                
                # Recompute score
                new_score = int((covered == 0).sum())
                if new_score < current_score:
                    classes[i][1] = best_alt_r
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    covered[best_alt_r::m] = 0
                    covered[r::m] = 1
    
    return classes, current_score

def construct():
    base = 51975
    deadline = time.time() + 140.0  # safety margin before 150s kill
    divisors = _divisors_of(base)
    n_div = len(divisors)
    
    best_classes = None
    best_score = base  # worst case: nothing covered
    
    # Multiple greedy strategies with different orderings
    strategies = []
    
    # Strategy 1: Ascending order (small moduli first)
    strategies.append(('asc', sorted(divisors)))
    
    # Strategy 2: Descending order (large moduli first)
    strategies.append(('desc', sorted(divisors, reverse=True)))
    
    # Strategy 3: By prime factorization complexity (fewer prime factors first)
    def prime_factor_count(m):
        count = 0
        temp = m
        for p in [3, 5, 7, 11]:
            while temp % p == 0:
                count += 1
                temp //= p
        return count
    
    strategies.append(('pf', sorted(divisors, key=prime_factor_count)))
    
    # Strategy 4: By value modulo some small number
    strategies.append(('mod3', sorted(divisors, key=lambda m: m % 3)))
    strategies.append(('mod5', sorted(divisors, key=lambda m: m % 5)))
    
    # Run deterministic greedy for each strategy
    for name, order in strategies:
        if time.time() > deadline:
            break
        classes, s = _greedy_construct(divisors, base, order)
        if s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
    
    # Randomized greedy with multiple restarts
    rng = np.random.RandomState(42)
    max_restarts = 300
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random ordering with some structure
        order = list(divisors)
        rng.shuffle(order)
        
        # Use randomized greedy
        classes, s = _greedy_construct(divisors, base, order, rng=rng, top_k=3, use_random=True)
        if s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
    
    # Local search on best solution
    if best_classes is not None and time.time() < deadline:
        best_classes, best_score = _local_search(best_classes, base, deadline, max_passes=5)
    
    # Try local search on other good solutions too
    if time.time() < deadline:
        # Re-run a few good strategies with local search
        for name, order in strategies[:3]:
            if time.time() > deadline:
                break
            classes, s = _greedy_construct(divisors, base, order)
            if s < best_score + 5:  # Only if close to best
                classes, s = _local_search(classes, base, deadline, max_passes=3)
                if s < best_score:
                    best_score = s
                    best_classes = [c[:] for c in classes]
    
    # Final: try adding back skipped moduli if they help
    if best_classes is not None and time.time() < deadline:
        used_moduli = set(m for m, r in best_classes)
        # Recompute coverage
        covered = np.zeros(base, dtype=np.uint8)
        for mj, rj in best_classes:
            covered[rj::mj] = 1
        current_score = int((covered == 0).sum())
        
        for m in divisors:
            if m in used_moduli:
                continue
            if time.time() > deadline:
                break
            
            # Try to find best residue for this modulus
            gains = _compute_gains(covered, m, base)
            if gains.max() > 0:
                best_r = int(np.argmax(gains))
                # Check if adding this improves score
                covered[best_r::m] = 1
                new_score = int((covered == 0).sum())
                if new_score < current_score:
                    best_classes.append([m, best_r])
                    current_score = new_score
                else:
                    covered[best_r::m] = 0
    
    return best_classes if best_classes is not None else []

# EVOLVE-BLOCK-END
