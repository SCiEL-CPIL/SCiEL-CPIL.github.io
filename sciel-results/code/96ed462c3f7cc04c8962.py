"""Fast vectorized approach to odd covering systems."""
import time
import random
import numpy as np

BASE = 51975

# Precompute all odd divisors > 1
_DIVISORS = [d for d in range(3, BASE + 1, 2) if BASE % d == 0]
_DIVISORS_SET = set(_DIVISORS)

# Precompute indices for each modulus to speed up mask updates
# indices[m] is a numpy array of indices i such that i % m == r? No, we need to handle r.
# It's better to precompute the full mask for each (m, r) or compute it on the fly.
# Given BASE is small (52k), computing mask is fast.
# mask = np.zeros(BASE, dtype=bool)
# mask[r::m] = True

def _get_mask(m, r):
    """Return a boolean numpy array of length BASE where True indicates coverage."""
    mask = np.zeros(BASE, dtype=bool)
    mask[r::m] = True
    return mask

def _count_uncovered(covered_mask):
    """Count number of False entries in covered_mask."""
    return np.count_nonzero(~covered_mask)

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 150s limit (evaluator kills at 120s, buffer for overhead)
    
    # Initial Greedy Construction
    # Sort divisors by size (larger moduli cover fewer points, so maybe start with smaller? 
    # Actually, in covering systems, larger moduli are often "more specific". 
    # Let's try adding divisors in an order that maximizes gain.
    # Greedy: Start with empty set. At each step, pick (m, r) that minimizes uncovered count.
    
    covered = np.zeros(BASE, dtype=bool)
    current_classes = []
    current_uncovered = BASE
    
    # To make greedy fast, we can pre-sort divisors. 
    # Let's try adding all divisors? No, we want fewest uncovered, but we are constrained to distinct moduli.
    # The task says "classes [[m, r], ...] with distinct moduli". It doesn't say we MUST use all divisors.
    # "With every divisor of 51975 available once" implies we can choose a subset.
    # Minimizing uncovered residues.
    
    # A good heuristic: Start with the largest moduli? Or smallest?
    # Smallest moduli cover more residues. 
    # Let's try a randomized greedy restart approach.
    
    best_classes = []
    best_score = BASE
    
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        # Randomize the order of divisors to explore different local minima
        divs = _DIVISORS[:]
        random.shuffle(divs)
        
        # Greedy Add
        covered = np.zeros(BASE, dtype=bool)
        classes = []
        uncovered_count = BASE
        
        # We iterate through divisors. For each, we pick the best residue r.
        # To speed this up, we can skip divisors if they don't improve the score.
        # But finding the best r requires checking all r in [0, m).
        # For small m, this is fast. For large m, it's slower.
        
        # Optimization: Precompute masks for all (m, r)? 
        # Total masks = sum(m for m in divisors). 
        # Sum of divisors of 51975 is approx 51975 * (1+1/3+1/9+1/27)*(1+1/5+1/25)*(1+1/7)*(1+1/11)
        # Sigma(n) is roughly n * log(log n). For 52k, it's manageable.
        # But storing 1000+ masks of size 52k is ~50MB. Acceptable.
        # However, we can just compute on the fly.
        
        for m in divs:
            if time.time() - start_time > DEADLINE:
                break
            
            # Find best r for this m
            best_r = 0
            best_gain = -1
            
            # Optimization: If m is large, the number of residues is small (BASE//m).
            # If m is small, the number of residues is large.
            
            # Try all residues?
            # For m=3, 3 residues. For m=51975, 1 residue.
            # Average m is moderate.
            # Let's just try all residues for now, it's O(m * BASE/m) = O(BASE) per m?
            # No, mask creation is O(BASE). So O(m * BASE) total per m?
            # Wait, mask[r::m] creates a view or copy?
            # np.zeros + slice assignment is fast.
            
            # We can vectorize the search for best r?
            # For a fixed m, we want to find r that minimizes sum(covered & ~mask(r))
            # which is equivalent to maximizing sum(~covered & mask(r))
            
            # Let's just loop r.
            for r in range(m):
                mask = np.zeros(BASE, dtype=bool)
                mask[r::m] = True
                # Count new coverage
                # gain = count of (mask AND NOT covered)
                # new_covered = covered | mask
                # new_uncovered = BASE - count(new_covered)
                
                # Efficiently:
                # gain = np.count_nonzero(mask & ~covered)
                # if gain > best_gain: ...
                
                # To avoid creating a new boolean array for intersection every time, 
                # we can just count.
                gain = np.count_nonzero(mask & ~covered)
                
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                # Add this class
                classes.append([m, best_r])
                # Update covered
                mask = np.zeros(BASE, dtype=bool)
                mask[best_r::m] = True
                covered |= mask
                uncovered_count -= best_gain
                
                if uncovered_count == 0:
                    break
        
        current_score = uncovered_count
        if current_score < best_score:
            best_score = current_score
            best_classes = classes[:]
            
        # Local Search / Refinement
        # Try to improve the current best solution found in this restart
        # We can try changing residues of existing classes or adding/removing.
        # Given time constraints, a simple hill-climbing on residues is good.
        
        if time.time() - start_time > DEADLINE:
            continue
            
        # Work on a copy
        curr_classes = best_classes[:]
        curr_covered = np.zeros(BASE, dtype=bool)
        for m, r in curr_classes:
            mask = np.zeros(BASE, dtype=bool)
            mask[r::m] = True
            curr_covered |= mask
        curr_score = np.count_nonzero(~curr_covered)
        
        improved = True
        max_iterations = 50
        iter_count = 0
        
        while improved and iter_count < max_iterations and time.time() - start_time < DEADLINE:
            improved = False
            iter_count += 1
            
            # 1. Try changing residue of existing classes
            for i in range(len(curr_classes)):
                m, r = curr_classes[i]
                old_mask = np.zeros(BASE, dtype=bool)
                old_mask[r::m] = True
                
                # Remove old coverage
                # Note: other classes might cover these spots, so we can't just unset bits.
                # We must recompute covered mask if we remove a class.
                # This is expensive. 
                # Alternative: Keep a count of coverage? 
                # Or just recompute the full mask from scratch for each candidate change?
                # With ~10-20 classes, recomputing is fast (20 * 52k ops).
                
                best_new_r = r
                best_new_score = curr_score
                
                # Try a few random residues + neighbors
                candidates = [r]
                if m > 1:
                    candidates.append((r - 1) % m)
                    candidates.append((r + 1) % m)
                    if m > 10:
                        candidates.extend(random.sample(range(m), min(5, m)))
                
                for new_r in candidates:
                    if new_r == r:
                        continue
                    
                    # Build new covered mask
                    new_covered = np.zeros(BASE, dtype=bool)
                    for j, (mj, rj) in enumerate(curr_classes):
                        if j == i:
                            mask_j = np.zeros(BASE, dtype=bool)
                            mask_j[new_r::mj] = True
                        else:
                            mask_j = np.zeros(BASE, dtype=bool)
                            mask_j[rj::mj] = True
                        new_covered |= mask_j
                    
                    new_score = np.count_nonzero(~new_covered)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_r = new_r
                
                if best_new_r != r and best_new_score < curr_score:
                    curr_classes[i][1] = best_new_r
                    curr_score = best_new_score
                    improved = True
            
            # 2. Try adding a new class
            # Pick a random unused modulus
            used_mods = set(m for m, _ in curr_classes)
            available = [m for m in _DIVISORS if m not in used_mods]
            
            if available:
                # Sample a few
                samples = random.sample(available, min(5, len(available)))
                for m in samples:
                    best_r = 0
                    best_gain = -1
                    for r in range(m):
                        mask = np.zeros(BASE, dtype=bool)
                        mask[r::m] = True
                        gain = np.count_nonzero(mask & ~curr_covered)
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                    
                    if best_gain > 0:
                        # Check if adding improves total score
                        # new_score = curr_score - best_gain
                        if curr_score - best_gain < curr_score:
                            curr_classes.append([m, best_r])
                            # Update covered
                            mask = np.zeros(BASE, dtype=bool)
                            mask[best_r::m] = True
                            curr_covered |= mask
                            curr_score -= best_gain
                            improved = True
            
            # 3. Try removing a class
            for i in range(len(curr_classes)):
                # Remove class i
                new_covered = np.zeros(BASE, dtype=bool)
                for j, (mj, rj) in enumerate(curr_classes):
                    if j == i:
                        continue
                    mask_j = np.zeros(BASE, dtype=bool)
                    mask_j[rj::mj] = True
                    new_covered |= mask_j
                
                new_score = np.count_nonzero(~new_covered)
                if new_score < curr_score:
                    # Remove it
                    del curr_classes[i]
                    curr_covered = new_covered
                    curr_score = new_score
                    improved = True
                    break # Restart loop as indices changed
        
        if curr_score < best_score:
            best_score = curr_score
            best_classes = curr_classes[:]
            
        if best_score == 0:
            break
            
    return best_classes
