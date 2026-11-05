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

def construct():
    base = 51975
    deadline = time.time() + 140.0  # safety margin before 150s kill
    divisors = _divisors_of(base)
    n_div = len(divisors)
    
    # Precompute for each divisor m the list of residues r in [0, m)
    # and the step pattern indices
    residue_lists = {m: list(range(m)) for m in divisors}
    
    # Greedy with randomized look-ahead and multiple restarts
    best_classes = None
    best_score = base  # worst case: nothing covered
    
    # First, do a deterministic greedy pass to get a good initial solution
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    remaining_moduli = list(divisors)
    for m in remaining_moduli:
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = int((covered[r::m] == 0).sum())
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
        if best_gain == 0:
            # No more gain possible from this modulus, but we must include it
            # Actually we can skip it since it adds no coverage
            classes.pop()
            continue
    s = int((covered == 0).sum())
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
    
    # Now do randomized greedy with look-ahead for improvement
    rng = np.random.RandomState(42)
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        remaining = list(divisors)
        rng.shuffle(remaining)
        
        # Randomized greedy: at each step, pick among top-k candidates
        top_k = 3
        for m in remaining:
            if time.time() > deadline:
                break
            # Compute gains for all residues of this modulus
            gains = np.array([(covered[r::m] == 0).sum() for r in range(m)], dtype=np.int32)
            if gains.max() == 0:
                continue
            # Pick from top-k residues with some randomness
            top_indices = np.argsort(gains)[::-1][:top_k]
            # Weight by gain for selection
            weights = gains[top_indices].astype(np.float64)
            weights = weights / weights.sum()
            r = int(top_indices[rng.choice(len(top_indices), p=weights)])
            classes.append([m, r])
            covered[r::m] = 1
        
        s = int((covered == 0).sum())
        if s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
    
    # Local improvement: try swapping residues for each modulus
    if best_classes is not None:
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                covered = np.zeros(base, dtype=np.uint8)
                for m, r in best_classes:
                    covered[r::m] = 1
                current_score = int((covered == 0).sum())
                
                for i, (m, r) in enumerate(best_classes):
                    if time.time() > deadline:
                        break
                    # Try all alternative residues for this modulus
                    best_alt_r = r
                    best_alt_score = current_score
                    for alt_r in range(m):
                        if alt_r == r:
                            continue
                        # Temporarily swap
                        # Remove old coverage contribution (approximate: just recompute)
                        # For efficiency, only try a subset if m is large
                        if m > 100 and alt_r % 2 != 0:
                            continue  # skip some for speed
                        # Recompute score with this change
                        cov2 = np.zeros(base, dtype=np.uint8)
                        for j, (mj, rj) in enumerate(best_classes):
                            if j == i:
                                cov2[alt_r::mj] = 1
                            else:
                                cov2[rj::mj] = 1
                        s2 = int((cov2 == 0).sum())
                        if s2 < best_alt_score:
                            best_alt_score = s2
                            best_alt_r = alt_r
                    if best_alt_r != r:
                        best_classes[i][1] = best_alt_r
                        improved = True
                        break  # restart the while loop
    
    # Final polish: try adding back skipped moduli if they help
    if best_classes is not None and time.time() < deadline:
        used_moduli = set(m for m, r in best_classes)
        for m in divisors:
            if m in used_moduli:
                continue
            if time.time() > deadline:
                break
            covered = np.zeros(base, dtype=np.uint8)
            for mj, rj in best_classes:
                covered[rj::mj] = 1
            current_score = int((covered == 0).sum())
            best_r = -1
            best_s = current_score
            for r in range(m):
                covered[r::m] = 1
                s = int((covered == 0).sum())
                if s < best_s:
                    best_s = s
                    best_r = r
                covered[r::m] = 0
            if best_r >= 0 and best_s < current_score:
                best_classes.append([m, best_r])
    
    return best_classes if best_classes is not None else []

# EVOLVE-BLOCK-END
