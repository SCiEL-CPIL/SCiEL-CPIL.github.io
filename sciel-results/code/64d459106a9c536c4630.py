"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

_BASE = 51975
_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _divisors():
    ds = []
    # Generate divisors of 51975 = 3^3 * 5^2 * 7 * 11
    # We want distinct moduli > 1.
    for d in range(3, _BASE + 1, 2):
        if _BASE % d == 0:
            ds.append(d)
    return ds

def _count_uncovered(covered):
    return int(np.count_nonzero(covered == 0))

def _evaluate_move(covered, m, r):
    """Calculate the change in uncovered count if we add class (m,r)."""
    # covered is a boolean array where True means covered.
    # We want to count how many 0s become 1s.
    mask = np.zeros(_BASE, dtype=bool)
    mask[r::m] = True
    # New covered = covered | mask
    # Uncovered change = (covered & ~mask).sum() - (~covered & mask).sum() is not quite right for score.
    # Score = count(covered == 0).
    # New score = count((covered | mask) == 0).
    # Delta = new_score - old_score = count(~covered & ~mask) - count(~covered) = -count(~covered & mask).
    # So delta = - (number of currently uncovered residues covered by new class).
    # If delta < 0, it's an improvement.
    uncovered_count = int(np.count_nonzero(~covered[r::m]))
    return -uncovered_count

def _greedy_solve(divisors, covered_init=None, seed=None):
    """Greedy: repeatedly pick (m,r) maximizing new coverage."""
    if seed is not None:
        random.seed(seed)
    
    if covered_init is None:
        covered = np.zeros(_BASE, dtype=bool)
    else:
        covered = covered_init.copy()
    
    classes = []
    remaining = list(divisors)
    
    # Sort divisors by size? Smaller moduli cover denser sets but have fewer residues to choose.
    # Larger moduli cover sparser sets.
    # Let's just iterate.
    
    while remaining:
        best_gain = -1
        best_m = None
        best_r = None
        
        for m in remaining:
            # Find best residue for this m
            # We want to maximize the number of newly covered residues.
            # This is equivalent to minimizing the number of already covered residues in the class?
            # No, we want to cover as many *uncovered* residues as possible.
            # gain = count(~covered[r::m])
            
            # Vectorized approach for speed:
            # Create a boolean array for each residue? Too slow.
            # Loop over r is O(m). Sum of m over all divisors is manageable if we do it smartly.
            # 51975 is small enough.
            
            best_r_for_m = 0
            best_gain_for_m = -1
            
            # Optimization: If m is large, the class is sparse. 
            # We can check residues in a randomized order or all of them.
            for r in range(m):
                g = int(np.count_nonzero(~covered[r::m]))
                if g > best_gain_for_m:
                    best_gain_for_m = g
                    best_r_for_m = r
            
            if best_gain_for_m > best_gain:
                best_gain = best_gain_for_m
                best_m = m
                best_r = best_r_for_m
        
        if best_m is None or best_gain <= 0:
            break
        
        covered[best_r::best_m] = True
        classes.append([best_m, best_r])
        remaining.remove(best_m)
        
        if time.time() > _DEADLINE - 10:
            break
    
    return classes, covered

def _local_search(classes, covered, divisors, max_iterations=1000):
    """Local search: try swapping residues or replacing moduli."""
    best_classes = [c[:] for c in classes]
    best_covered = covered.copy()
    best_score = _count_uncovered(best_covered)
    
    used_moduli = set(m for m, r in best_classes)
    available_moduli = [m for m in divisors if m not in used_moduli]
    
    for _ in range(max_iterations):
        if time.time() > _DEADLINE - 5:
            break
        
        # Try different move types
        move_type = random.choice(['swap_residue', 'replace_modulus', 'add_modulus'])
        
        new_classes = [c[:] for c in best_classes]
        
        if move_type == 'swap_residue' and len(best_classes) > 0:
            idx = random.randint(0, len(best_classes) - 1)
            m, old_r = best_classes[idx]
            
            # Try to find a better residue
            best_new_r = old_r
            best_delta = 0
            
            # Sample residues to save time, or all if m is small
            if m < 100:
                candidates = range(m)
            else:
                candidates = random.sample(range(m), min(50, m))
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                
                # Calculate delta
                # Gain: newly covered by new_r
                # Loss: newly uncovered because old_r removed
                # We need to know if old_r's coverage was redundant.
                
                # Efficient check:
                # Count uncovered in new_r class: gain
                gain = int(np.count_nonzero(~best_covered[new_r::m]))
                
                # Count residues covered by old_r but NOT by new_r AND NOT by any other class.
                # This is tricky without rebuilding.
                # Approximation: If gain > 0, it's likely good.
                # Better: Compute exact delta.
                
                # Let's do exact delta for the top few candidates
                if gain > 0:
                    # Check loss
                    # Remove old class from coverage
                    # This is expensive. 
                    # Alternative: Just try the move and see if score improves.
                    pass
            
            # Simplified: Try a random new residue and evaluate
            if random.random() < 0.5: # 50% chance to actually try a swap
                new_r = random.randint(0, m - 1)
                if new_r != old_r:
                    new_covered = best_covered.copy()
                    # Remove old
                    # We need to recalculate coverage from scratch or update carefully.
                    # Updating carefully:
                    # 1. Mark old_r::m as "maybe uncovered"
                    # 2. Mark new_r::m as "covered"
                    # This is hard with a single boolean array.
                    
                    # Fast path: Rebuild coverage from new_classes if we commit?
                    # No, let's just do a full evaluation for the candidate move.
                    
                    # Create a temporary coverage array
                    temp_covered = np.zeros(_BASE, dtype=bool)
                    for mm, rr in best_classes:
                        if mm == m:
                            temp_covered[new_r::mm] = True
                        else:
                            temp_covered[rr::mm] = True
                    
                    new_score = _count_uncovered(temp_covered)
                    
                    if new_score < best_score:
                        best_classes[idx] = [m, new_r]
                        best_covered = temp_covered
                        best_score = new_score
        
        elif move_type == 'replace_modulus' and available_moduli and len(best_classes) > 0:
            idx = random.randint(0, len(best_classes) - 1)
            old_m, old_r = best_classes[idx]
            new_m = random.choice(available_moduli)
            
            # Find best residue for new_m
            best_r = 0
            best_gain = -1
            # Sample residues
            if new_m < 100:
                candidates = range(new_m)
            else:
                candidates = random.sample(range(new_m), min(50, new_m))
                
            for r in candidates:
                g = int(np.count_nonzero(~best_covered[r::new_m]))
                if g > best_gain:
                    best_gain = g
                    best_r = r
            
            if best_gain > 0:
                # Evaluate move
                temp_covered = np.zeros(_BASE, dtype=bool)
                for mm, rr in best_classes:
                    if mm == old_m:
                        temp_covered[best_r::new_m] = True
                    else:
                        temp_covered[rr::mm] = True
                
                new_score = _count_uncovered(temp_covered)
                
                if new_score < best_score:
                    best_classes[idx] = [new_m, best_r]
                    best_covered = temp_covered
                    best_score = new_score
                    used_moduli.discard(old_m)
                    used_moduli.add(new_m)
                    available_moduli = [m for m in divisors if m not in used_moduli]
        
        elif move_type == 'add_modulus' and available_moduli:
            new_m = random.choice(available_moduli)
            best_r = 0
            best_gain = -1
            if new_m < 100:
                candidates = range(new_m)
            else:
                candidates = random.sample(range(new_m), min(50, new_m))
                
            for r in candidates:
                g = int(np.count_nonzero(~best_covered[r::new_m]))
                if g > best_gain:
                    best_gain = g
                    best_r = r
            
            if best_gain > 0:
                # Evaluate move
                temp_covered = best_covered.copy()
                temp_covered[best_r::new_m] = True
                new_score = _count_uncovered(temp_covered)
                
                if new_score < best_score:
                    best_classes.append([new_m, best_r])
                    best_covered = temp_covered
                    best_score = new_score
                    used_moduli.add(new_m)
                    available_moduli = [m for m in divisors if m not in used_moduli]
    
    return best_classes, best_covered, best_score

def construct():
    divisors = _divisors()
    random.seed(42)
    
    best_classes = []
    best_covered = np.zeros(_BASE, dtype=bool)
    best_score = _BASE
    
    # Phase 1: Run greedy with different random seeds to get diverse starting points
    for seed in range(50):
        if time.time() > _DEADLINE - 20:
            break
        classes, covered = _greedy_solve(divisors, seed=seed)
        
        # Ensure we have at least one class
        if not classes:
            # Pick a random valid class
            m = random.choice(divisors)
            r = random.randint(0, m - 1)
            classes = [[m, r]]
            covered = np.zeros(_BASE, dtype=bool)
            covered[r::m] = True
            
        score = _count_uncovered(covered)
        if score < best_score:
            best_classes = classes
            best_covered = covered
            best_score = score
    
    # Phase 2: Local search from best greedy solution
    if best_classes:
        if time.time() > _DEADLINE - 10:
            return best_classes
        best_classes, best_covered, best_score = _local_search(
            best_classes, best_covered, divisors, max_iterations=2000
        )
    
    # Phase 3: Try a few more random restarts with local search
    for seed in range(20):
        if time.time() > _DEADLINE - 5:
            break
        classes, covered = _greedy_solve(divisors, seed=100 + seed)
        if not classes:
            m = random.choice(divisors)
            r = random.randint(0, m - 1)
            classes = [[m, r]]
            covered = np.zeros(_BASE, dtype=bool)
            covered[r::m] = True
            
        score = _count_uncovered(covered)
        if score < best_score:
            best_classes = classes
            best_covered = covered
            best_score = score
        
        if time.time() > _DEADLINE - 5:
            break
        classes, covered, score = _local_search(
            classes, covered, divisors, max_iterations=500
        )
        if score < best_score:
            best_classes = classes
            best_covered = covered
            best_score = score
    
    # Final safety check: ensure non-empty
    if not best_classes:
        m = random.choice(divisors)
        r = random.randint(0, m - 1)
        best_classes = [[m, r]]
        
    return best_classes
# EVOLVE-BLOCK-END
