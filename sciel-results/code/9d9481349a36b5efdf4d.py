import time
import random
import numpy as np
from itertools import combinations

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# Divisors > 1
# We will precompute all divisors
def get_divisors(n):
    divs = []
    for i in range(2, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs

DIVISORS = get_divisors(BASE)
# Filter to odd divisors > 1 (all divisors are odd since base is odd)
# Sort by size, maybe smaller first for greedy? Or larger first?
# Smaller moduli cover more residues.
# Let's sort ascending.
DIVISORS.sort()

def count_uncovered(covered, m, r):
    # covered is a boolean array of length BASE
    # Returns number of uncovered residues in class r mod m
    # i.e., count of i in [0, BASE) such that i % m == r and covered[i] == 0
    # This is equivalent to sum of ~covered[r::m]
    # Using numpy for speed
    return np.sum(~covered[r::m])

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Initial greedy solution
    covered = np.zeros(BASE, dtype=bool)
    classes = []
    # Greedy: pick modulus and residue that covers most uncovered
    # To speed up, we can iterate moduli in some order.
    # Smaller moduli are more "efficient" (cover more per class).
    # Let's try a greedy that picks the best (m, r) pair globally at each step?
    # That might be expensive. 
    # Alternative: fix an order of moduli, and for each pick best r.
    # Let's try sorting divisors by size (ascending).
    
    # Precompute indices for each modulus?
    # For m, indices are r, r+m, ...
    
    # Let's do a simple greedy first:
    # Sort divisors by m (ascending).
    # For each m, find r that maximizes newly covered residues.
    # Note: "newly covered" = uncovered before this step.
    
    # However, the order matters. Maybe we should try different orders?
    # Or use local search on the greedy result.
    
    # 1. Initial Greedy
    covered = np.zeros(BASE, dtype=bool)
    current_classes = []
    # Use all divisors? Or a subset?
    # Using all divisors might be overkill if smaller ones cover everything.
    # But we can only use each modulus once.
    # Let's iterate through divisors sorted by size.
    
    # Optimization: Precompute which residues are covered by which modulus?
    # Too much memory.
    
    # Let's just do the greedy step by step.
    # To make it faster, we can restrict to a subset of divisors if time is tight, 
    # but let's try all first.
    
    # Sort divisors by size
    divs = DIVISORS[:]
    # Maybe prioritize smaller moduli? Yes, they have more residues per class.
    
    for m in divs:
        if time.time() > deadline - 5:
            break
            
        # Find best residue r for this m
        # We want to maximize count of uncovered[i] where i % m == r
        # This is max over r of sum(uncovered[r::m])
        
        # Efficient computation:
        # We can sum uncovered by taking the sum of slices.
        # But doing this for all r (0 to m-1) for every m might be slow.
        # Total work sum(m) for all m.
        # Sum of divisors of 51975.
        # 51975 is not huge.
        
        best_r = 0
        best_count = -1
        
        # Vectorized approach:
        # Reshape covered into shape (BASE // m, m) if possible?
        # BASE is divisible by m.
        # covered.reshape(-1, m) gives rows of length m.
        # Sum along axis 0 gives count of True for each residue.
        # We want sum of False, which is (BASE // m) - sum(True).
        
        # Let's try this.
        try:
            reshaped = covered.reshape(-1, m)
            # Sum of covered (True) for each residue
            covered_counts = np.sum(reshaped, axis=0)
            total_in_class = BASE // m
            uncovered_counts = total_in_class - covered_counts
            
            best_r = np.argmax(uncovered_counts)
            best_count = uncovered_counts[best_r]
        except:
            # Fallback to loop if reshape fails (shouldn't happen)
            for r in range(m):
                c = np.sum(~covered[r::m])
                if c > best_count:
                    best_count = c
                    best_r = r
        
        if best_count > 0:
            # Apply this class
            covered[best_r::m] = True
            current_classes.append([m, int(best_r)])
        else:
            # No more uncovered?
            # We can break if all covered
            if np.all(covered):
                break
            # If best_count is 0, adding this class doesn't help.
            # But maybe a larger modulus later could? 
            # Unlikely if smaller ones failed to cover remaining.
            # Actually, if a small modulus covers 0 new, it means its classes are full.
            # It's possible that a larger modulus covers something else.
            # So we shouldn't break, just skip.
            pass
    
    # Check if complete
    if np.all(covered):
        return current_classes
    
    # Local Search / Hill Climbing
    # We have a set of classes.
    # We can try to:
    # 1. Remove a class and see if score increases (worsens) or stays same? 
    #    We want to minimize uncovered.
    #    If removing a class doesn't change uncovered count, remove it (simpler).
    #    If removing a class increases uncovered, keep it.
    # 2. Replace a class (m, r) with (m, r') to cover more.
    # 3. Add a new class from remaining divisors.
    # 4. Swap two classes?
    
    # Let's define a function to evaluate a set of classes.
    def evaluate(classes):
        cov = np.zeros(BASE, dtype=bool)
        for m, r in classes:
            cov[r::m] = True
        return np.sum(~cov)
    
    current_score = np.sum(~covered)
    
    # Improvement loop
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # 1. Try removing classes
        # If a class is redundant, remove it.
        for i in range(len(current_classes)):
            if time.time() > deadline - 2:
                break
            m, r = current_classes[i]
            # Temporarily remove
            # Check if any residue in this class is NOT covered by other classes.
            # This is expensive to check directly.
            # Alternative: Recalculate score without this class.
            # Or: Check if covered[r::m] is subset of union of others.
            
            # Let's just recalculate score for simplicity if N is small enough.
            # But N=51975. Recalculating is O(N * num_classes).
            # Might be too slow for many iterations.
            
            # Faster check:
            # For each residue x in r::m, is it covered by another class?
            # We can precompute a "coverage count" array.
            # count[i] = number of classes covering i.
            
            # Let's maintain a coverage count array.
            # But we don't have it in the current structure easily.
            
            # Let's stick to simple local moves:
            # Change residue r for a fixed m.
            for idx, (m, r) in enumerate(current_classes):
                if time.time() > deadline - 2:
                    break
                # Try changing r to r'
                # We want to maximize new coverage.
                # Current coverage of this class: covered_by_this = (covered & (indices in r::m))
                # Actually, we can just evaluate the change.
                
                # What if we change r to r'?
                # New covered set = (Old covered - old class) + new class
                # Score change = (uncovered in old class not covered by others) - (uncovered in new class not covered by others)
                
                # Let's compute "essential" coverage.
                # essential[i] = 1 if i is covered ONLY by this class.
                
                # This is getting complex. Let's try a simpler heuristic:
                # Randomly perturb and accept if better.
                
                # Let's try: Pick a random class, change its residue to a random other residue.
                # Or pick a random class, remove it.
                # Or pick a random unused modulus, add it with best residue.
                
                # Let's implement "Best Residue" for each existing class again?
                # If we re-optimize residues for current moduli, does it help?
                # Yes, if the greedy order was suboptimal.
                
                # Let's try re-optimizing residues for the current set of moduli.
                # This is like a coordinate descent.
                
                # Sort classes by m?
                # For each class, find best r given the coverage of OTHER classes.
                
                # Compute coverage by others
                # cov_others = np.zeros(BASE, dtype=bool)
                # for j, (mj, rj) in enumerate(current_classes):
                #     if j != idx:
                #         cov_others[rj::mj] = True
                
                # This is O(K * N) per step. K is ~20-30. N=51975.
                # 30 * 51975 ~ 1.5M ops. Fast enough in numpy.
                
                # Let's do this for all classes once per iteration.
                
                # We'll break out of inner loop and do a full pass.
                break # Break inner loop to do full pass
        
        # Full pass: Re-optimize residues for current moduli
        # We need to know which moduli are used.
        used_moduli = [m for m, r in current_classes]
        
        # Calculate coverage by all classes except the one being optimized?
        # Or just calculate total coverage and see if changing r improves it?
        # If we change r for class i, the new score is:
        # Score = Uncovered( (C \ C_i) U C_i' )
        # = Uncovered(C \ C_i) + Uncovered(C_i' \ (C \ C_i))
        # = Uncovered(C \ C_i) + Uncovered(C_i' \ C_others)
        
        # Let's compute C_others (coverage by all other classes).
        # Then for each class i, find r' that minimizes Uncovered(C_i' \ C_others).
        # Which is equivalent to maximizing Covered(C_i' \ C_others).
        
        # Compute C_others
        # This is expensive if done from scratch for each i.
        # But we can compute total coverage, then subtract? No, boolean OR is not invertible.
        
        # Let's just recompute C_others for each i in the pass.
        # K is small.
        
        for idx in range(len(current_classes)):
            if time.time() > deadline - 2:
                break
            
            m, r = current_classes[idx]
            
            # Compute coverage by others
            cov_others = np.zeros(BASE, dtype=bool)
            for j in range(len(current_classes)):
                if j == idx:
                    continue
                mj, rj = current_classes[j]
                cov_others[rj::mj] = True
            
            # Find best r' for m
            # Maximize count of i in r'::m such that ~cov_others[i] is True
            # i.e., maximize sum(~cov_others[r'::m])
            
            try:
                reshaped = (~cov_others).reshape(-1, m)
                counts = np.sum(reshaped, axis=0)
                best_r = np.argmax(counts)
                best_gain = counts[best_r]
            except:
                best_r = r
                best_gain = 0
            
            # Check if this improves total score
            # Old contribution of this class to coverage (excluding overlaps with others? No, score is global)
            # Let's just calculate the new score if we swap r with best_r.
            
            if best_r != r:
                # Calculate new score
                # We can simulate:
                # new_cov = cov_others.copy()
                # new_cov[best_r::m] = True
                # new_score = np.sum(~new_cov)
                
                # Old score:
                # We have `covered` which is the current total coverage.
                # But `covered` includes this class.
                # We don't have `cov_others` stored as `covered` minus this class easily.
                
                # Let's just compute the score difference directly.
                # Score = sum(~cov)
                # Change: replace class (m, r) with (m, best_r).
                # Residues affected: those in r::m or best_r::m.
                
                # Let's just recalculate the full score for this candidate.
                # It's O(N).
                
                new_classes = current_classes[:]
                new_classes[idx] = [m, int(best_r)]
                
                # Evaluate
                cov_test = np.zeros(BASE, dtype=bool)
                for mm, rr in new_classes:
                    cov_test[rr::mm] = True
                new_score = np.sum(~cov_test)
                
                if new_score < current_score:
                    current_score = new_score
                    current_classes = new_classes
                    covered = cov_test
                    improved = True
                    break # Restart pass? Or continue?
                    # If we improved, the `cov_others` for subsequent indices is invalid.
                    # So break and restart the while loop or the for loop.
                    # Let's break the for loop.
        
        # Try adding a new class
        if not improved:
            # Find unused moduli
            used_set = set(used_moduli)
            unused = [m for m in DIVISORS if m not in used_set]
            
            if unused:
                # Pick best unused modulus
                # For each unused m, find best r that maximizes gain.
                # Gain = sum(~covered[r::m])
                
                best_m = None
                best_r = None
                best_gain = 0
                
                for m in unused:
                    if time.time() > deadline - 2:
                        break
                    try:
                        reshaped = (~covered).reshape(-1, m)
                        counts = np.sum(reshaped, axis=0)
                        max_gain = np.max(counts)
                        if max_gain > best_gain:
                            best_gain = max_gain
                            best_m = m
                            best_r = np.argmax(counts)
                    except:
                        pass
                
                if best_m is not None and best_gain > 0:
                    current_classes.append([best_m, int(best_r)])
                    covered[best_r::best_m] = True
                    current_score -= best_gain # Approximation, need exact recalc?
                    # Exact recalc:
                    # current_score = np.sum(~covered)
                    improved = True
    
    return current_classes

# Pre-compute divisors outside construct to avoid repeated work?
# Yes, DIVISORS is global.

# Let's refine the local search.
# The "re-optimize residues" step is good.
# The "add class" step is good.
# The "remove class" step is missing.
# If we have redundant classes, removing them doesn't change score but simplifies.
# More importantly, if a class is "bad" (covers things already covered), removing it might allow better placement of others?
# No, coverage is monotonic. Adding classes only helps (or is neutral).
# So we should never remove a class if it adds unique coverage.
# If it adds NO unique coverage, removing it is safe.
# Let's add a cleanup step.

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    covered = np.zeros(BASE, dtype=bool)
    current_classes = []
    
    # 1. Initial Greedy
    divs = DIVISORS[:]
    for m in divs:
        if time.time() > deadline - 10:
            break
        
        try:
            reshaped = (~covered).reshape(-1, m)
            counts = np.sum(reshaped, axis=0)
            best_r = np.argmax(counts)
            best_gain = counts[best_r]
        except:
            continue
        
        if best_gain > 0:
            covered[best_r::m] = True
            current_classes.append([m, int(best_r)])
    
    current_score = np.sum(~covered)
    
    if current_score == 0:
        return current_classes
    
    # 2. Local Search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # A. Re-optimize residues
        # Try changing residue for each existing class
        for idx in range(len(current_classes)):
            if time.time() > deadline - 5:
                break
            
            m, r = current_classes[idx]
            
            # Compute coverage by others
            cov_others = np.zeros(BASE, dtype=bool)
            for j in range(len(current_classes)):
                if j == idx:
                    continue
                mj, rj = current_classes[j]
                cov_others[rj::mj] = True
            
            # Find best r'
            try:
                reshaped = (~cov_others).reshape(-1, m)
                counts = np.sum(reshaped, axis=0)
                best_r = np.argmax(counts)
                best_gain = counts[best_r]
            except:
                continue
            
            if best_r != r:
                # Check if it actually improves global score
                # Simulate
                new_classes = current_classes[:]
                new_classes[idx] = [m, int(best_r)]
                
                cov_test = np.zeros(BASE, dtype=bool)
                for mm, rr in new_classes:
                    cov_test[rr::mm] = True
                new_score = np.sum(~cov_test)
                
                if new_score < current_score:
                    current_score = new_score
                    current_classes = new_classes
                    covered = cov_test
                    improved = True
                    break # Restart loop
        
        # B. Add a new class
        if not improved:
            used_set = set(m for m, r in current_classes)
            unused = [m for m in DIVISORS if m not in used_set]
            
            if unused:
                best_m = None
                best_r = None
                best_gain = 0
                
                for m in unused:
                    if time.time() > deadline - 5:
                        break
                    try:
                        reshaped = (~covered).reshape(-1, m)
                        counts = np.sum(reshaped, axis=0)
                        max_gain = np.max(counts)
                        if max_gain > best_gain:
                            best_gain = max_gain
                            best_m = m
                            best_r = np.argmax(counts)
                    except:
                        pass
                
                if best_m is not None and best_gain > 0:
                    current_classes.append([best_m, int(best_r)])
                    covered[best_r::best_m] = True
                    current_score = np.sum(~covered)
                    improved = True
        
        # C. Remove redundant classes
        # If a class covers no unique residues, remove it.
        if not improved:
            for idx in range(len(current_classes)):
                if time.time() > deadline - 5:
                    break
                
                m, r = current_classes[idx]
                
                # Check if any residue in r::m is covered ONLY by this class.
                # i.e., is there an i in r::m such that ~cov_others[i] is True?
                
                cov_others = np.zeros(BASE, dtype=bool)
                for j in range(len(current_classes)):
                    if j == idx:
                        continue
                    mj, rj = current_classes[j]
                    cov_others[rj::mj] = True
                
                # Check if class is redundant
                # If sum(~cov_others[r::m]) == 0, it is redundant.
                # Actually, if it covers something not covered by others, it's not redundant.
                # If it covers ONLY things covered by others, it is redundant.
                
                unique_covered = np.sum((~cov_others) & (np.arange(BASE) % m == r))
                
                if unique_covered == 0:
                    # Remove it
                    current_classes.pop(idx)
                    # Recalculate covered
                    covered = np.zeros(BASE, dtype=bool)
                    for mm, rr in current_classes:
                        covered[rr::mm] = True
                    current_score = np.sum(~covered)
                    improved = True
                    break # Restart
        
    return current_classes

# Ensure DIVISORS is computed
DIVISORS = get_divisors(BASE)
DIVISORS.sort()

# Double check get_divisors
def get_divisors(n):
    divs = []
    for i in range(2, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs

# One more thing: The problem asks for distinct moduli.
# My code ensures this by tracking used_set.

# Let's verify the score calculation.
# Score = number of residues modulo 51975 covered by no class.
# My `current_score` is `np.sum(~covered)`.
# `covered` is a boolean array of length 51975.
# This is correct.

# Potential issue: `reshape(-1, m)` requires BASE to be divisible by m.
# Since m is a divisor of BASE, this is always true.

# Let's add a small randomization to break ties or explore?
# Greedy is deterministic. Local search is deterministic.
# Maybe add a few random restarts?
# Given the time limit, one run with local search should be enough.
# But if we get stuck in a local minimum, randomization might help.
# Let's try a few random restarts if time permits.

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_classes = None
    best_score = BASE
    
    # Run a few iterations with different random seeds/orders?
    # Or just one solid run.
    # Let's do one solid run with the logic above.
    
    covered = np.zeros(BASE, dtype=bool)
    current_classes = []
    
    # 1. Initial Greedy
    # Randomize order of divisors?
    # Smaller divisors are better.
    # Let's stick to sorted.
    
    divs = DIVISORS[:]
    
    for m in divs:
        if time.time() > deadline - 20:
            break
        
        try:
            reshaped = (~covered).reshape(-1, m)
            counts = np.sum(reshaped, axis=0)
            best_r = np.argmax(counts)
            best_gain = counts[best_r]
        except:
            continue
        
        if best_gain > 0:
            covered[best_r::m] = True
            current_classes.append([m, int(best_r)])
    
    current_score = np.sum(~covered)
    
    if current_score < best_score:
        best_score = current_score
        best_classes = current_classes[:]
    
    if current_score == 0:
        return current_classes
    
    # 2. Local Search
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # A. Re-optimize residues
        for idx in range(len(current_classes)):
            if time.time() > deadline - 10:
                break
            
            m, r = current_classes[idx]
            
            cov_others = np.zeros(BASE, dtype=bool)
            for j in range(len(current_classes)):
                if j == idx:
                    continue
                mj, rj = current_classes[j]
                cov_others[rj::mj] = True
            
            try:
                reshaped = (~cov_others).reshape(-1, m)
                counts = np.sum(reshaped, axis=0)
                best_r = np.argmax(counts)
            except:
                continue
            
            if best_r != r:
                new_classes = current_classes[:]
                new_classes[idx] = [m, int(best_r)]
                
                cov_test = np.zeros(BASE, dtype=bool)
                for mm, rr in new_classes:
                    cov_test[rr::mm] = True
                new_score = np.sum(~cov_test)
                
                if new_score < current_score:
                    current_score = new_score
                    current_classes = new_classes
                    covered = cov_test
                    improved = True
                    break
        
        # B. Add a new class
        if not improved:
            used_set = set(m for m, r in current_classes)
            unused = [m for m in DIVISORS if m not in used_set]
            
            if unused:
                best_m = None
                best_r = None
                best_gain = 0
                
                for m in unused:
                    if time.time() > deadline - 10:
                        break
                    try:
                        reshaped = (~covered).reshape(-1, m)
                        counts = np.sum(reshaped, axis=0)
                        max_gain = np.max(counts)
                        if max_gain > best_gain:
                            best_gain = max_gain
                            best_m = m
                            best_r = np.argmax(counts)
                    except:
                        pass
                
                if best_m is not None and best_gain > 0:
                    current_classes.append([best_m, int(best_r)])
                    covered[best_r::best_m] = True
                    current_score = np.sum(~covered)
                    improved = True
        
        # C. Remove redundant classes
        if not improved:
            for idx in range(len(current_classes)):
                if time.time() > deadline - 10:
                    break
                
                m, r = current_classes[idx]
                
                cov_others = np.zeros(BASE, dtype=bool)
                for j in range(len(current_classes)):
                    if j == idx:
                        continue
                    mj, rj = current_classes[j]
                    cov_others[rj::mj] = True
                
                # Check if class is redundant
                # unique_covered = np.sum((~cov_others) & (np.arange(BASE) % m == r))
                # Faster:
                # indices = np.arange(r, BASE, m)
                # unique_covered = np.sum(~cov_others[indices])
                
                indices = np.arange(r, BASE, m)
                unique_covered = np.sum(~cov_others[indices])
                
                if unique_covered == 0:
                    current_classes.pop(idx)
                    covered = np.zeros(BASE, dtype=bool)
                    for mm, rr in current_classes:
                        covered[rr::mm] = True
                    current_score = np.sum(~covered)
                    improved = True
                    break
        
        if current_score < best_score:
            best_score = current_score
            best_classes = current_classes[:]
        
        if best_score == 0:
            break
    
    return best_classes

# Final check on imports and structure.
import time
import numpy as np

BASE = 51975

def get_divisors(n):
    divs = []
    for i in range(2, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs

DIVISORS = get_divisors(BASE)
DIVISORS.sort()

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_classes = []
    best_score = BASE
    
    covered = np.zeros(BASE, dtype=bool)
    current_classes = []
    
    # 1. Initial Greedy
    for m in DIVISORS:
        if time.time() > deadline - 20:
            break
        
        try:
            reshaped = (~covered).reshape(-1, m)
            counts = np.sum(reshaped, axis=0)
            best_r = np.argmax(counts)
            best_gain = counts[best_r]
        except:
            continue
        
        if best_gain > 0:
            covered[best_r::m] = True
            current_classes.append([m, int(best_r)])
    
    current_score = np.sum(~covered)
    
    if current_score < best_score:
        best_score = current_score
        best_classes = current_classes[:]
    
    if current_score == 0:
        return current_classes
    
    # 2. Local Search
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # A. Re-optimize residues
        for idx in range(len(current_classes)):
            if time.time() > deadline - 10:
                break
            
            m, r = current_classes[idx]
            
            cov_others = np.zeros(BASE, dtype=bool)
            for j in range(len(current_classes)):
                if j == idx:
                    continue
                mj, rj = current_classes[j]
                cov_others[rj::mj] = True
            
            try:
                reshaped = (~cov_others).reshape(-1, m)
                counts = np.sum(reshaped, axis=0)
                best_r = np.argmax(counts)
            except:
                continue
            
            if best_r != r:
                new_classes = current_classes[:]
                new_classes[idx] = [m, int(best_r)]
                
                cov_test = np.zeros(BASE, dtype=bool)
                for mm, rr in new_classes:
                    cov_test[rr::mm] = True
                new_score = np.sum(~cov_test)
                
                if new_score < current_score:
                    current_score = new_score
                    current_classes = new_classes
                    covered = cov_test
                    improved = True
                    break
        
        # B. Add a new class
        if not improved:
            used_set = set(m for m, r in current_classes)
            unused = [m for m in DIVISORS if m not in used_set]
            
            if unused:
                best_m = None
                best_r = None
                best_gain = 0
                
                for m in unused:
                    if time.time() > deadline - 10:
                        break
                    try:
                        reshaped = (~covered).reshape(-1, m)
                        counts = np.sum(reshaped, axis=0)
                        max_gain = np.max(counts)
                        if max_gain > best_gain:
                            best_gain = max_gain
                            best_m = m
                            best_r = np.argmax(counts)
                    except:
                        pass
                
                if best_m is not None and best_gain > 0:
                    current_classes.append([best_m, int(best_r)])
                    covered[best_r::best_m] = True
                    current_score = np.sum(~covered)
                    improved = True
        
        # C. Remove redundant classes
        if not improved:
            for idx in range(len(current_classes)):
                if time.time() > deadline - 10:
                    break
                
                m, r = current_classes[idx]
                
                cov_others = np.zeros(BASE, dtype=bool)
                for j in range(len(current_classes)):
                    if j == idx:
                        continue
                    mj, rj = current_classes[j]
                    cov_others[rj::mj] = True
                
                indices = np.arange(r, BASE, m)
                unique_covered = np.sum(~cov_others[indices])
                
                if unique_covered == 0:
                    current_classes.pop(idx)
                    covered = np.zeros(BASE, dtype=bool)
                    for mm, rr in current_classes:
                        covered[rr::mm] = True
                    current_score = np.sum(~covered)
                    improved = True
                    break
        
        if current_score < best_score:
            best_score = current_score
            best_classes = current_classes[:]
        
        if best_score == 0:
            break
    
    return best_classes
