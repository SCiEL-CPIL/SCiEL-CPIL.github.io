import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    base = 51975
    
    # Get all odd divisors > 1 of 51975
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Sort divisors
    divisors.sort()
    
    def count_uncovered(covered):
        return sum(1 for x in covered if x == 0)
    
    def greedy_initial(order):
        covered = bytearray(base)
        classes = []
        for m in order:
            best_r = 0
            best_count = -1
            for r in range(m):
                # Count uncovered in this residue class
                cnt = 0
                for i in range(r, base, m):
                    if covered[i] == 0:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            for i in range(best_r, base, m):
                covered[i] = 1
        return classes, covered, count_uncovered(covered)
    
    # Try multiple greedy orderings
    best_classes = None
    best_uncovered = base
    best_covered = None
    
    # Standard increasing order
    classes, covered, unc = greedy_initial(divisors)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = classes
        best_covered = covered
    
    # Try decreasing order
    if time.time() < deadline:
        classes, covered, unc = greedy_initial(sorted(divisors, reverse=True))
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = classes
            best_covered = covered
    
    # Try a few random orderings
    random.seed(42)
    for attempt in range(5):
        if time.time() >= deadline:
            break
        random_order = divisors.copy()
        random.shuffle(random_order)
        classes, covered, unc = greedy_initial(random_order)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = classes
            best_covered = covered
    
    # Local search improvement
    if best_classes is not None and time.time() < deadline:
        classes = [c[:] for c in best_classes]
        covered = bytearray(best_covered)
        uncovered = best_uncovered
        
        # Build a set of moduli currently used
        used_moduli = set(m for m, r in classes)
        available_moduli = [m for m in divisors if m not in used_moduli]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try changing residue for each existing class
            for idx, (m, r) in enumerate(classes):
                if time.time() >= deadline:
                    break
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Calculate change in uncovered count
                    # Remove old coverage, add new coverage
                    old_uncovered_gain = 0
                    new_uncovered_gain = 0
                    for i in range(r, base, m):
                        if covered[i] == 1:
                            # This residue would become uncovered if no other class covers it
                            # But we need to check if another class covers it
                            pass
                    
                    # Simpler: just try it
                    # Save state
                    old_covered = bytearray(covered)
                    old_r = r
                    
                    # Remove old coverage
                    for i in range(r, base, m):
                        # Check if any other class covers i
                        other_covers = False
                        for j, (m2, r2) in enumerate(classes):
                            if j == idx:
                                continue
                            if (i - r2) % m2 == 0:
                                other_covers = True
                                break
                        if not other_covers:
                            covered[i] = 0
                    
                    # Add new coverage
                    for i in range(new_r, base, m):
                        covered[i] = 1
                    
                    new_unc = count_uncovered(covered)
                    if new_unc < uncovered:
                        uncovered = new_unc
                        classes[idx][1] = new_r
                        improved = True
                    else:
                        # Revert
                        covered = old_covered
                        classes[idx][1] = old_r
            
            # Try adding a new modulus
            if time.time() < deadline:
                best_add = None
                best_add_unc = uncovered
                for m in available_moduli:
                    if time.time() >= deadline:
                        break
                    for r in range(m):
                        # Temporarily add this class
                        old_covered = bytearray(covered)
                        for i in range(r, base, m):
                            covered[i] = 1
                        new_unc = count_uncovered(covered)
                        if new_unc < best_add_unc:
                            best_add_unc = new_unc
                            best_add = (m, r, bytearray(covered))
                        covered = old_covered
                    if best_add and best_add[0] == m:
                        # Found improvement with this m
                        break
                if best_add:
                    m, r, new_covered = best_add
                    classes.append([m, r])
                    covered = new_covered
                    uncovered = best_add_unc
                    used_moduli.add(m)
                    available_moduli.remove(m)
                    improved = True
            
            # Try removing a modulus
            if time.time() < deadline:
                for idx in range(len(classes)):
                    if time.time() >= deadline:
                        break
                    m, r = classes[idx]
                    old_covered = bytearray(covered)
                    # Remove this class
                    for i in range(r, base, m):
                        other_covers = False
                        for j, (m2, r2) in enumerate(classes):
                            if j == idx:
                                continue
                            if (i - r2) % m2 == 0:
                                other_covers = True
                                break
                        if not other_covers:
                            covered[i] = 0
                    new_unc = count_uncovered(covered)
                    if new_unc <= uncovered:
                        # Removing doesn't hurt (or helps if ties and we want fewer classes)
                        if new_unc < uncovered:
                            classes.pop(idx)
                            covered = bytearray(covered)  # keep as is
                            uncovered = new_unc
                            used_moduli.discard(m)
                            if m in available_moduli:
                                pass
                            else:
                                available_moduli.append(m)
                            improved = True
                    else:
                        covered = old_covered
    
    if best_classes is None:
        # Fallback
        best_classes, best_covered, best_uncovered = greedy_initial(divisors)
    
    return best_classes
