"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin
    
    # Greedy approach: try to add moduli in increasing order
    # For each modulus, find a residue that is disjoint with all existing classes
    # We want to maximize sum of 1/m, so prefer smaller moduli
    
    def is_disjoint(m, r, classes):
        for m2, r2 in classes:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    # Start with a good initial solution
    classes = [[20, 0]]
    
    # Try to add more classes greedily
    for m in range(21, 201):
        if time.time() > deadline:
            break
        
        # Try to find a valid residue for this modulus
        # We can use a randomized search with restarts
        found = False
        
        # First try a quick deterministic scan
        for r in range(m):
            if is_disjoint(m, r, classes):
                classes.append([m, r])
                found = True
                break
        
        if not found:
            # Try randomized search
            for _ in range(100):
                if time.time() > deadline:
                    break
                r = random.randint(0, m - 1)
                if is_disjoint(m, r, classes):
                    classes.append([m, r])
                    found = True
                    break
    
    # Now try to improve by removing some classes and re-adding different ones
    # This is a local search: try to replace a class with a better one
    
    best_score = sum(1.0 / m for m, r in classes)
    best_classes = [c[:] for c in classes]
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Try to add a random modulus
        if len(classes) < 181:  # max possible is 181 (200-20+1)
            m = random.randint(20, 200)
            if not any(c[0] == m for c in classes):
                # Try to find a valid residue
                for _ in range(50):
                    r = random.randint(0, m - 1)
                    if is_disjoint(m, r, classes):
                        new_classes = classes + [[m, r]]
                        new_score = sum(1.0 / mm for mm, rr in new_classes)
                        if new_score > best_score:
                            best_score = new_score
                            best_classes = [c[:] for c in new_classes]
                            classes = new_classes
                        break
        
        # Try to remove a random class and re-add it with a different residue
        if classes:
            idx = random.randint(0, len(classes) - 1)
            m, r = classes[idx]
            for _ in range(50):
                new_r = random.randint(0, m - 1)
                if new_r != r and is_disjoint(m, new_r, [c for i, c in enumerate(classes) if i != idx]):
                    new_classes = [c[:] for c in classes]
                    new_classes[idx] = [m, new_r]
                    new_score = sum(1.0 / mm for mm, rr in new_classes)
                    if new_score > best_score:
                        best_score = new_score
                        best_classes = [c[:] for c in new_classes]
                        classes = new_classes
                    break
    
    return best_classes
# EVOLVE-BLOCK-END
