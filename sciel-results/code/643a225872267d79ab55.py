"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    base = 51975  # 3^3 * 5^2 * 7 * 11

    # Generate all odd divisors of base greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # increasing order

    def score(classes):
        """Compute number of uncovered residues modulo base."""
        covered = np.zeros(base, dtype=bool)
        for m, r in classes:
            covered[r::m] = True
        return base - np.sum(covered)

    def greedy_construct(order=None):
        """Greedy: moduli in specified order, each takes the class covering most uncovered residues."""
        if order is None:
            order = divisors
        covered = np.zeros(base, dtype=bool)
        classes = []
        for m in order:
            # Find r that maximizes coverage of currently uncovered residues
            # We can vectorize the counting of uncovered residues for each r
            # counts[r] = sum of not covered[r::m]
            # This is O(B) per m
            best_r = 0
            best_count = -1
            # Optimization: If m is large, the number of residues is small.
            # If m is small, the number of residues is large.
            # We just loop over r.
            for r in range(m):
                count = np.sum(~covered[r::m])
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = True
        return classes, score(classes)

    def local_search(classes, max_iter=100):
        """Local search: try swapping residues for each modulus to improve score."""
        classes = [list(c) for c in classes]
        current_score = score(classes)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for i, (m, r) in enumerate(classes):
                # Compute covered without this class
                # This is O(K * B) where K is number of classes
                covered_without = np.zeros(base, dtype=bool)
                for j, (mj, rj) in enumerate(classes):
                    if j != i:
                        covered_without[rj::mj] = True
                
                # Find best residue for m given covered_without
                # We want to maximize the number of NEW residues covered
                # i.e., minimize the number of residues in covered_without[r::m]
                # Or maximize sum of ~covered_without[r::m]
                best_r = r
                best_new_covers = -1
                
                for r_new in range(m):
                    # Count how many of these are NOT in covered_without
                    new_covers = np.sum(~covered_without[r_new::m])
                    if new_covers > best_new_covers:
                        best_new_covers = new_covers
                        best_r = r_new
                
                if best_r != r:
                    classes[i][1] = best_r
                    # Update score efficiently?
                    # The change in score is - (new_covers - old_covers)
                    # But we need to know old_covers.
                    # It's safer to just recalculate score or track it.
                    # Recalculating is O(K*B). K is ~100. B is 50k. 5M ops. Fast in numpy.
                    current_score = score(classes)
                    improved = True
            
            if not improved:
                break
        
        return classes, current_score

    # Initial greedy
    classes, current_score = greedy_construct()
    best_classes = [list(c) for c in classes]
    best_score = current_score

    # Local search
    classes, current_score = local_search(classes, max_iter=50)
    if current_score < best_score:
        best_classes = [list(c) for c in classes]
        best_score = current_score

    # Try multiple random restarts with perturbed greedy
    # Use remaining time for restarts
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Perturbed greedy: randomize the order of moduli
        random_divs = list(divisors)
        random.shuffle(random_divs)
        # Keep small moduli first? Or just shuffle?
        # Shuffling often helps escape local minima.
        # Let's try a mix: sort by random weight but keep small ones somewhat prioritized?
        # Actually, just shuffle is a good restart strategy.
        
        classes, current_score = greedy_construct(order=random_divs)
        
        # Local search
        classes, current_score = local_search(classes, max_iter=20)
        
        if current_score < best_score:
            best_classes = [list(c) for c in classes]
            best_score = current_score

    return best_classes
# EVOLVE-BLOCK-END
