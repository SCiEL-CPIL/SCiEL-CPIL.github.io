import numpy as np
import time

def construct():
    base = 51975
    
    # Get all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort divisors in a specific order for greedy
    # Prefer larger moduli first (they cover fewer residues, so more selective)
    # Actually, for coverage, smaller moduli cover more residues
    # Let's try both orders and pick the best
    
    def compute_coverage(classes):
        """Return boolean array of which residues are covered"""
        covered = np.zeros(base, dtype=np.int8)
        for m, r in classes:
            covered[np.arange(r, base, m)] = 1
        return covered
    
    def compute_score(classes):
        covered = compute_coverage(classes)
        return int(np.sum(covered == 0))
    
    # Greedy: add moduli one at a time, picking the residue that covers most new residues
    def greedy(order):
        covered = np.zeros(base, dtype=np.int8)
        classes = []
        for m in order:
            best_r = 0
            best_count = -1
            for r in range(m):
                indices = np.arange(r, base, m)
                count = np.sum(covered[indices] == 0)
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            indices = np.arange(best_r, base, m)
            covered[indices] = 1
        return classes
    
    # Try greedy with different orders
    order1 = sorted(divisors, key=lambda x: x)  # ascending
    order2 = sorted(divisors, key=lambda x: -x)  # descending
    order3 = sorted(divisors, key=lambda x: base // x)  # by number of residues covered
    
    best_classes = None
    best_score = float('inf')
    
    for order in [order1, order2, order3]:
        classes = greedy(order)
        score = compute_score(classes)
        if score < best_score:
            best_score = score
            best_classes = [list(c) for c in classes]
    
    # Local search: try changing residues
    classes = [list(c) for c in best_classes]
    improved = True
    max_iterations = 20
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(len(classes)):
            m, r = classes[i]
            best_r = r
            best_local_score = best_score
            for r_new in range(m):
                if r_new == r:
                    continue
                classes[i][1] = r_new
                new_score = compute_score(classes)
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_r = r_new
            if best_r != r:
                classes[i][1] = best_r
                best_score = best_local_score
                improved = True
            else:
                classes[i][1] = r
    
    # Local search: try swapping moduli
    # Try adding a new modulus and removing another
    improved = True
    max_iterations2 = 10
    iteration2 = 0
    while improved and iteration2 < max_iterations2:
        improved = False
        iteration2 += 1
        current_moduli = set(m for m, r in classes)
        
        # Try swapping one modulus for another
        for i in range(len(classes)):
            m_old = classes[i][0]
            # Try replacing m_old with a different modulus
            for m_new in divisors:
                if m_new in current_moduli:
                    continue
                if m_new == m_old:
                    continue
                
                # Try all residues for m_new
                best_r = 0
                best_local_score = best_score
                for r_new in range(m_new):
                    classes[i][0] = m_new
                    classes[i][1] = r_new
                    new_score = compute_score(classes)
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_r = r_new
                
                # Restore if no improvement
                classes[i][0] = m_old
                if best_local_score < best_score:
                    classes[i][0] = m_new
                    classes[i][1] = best_r
                    best_score = best_local_score
                    improved = True
                else:
                    classes[i][1] = 0  # will be fixed by next residue search
    
    # Final residue optimization
    improved = True
    max_iterations3 = 15
    iteration3 = 0
    while improved and iteration3 < max_iterations3:
        improved = False
        iteration3 += 1
        for i in range(len(classes)):
            m, r = classes[i]
            best_r = r
            best_local_score = best_score
            for r_new in range(m):
                if r_new == r:
                    continue
                classes[i][1] = r_new
                new_score = compute_score(classes)
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_r = r_new
            if best_r != r:
                classes[i][1] = best_r
                best_score = best_local_score
                improved = True
            else:
                classes[i][1] = r
    
    return classes
