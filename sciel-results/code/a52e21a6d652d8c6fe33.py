"""Improved program for odd covering systems with moduli dividing 51975."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 110
    
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    best_classes = None
    best_score = base
    
    def greedy(ordering):
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m in ordering:
            reshaped = covered.reshape(m, base // m)
            uncovered_counts = (base // m) - reshaped.sum(axis=1)
            best_r = int(np.argmax(uncovered_counts))
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes, int(np.count_nonzero(covered == 0))
    
    def local_search(classes):
        n = len(classes)
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        current_score = int(np.count_nonzero(covered == 0))
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                m, r = classes[i]
                other = np.zeros(base, dtype=np.uint8)
                for j in range(n):
                    if j != i:
                        m2, r2 = classes[j]
                        other[r2::m2] = 1
                
                reshaped = other.reshape(m, base // m)
                uncovered_counts = (base // m) - reshaped.sum(axis=1)
                best_r = int(np.argmax(uncovered_counts))
                
                if best_r != r:
                    new_cov = other.copy()
                    new_cov[best_r::m] = 1
                    new_score = int(np.count_nonzero(new_cov == 0))
                    if new_score < current_score:
                        classes[i] = [m, best_r]
                        current_score = new_score
                        covered = np.zeros(base, dtype=np.uint8)
                        for m2, r2 in classes:
                            covered[r2::m2] = 1
                        improved = True
        return classes, current_score
    
    orderings = [
        sorted(divisors),
        sorted(divisors, reverse=True),
        sorted(divisors, key=lambda m: -base // m),
        sorted(divisors, key=lambda m: base // m),
    ]
    rng = np.random.default_rng(42)
    for _ in range(10):
        s = divisors.copy()
        rng.shuffle(s)
        orderings.append(s)
    
    for ordering in orderings:
        if time.time() > deadline - 30:
            break
        classes, score = greedy(ordering)
        if score < best_score:
            best_score = score
            best_classes = [list(c) for c in classes]
        if time.time() < deadline - 15:
            cls_ls, score_ls = local_search([list(c) for c in classes])
            if score_ls < best_score:
                best_score = score_ls
                best_classes = cls_ls
    
    return best_classes
# EVOLVE-BLOCK-END
