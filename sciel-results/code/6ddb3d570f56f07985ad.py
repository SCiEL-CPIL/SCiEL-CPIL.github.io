import time
import random

def construct():
    deadline = time.time() + 80
    n = 12
    
    # Start with powers of 2
    elements = [1 << i for i in range(n)]
    
    def get_subset_sums(elements):
        sums = {0}
        for e in elements:
            sums = sums | {s + e for s in sums}
        return sums
    
    def is_valid(elements):
        sums = get_subset_sums(elements)
        return len(sums) == (1 << n)
    
    # Local search: try to reduce the max element
    best = elements[:]
    best_max = max(best)
    
    while time.time() < deadline:
        improved = False
        # Try to reduce the largest element
        max_idx = best.index(max(best))
        for new_val in range(1, max(best)):
            if time.time() > deadline:
                break
            if new_val in best:
                continue
            candidate = best[:]
            candidate[max_idx] = new_val
            if is_valid(candidate):
                best = candidate
                best_max = max(best)
                improved = True
                break
        if not improved:
            # Try swapping elements
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    candidate = best[:]
                    candidate[i], candidate[j] = best[j], best[i]
                    if is_valid(candidate):
                        if max(candidate) < best_max:
                            best = candidate
                            best_max = max(best)
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            # Try more aggressive changes
            for i in range(n):
                if time.time() > deadline:
                    break
                for new_val in range(1, best_max):
                    if time.time() > deadline:
                        break
                    if new_val in best:
                        continue
                    candidate = best[:]
                    candidate[i] = new_val
                    if is_valid(candidate):
                        if max(candidate) < best_max:
                            best = candidate
                            best_max = max(best)
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            break
    
    return best
