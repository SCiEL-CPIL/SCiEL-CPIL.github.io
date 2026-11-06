"""Combined approach: greedy with multiple orders + local search for odd covering."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    base = 51975
    deadline = time.time() + 110  # safety margin before 120s limit (evaluator kills at 150s)
    
    # Get all odd divisors of 51975 greater than 1
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    def score_classes(classes):
        """Count uncovered residues."""
        covered = bytearray(base)
        for m, r in classes:
            covered[r::m] = b'\x01' * len(range(r, base, m))
        return covered.count(0)
    
    def greedy(order):
        """Greedy construction with given modulus order."""
        covered = bytearray(base)
        classes = []
        for m in order:
            # Find residue that covers most uncovered residues
            best_r = 0
            best_count = -1
            for r in range(m):
                count = covered[r::m].count(0)
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = b'\x01' * len(range(best_r, base, m))
        return classes
    
    def local_search(classes, covered):
        """Try to improve by changing individual residues."""
        best_classes = [list(c) for c in classes]
        best_score = score_classes(best_classes)
        improved = True
        passes = 0
        max_passes = 50
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            for i in range(len(best_classes)):
                m, r = best_classes[i]
                # Compute covered without class i using numpy for speed
                # Build mask of all other classes
                covered_wo_i = np.zeros(base, dtype=np.uint8)
                for j in range(len(best_classes)):
                    if j == i:
                        continue
                    mj, rj = best_classes[j]
                    covered_wo_i[rj::mj] = 1
                base_uncov_wo_i = base - int(np.sum(covered_wo_i))
                
                best_r = r
                best_uncov = base_uncov_wo_i
                for new_r in range(m):
                    newly_covered = int(np.sum(covered_wo_i[new_r::m] == 0))
                    uncov = base_uncov_wo_i - newly_covered
                    if uncov < best_uncov:
                        best_uncov = uncov
                        best_r = new_r
                
                if best_r != r:
                    best_classes[i][1] = best_r
                    best_score = best_uncov
                    improved = True
                    if best_score == 0:
                        return best_classes, best_score
        return best_classes, best_score
    
    # Initial greedy with increasing order
    best_classes = greedy(divisors)
    best_score = score_classes(best_classes)
    
    # Try decreasing order
    classes_dec = greedy(divisors[::-1])
    score_dec = score_classes(classes_dec)
    if score_dec < best_score:
        best_score = score_dec
        best_classes = classes_dec
    
    # Try several random orders
    random.seed(42)
    num_random_tries = 0
    while time.time() < deadline and num_random_tries < 200:
        order = divisors[:]
        random.shuffle(order)
        classes = greedy(order)
        score = score_classes(classes)
        if score < best_score:
            best_score = score
            best_classes = classes
            if best_score == 0:
                break
        num_random_tries += 1
    
    # Local search on the best found
    if time.time() < deadline - 5:
        best_classes, best_score = local_search(best_classes, None)
    
    return best_classes
# EVOLVE-BLOCK-END
