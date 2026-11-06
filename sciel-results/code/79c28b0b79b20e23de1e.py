"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    base = 51975
    deadline = time.time() + 115  # safety margin before 150s limit
    
    # Get all odd divisors of 51975 greater than 1
    divisors = sorted([d for d in range(3, base + 1, 2) if base % d == 0])
    
    def score_classes(classes):
        """Count uncovered residues."""
        covered = bytearray(base)
        for m, r in classes:
            covered[r::m] = b'\x01' * len(range(r, base, m))
        return covered.count(0)
    
    def local_search(classes, max_passes=3):
        """Optimize residues for a fixed set of moduli."""
        classes = [c[:] for c in classes]
        for _ in range(max_passes):
            improved = False
            for i, (m, r) in enumerate(classes):
                # Try each residue and see if it improves coverage
                covered = bytearray(base)
                for j, (m2, r2) in enumerate(classes):
                    if j != i:
                        covered[r2::m2] = b'\x01' * len(range(r2, base, m2))
                best_r = r
                best_count = covered[r::m].count(0)
                for tr in range(m):
                    count = covered[tr::m].count(0)
                    if count > best_count:
                        best_count = count
                        best_r = tr
                if best_r != r:
                    classes[i] = [m, best_r]
                    improved = True
            if not improved:
                break
        return classes
    
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
    
    # Initial greedy with increasing order
    best_classes = greedy(divisors)
    best_score = score_classes(best_classes)
    
    # Local search on initial solution
    improved = local_search(best_classes)
    improved_score = score_classes(improved)
    if improved_score < best_score:
        best_score = improved_score
        best_classes = improved
    
    random.seed(42)
    while time.time() < deadline:
        # Try different strategies
        strategy = random.random()
        
        if strategy < 0.3:
            # Random permutation of all divisors
            order = divisors[:]
            random.shuffle(order)
            classes = greedy(order)
            classes = local_search(classes, max_passes=2)
            score = score_classes(classes)
            if score < best_score:
                best_score = score
                best_classes = classes
                if best_score == 0:
                    break
        elif strategy < 0.6:
            # Random subset of divisors (varying size)
            subset_size = random.randint(len(divisors) // 2, len(divisors))
            subset = random.sample(divisors, subset_size)
            random.shuffle(subset)
            classes = greedy(subset)
            classes = local_search(classes, max_passes=2)
            score = score_classes(classes)
            if score < best_score:
                best_score = score
                best_classes = classes
                if best_score == 0:
                    break
        else:
            # Perturb current best: change one modulus or residue
            classes = [c[:] for c in best_classes]
            if random.random() < 0.5 and len(classes) > 1:
                # Remove a random class
                idx = random.randint(0, len(classes) - 1)
                classes.pop(idx)
                classes = local_search(classes, max_passes=2)
            else:
                # Change residue of a random class
                idx = random.randint(0, len(classes) - 1)
                m = classes[idx][0]
                classes[idx][1] = random.randint(0, m - 1)
                classes = local_search(classes, max_passes=2)
            score = score_classes(classes)
            if score < best_score:
                best_score = score
                best_classes = classes
                if best_score == 0:
                    break
    
    return best_classes
# EVOLVE-BLOCK-END
