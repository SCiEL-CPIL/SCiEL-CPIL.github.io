"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    base = 51975
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Get all odd divisors of 51975 greater than 1
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
    
    # Initial greedy with increasing order
    best_classes = greedy(divisors)
    best_score = score_classes(best_classes)
    
    # Try different orders and local search
    random.seed(42)
    while time.time() < deadline:
        # Generate a random permutation of divisors
        order = divisors[:]
        random.shuffle(order)
        classes = greedy(order)
        score = score_classes(classes)
        if score < best_score:
            best_score = score
            best_classes = classes
            if best_score == 0:
                break
    
    return best_classes
# EVOLVE-BLOCK-END
