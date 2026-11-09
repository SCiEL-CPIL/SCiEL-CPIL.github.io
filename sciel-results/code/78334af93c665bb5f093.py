<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Greedy: moduli in increasing order, each takes the class covering most uncovered residues.
    base = 51975
    covered = bytearray(base)
    classes = []
    for m in [d for d in range(3, base + 1, 2) if base % d == 0]:
        best = max(range(m), key=lambda r: covered[r::m].count(0))
        classes.append([m, best])
        covered[best::m] = b'\x01' * len(range(best, base, m))
    return classes
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _get_divisors(n):
    """Get all odd divisors > 1 of n."""
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

def _score_classes(classes, base=51975):
    """Count uncovered residues modulo base."""
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.sum(covered == 0))

def _greedy_construct(base=51975):
    """Greedy: moduli in increasing order, each takes the class covering most uncovered residues."""
    divisors = _get_divisors(base)
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    for m in divisors:
        # Count uncovered residues for each residue class
        best_r = 0
        best_count = -1
        for r in range(m):
            count = np.sum(covered[r::m] == 0)
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes

def _local_search(classes, base=51975, max_iter=200):
    """Local search: try swapping residues of existing classes."""
    divisors = _get_divisors(base)
    used_moduli = set(m for m, r in classes)
    current_score = _score_classes(classes, base)
    best_classes = [list(c) for c in classes]
    best_score = current_score
    
    for iteration in range(max_iter):
        improved = False
        for i, (m, r) in enumerate(best_classes):
            for new_r in range(m):
                if new_r == r:
                    continue
                # Try this swap
                test_classes = [list(c) for c in best_classes]
                test_classes[i] = [m, new_r]
                test_score = _score_classes(test_classes, base)
                if test_score < best_score:
                    best_classes = test_classes
                    best_score = test_score
                    improved = True
                    break
        if not improved:
            break
    return best_classes, best_score

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with greedy
    classes = _greedy_construct(base)
    best_classes = [list(c) for c in classes]
    best_score = _score_classes(best_classes, base)
    
    # Local search
    best_classes, best_score = _local_search(best_classes, base, max_iter=50)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_classes
    
    # Multi-start: try different greedy orderings
    divisors = _get_divisors(base)
    import random
    rng = random.Random(42)
    
    while time.time() - start_time < deadline - 5:
        # Shuffle divisors and do greedy
        shuffled_divs = divisors[:]
        rng.shuffle(shuffled_divs)
        
        covered = np.zeros(base, dtype=np.uint8)
        greedy_classes = []
        for m in shuffled_divs:
            best_r = 0
            best_count = -1
            for r in range(m):
                count = np.sum(covered[r::m] == 0)
                if count > best_count:
                    best_count = count
                    best_r = r
            greedy_classes.append([m, best_r])
            covered[best_r::m] = 1
        
        # Local search on this result
        improved_classes, improved_score = _local_search(greedy_classes, base, max_iter=30)
        
        if improved_score < best_score:
            best_score = improved_score
            best_classes = improved_classes
    
    return best_classes
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
