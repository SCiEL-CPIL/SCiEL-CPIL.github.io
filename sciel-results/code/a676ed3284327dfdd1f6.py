"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the score for a set U (must contain 0)."""
    if 0 not in U:
        return -1.0
    U = sorted(set(U))
    if len(U) < 2:
        return -1.0
    M = max(U)
    if M == 0:
        return -1.0
    # Compute U-U and U+U
    diffs = set()
    sums = set()
    for a in U:
        for b in U:
            diffs.add(a - b)
            sums.add(a + b)
    ratio = len(diffs) / len(sums)
    if ratio <= 0:
        return -1.0
    import math
    val = 1.0 + math.log(ratio) / math.log(2 * M + 1)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Try small exhaustive searches
    for max_val in range(1, 15):
        if time.time() > deadline:
            break
        # Try all subsets of {0, 1, ..., max_val} containing 0
        # For small max_val, this is feasible
        elements = list(range(1, max_val + 1))
        n = len(elements)
        # Only try subsets up to size ~10 for feasibility
        if n > 12:
            break
        # Try all subsets of size 2 to min(n, 8)
        for size in range(2, min(n, 8) + 1):
            if time.time() > deadline:
                break
            for subset in itertools.combinations(elements, size):
                U = [0] + list(subset)
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Try structured families: geometric progressions
    if time.time() < deadline:
        for base in range(2, 50):
            for k in range(2, 12):
                if time.time() > deadline:
                    break
                U = [0] + [base ** i for i in range(1, k + 1)]
                M = max(U)
                if M > 10**7:
                    break
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Try mixed-base constructions: {0, 1, B, B^2, ..., B^k} with various B
    if time.time() < deadline:
        for B in range(2, 30):
            for k in range(2, 10):
                if time.time() > deadline:
                    break
                U = [0, 1] + [B ** i for i in range(2, k + 1)]
                M = max(U)
                if M > 10**7:
                    break
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Local search: try adding or removing elements from best_set
    if time.time() < deadline:
        current = best_set[:]
        current_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding elements
            max_elem = max(current)
            # Try adding elements up to 2*max_elem + some
            candidates_to_add = []
            for x in range(1, min(2 * max_elem + 1, max_elem + 200)):
                if x not in current:
                    candidates_to_add.append(x)
            
            # Also try larger jumps
            for x in range(max_elem + 1, max_elem + 100):
                candidates_to_add.append(x)
            
            for x in candidates_to_add:
                if time.time() > deadline:
                    break
                new_set = current + [x]
                s = _score_set(new_set)
                if s > current_score + 1e-9:
                    current = new_set
                    current_score = s
                    improved = True
                    break
            
            # Try removing elements (except 0)
            if not improved:
                for i, x in enumerate(current):
                    if x == 0:
                        continue
                    if time.time() > deadline:
                        break
                    new_set = current[:i] + current[i+1:]
                    s = _score_set(new_set)
                    if s > current_score + 1e-9:
                        current = new_set
                        current_score = s
                        improved = True
                        break
        
        if current_score > best_score:
            best_score = current_score
            best_set = current[:]
    
    # Try a few more structured approaches: arithmetic progressions
    if time.time() < deadline:
        for start in range(1, 10):
            for step in range(1, 30):
                for length in range(2, 15):
                    if time.time() > deadline:
                        break
                    U = [0] + [start + i * step for i in range(length)]
                    M = max(U)
                    if M > 10**6:
                        break
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U[:]
    
    # Try "sparse" sets: {0} + sparse set
    if time.time() < deadline:
        # Use a set with large gaps
        for n in range(3, 12):
            if time.time() > deadline:
                break
            # Try random sparse sets
            for trial in range(50):
                if time.time() > deadline:
                    break
                U = [0]
                max_so_far = 0
                for i in range(n - 1):
                    # Choose next element with some randomness
                    min_next = max_so_far + 1
                    max_next = max_so_far * 2 + 10 if max_so_far > 0 else 10
                    x = min_next + (trial * (i + 1) * 7 + i * 13) % max(1, max_next - min_next)
                    x = min(x, max_next)
                    if x not in U:
                        U.append(x)
                        max_so_far = max(max_so_far, x)
                U.sort()
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    return best_set
# EVOLVE-BLOCK-END
