"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _score(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    sumset_size = len(sumset)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    log_arg = 2 * max_u + 1
    
    if log_arg <= 1:
        return 0.0
    
    score = 1 + math.log(ratio) / math.log(log_arg)
    return score


def _score_fast(U):
    """Faster score computation for local search."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    sumset_size = len(sumset)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    log_arg = 2 * max_u + 1
    
    if log_arg <= 1:
        return 0.0
    
    score = 1 + math.log(ratio) / math.log(log_arg)
    return score


def _local_search(U, max_time, rng):
    """Perform local search to improve the set U."""
    U = sorted(set(U))
    best_score = _score_fast(U)
    best_U = list(U)
    start_time = time.time()
    
    max_u = max(U)
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Generate candidates for addition
        candidates = set()
        for u in U:
            for delta in range(-5, 6):
                v = u + delta
                if v >= 0 and v not in U:
                    candidates.add(v)
        
        # Larger jumps
        if max_u > 0:
            for _ in range(30):
                v = rng.randint(0, int(max_u * 1.5) + 10)
                if v not in U:
                    candidates.add(v)
        
        # Try additions
        for v in sorted(candidates):
            new_U = sorted(U + [v])
            new_score = _score_fast(new_U)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_U = new_U
                U = new_U
                max_u = max(U)
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            # Try removals
            removed = False
            for idx in range(len(U)):
                if U[idx] == 0:
                    continue
                new_U = U[:idx] + U[idx+1:]
                if len(new_U) < 2:
                    continue
                new_score = _score_fast(new_U)
                if new_score > best_score + 1e-12:
                    best_score = new_score
                    best_U = new_U
                    U = new_U
                    max_u = max(U)
                    removed = True
                    no_improve_count = 0
                    break
            
            if not removed:
                # Try swaps
                swapped = False
                for idx in range(len(U)):
                    if U[idx] == 0:
                        continue
                    U_without = U[:idx] + U[idx+1:]
                    for v in sorted(candidates):
                        if v in U_without:
                            continue
                        new_U = sorted(U_without + [v])
                        new_score = _score_fast(new_U)
                        if new_score > best_score + 1e-12:
                            best_score = new_score
                            best_U = new_U
                            U = new_U
                            max_u = max(U)
                            swapped = True
                            no_improve_count = 0
                            break
                    if swapped:
                        break
                
                if not swapped:
                    no_improve_count += 1
                    if no_improve_count >= 3:
                        break
    
    return best_U, best_score


def _exhaustive_small_search(deadline):
    """Exhaustive search over subsets of small ranges."""
    best_U = [0, 1, 3]
    best_score = _score_fast(best_U)
    
    # Search subsets of {0, 1, ..., 19} of size 3-8
    base_range = list(range(20))
    
    for size in range(3, 9):
        if time.time() > deadline - 5:
            break
        # Use combinations but be selective
        # Focus on sets that include 0
        non_zero = base_range[1:]
        count = 0
        for comb in combinations(non_zero, size - 1):
            if time.time() > deadline - 5:
                break
            U = [0] + list(comb)
            score = _score_fast(U)
            if score > best_score:
                best_score = score
                best_U = list(U)
            count += 1
            if count % 10000 == 0:
                pass
    
    # Also try structured sets
    # Arithmetic progressions
    for start in range(0, 5):
        for step in range(1, 15):
            for length in range(3, 12):
                U = [start + i * step for i in range(length)]
                if 0 not in U:
                    U.insert(0, 0)
                U = sorted(set(U))
                if len(U) < 2:
                    continue
                score = _score_fast(U)
                if score > best_score:
                    best_score = score
                    best_U = list(U)
    
    # Geometric progressions
    for base in range(2, 10):
        for length in range(3, 15):
            U = [0]
            val = 1
            for _ in range(length):
                U.append(val)
                val *= base
            if max(U) > 10000:
                break
            score = _score_fast(U)
            if score > best_score:
                best_score = score
                best_U = list(U)
    
    # Mixed: geometric with perturbations
    for base in range(2, 6):
        for length in range(3, 10):
            U = [0]
            val = 1
            for i in range(length):
                U.append(val)
                val *= base
            # Try perturbing some elements
            for perturb_idx in range(1, len(U)):
                for delta in range(-3, 4):
                    if delta == 0:
                        continue
                    new_U = list(U)
                    new_val = new_U[perturb_idx] + delta
                    if new_val < 0:
                        continue
                    new_U[perturb_idx] = new_val
                    new_U = sorted(set(new_U))
                    if len(new_U) < 2:
                        continue
                    score = _score_fast(new_U)
                    if score > best_score:
                        best_score = score
                        best_U = list(new_U)
    
    return best_U, best_score


def _structured_search(deadline, rng):
    """Search over structured families: APs, GPs, mixed."""
    best_U = [0, 1, 3]
    best_score = _score_fast(best_U)
    
    # Two-segment arithmetic progressions
    for n1 in range(2, 8):
        for n2 in range(2, 8):
            for s1 in range(1, 10):
                for s2 in range(1, 10):
                    if s2 <= s1:
                        continue
                    base1 = 0
                    base2 = n1 * s1
                    U = set()
                    U.add(0)
                    for i in range(n1):
                        U.add(base1 + i * s1)
                    for i in range(n2):
                        U.add(base2 + i * s2)
                    U = sorted(U)
                    if len(U) < 2:
                        continue
                    score = _score_fast(U)
                    if score > best_score:
                        best_score = score
                        best_U = list(U)
    
    # Three-segment
    for n1 in range(2, 5):
        for n2 in range(2, 5):
            for n3 in range(2, 5):
                for s1 in range(1, 5):
                    for s2 in range(s1 + 1, 10):
                        for s3 in range(s2 + 1, 15):
                            base1 = 0
                            base2 = n1 * s1
                            base3 = base2 + n2 * s2
                            U = set()
                            U.add(0)
                            for i in range(n1):
                                U.add(base1 + i * s1)
                            for i in range(n2):
                                U.add(base2 + i * s2)
                            for i in range(n3):
                                U.add(base3 + i * s3)
                            U = sorted(U)
                            if len(U) < 2 or max(U) > 500:
                                continue
                            score = _score_fast(U)
                            if score > best_score:
                                best_score = score
                                best_U = list(U)
    
    # Randomized structured search
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        # Random geometric-like with noise
        n = rng.randint(4, 12)
        base = rng.uniform(1.5, 4.0)
        U = [0]
        val = 1
        for i in range(n - 1):
            noisy_val = int(val * (1 + rng.uniform(-0.2, 0.2)))
            if noisy_val > 0 and noisy_val not in U:
                U.append(noisy_val)
            val *= base
        U = sorted(set(U))
        if len(U) >= 2:
            score = _score_fast(U)
            if score > best_score:
                best_score = score
                best_U = list(U)
        
        # Random two-AP
        n1 = rng.randint(2, 6)
        n2 = rng.randint(2, 6)
        s1 = rng.randint(1, 5)
        s2 = rng.randint(s1 + 1, 15)
        base2 = rng.randint(n1 * s1, n1 * s1 + s2 * 3)
        U = set()
        U.add(0)
        for i in range(n1):
            U.add(i * s1)
        for i in range(n2):
            U.add(base2 + i * s2)
        U = sorted(U)
        if len(U) >= 2 and max(U) < 500:
            score = _score_fast(U)
            if score > best_score:
                best_score = score
                best_U = list(U)
    
    return best_U, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin from 150s limit
    
    rng = random.Random(42)
    
    best_U = [0, 1, 3]
    best_score = _score_fast(best_U)
    
    # Phase 1: Exhaustive small search
    time_left = deadline - time.time()
    if time_left > 10:
        U, score = _exhaustive_small_search(deadline - 5)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Phase 2: Structured search
    time_left = deadline - time.time()
    if time_left > 10:
        U, score = _structured_search(deadline - 5, rng)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Phase 3: Local search from multiple starting points
    starting_sets = [
        list(best_U),
        [0, 1, 3],
        [0, 1, 3, 9],
        [0, 1, 3, 9, 27],
        [0, 2, 4, 8, 16, 32],
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 4, 16, 64],
        [0, 3, 9, 27, 81],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 5, 10, 20, 40],
        [0, 1, 2, 3, 6, 12, 24],
        [0, 1, 2, 5, 10, 21, 42],
    ]
    
    # Add some random starting points
    for _ in range(10):
        size = rng.randint(4, 12)
        max_val = rng.randint(15, 100)
        U = [0]
        while len(U) < size:
            v = rng.randint(1, max_val)
            if v not in U:
                U.append(v)
        starting_sets.append(sorted(U))
    
    for start_set in starting_sets:
        if time.time() > deadline - 15:
            break
        alloc_time = min(deadline - time.time() - 10, 8)
        if alloc_time < 1:
            break
        U, score = _local_search(list(start_set), alloc_time, rng)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Phase 4: Final polish on best
    time_left = deadline - time.time()
    if time_left > 5:
        U, score = _local_search(list(best_U), time_left - 3, rng)
        if score > best_score:
            best_score = score
            best_U = U
    
    return sorted(best_U)

# EVOLVE-BLOCK-END
