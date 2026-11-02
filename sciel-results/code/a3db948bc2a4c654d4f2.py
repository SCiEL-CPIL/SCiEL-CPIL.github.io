"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import itertools
from collections import defaultdict

_start_time = time.time()
_DEADLINE = 110  # safety margin before 120s limit

def _elapsed():
    return time.time() - _start_time

def _score_set(U):
    """Compute the score for a given set U."""
    if not U or 0 not in U:
        return 0
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return 0
    # Compute U+U and U-U
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    num_sums = len(sum_set)
    num_diffs = len(diff_set)
    if num_sums == 0:
        return 0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return 1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return 1.0
    return 1.0 + math.log(ratio) / log_denom

def _try_geometric(base, n):
    """Try geometric progression {0, 1, base, base^2, ..., base^(n-1)}."""
    U = [0]
    val = 1
    for i in range(1, n):
        U.append(val)
        val *= base
    return U

def _local_search(initial, max_steps=5000):
    """Local search to improve a set."""
    best_U = list(initial)
    best_score = _score_set(best_U)
    current_U = list(best_U)
    current_score = best_score
    
    for step in range(max_steps):
        if _elapsed() > _DEADLINE:
            break
        # Try adding a new element
        max_val = max(current_U) if current_U else 0
        # Candidate new values
        candidates = []
        for delta in range(1, 50):
            candidates.append(max_val + delta)
        for delta in range(1, 30):
            candidates.append(max_val * 2 + delta)
        # Also try smaller gaps
        for v in range(1, max_val + 1):
            if v not in current_U:
                candidates.append(v)
        
        random.shuffle(candidates)
        
        improved = False
        for new_val in candidates[:20]:
            if new_val <= 0:
                continue
            new_U = list(current_U) + [new_val]
            new_U.sort()
            new_score = _score_set(new_U)
            if new_score > current_score + 1e-10:
                current_U = new_U
                current_score = new_score
                if current_score > best_score:
                    best_U = list(current_U)
                    best_score = current_score
                improved = True
                break
        
        if not improved:
            # Try removing an element
            if len(current_U) > 2:
                remove_candidates = [i for i in range(1, len(current_U))]
                random.shuffle(remove_candidates)
                for idx in remove_candidates[:10]:
                    new_U = [x for i, x in enumerate(current_U) if i != idx]
                    new_score = _score_set(new_U)
                    if new_score > current_score + 1e-10:
                        current_U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_U = list(current_U)
                            best_score = current_score
                        improved = True
                        break
        
        if not improved and step > 100:
            # Restart with perturbation
            if random.random() < 0.1:
                # Try a fresh geometric progression
                base = random.choice([2, 3, 4, 5])
                n = random.randint(3, 8)
                current_U = _try_geometric(base, n)
                current_score = _score_set(current_U)
            else:
                break
    
    return best_U, best_score

def construct():
    global _start_time
    _start_time = time.time()
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try various geometric progressions
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if _elapsed() > _DEADLINE - 30:
            break
        for n in range(3, 12):
            if _elapsed() > _DEADLINE - 30:
                break
            U = _try_geometric(base, n)
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Local search from best
    if _elapsed() < _DEADLINE - 10:
        U, score = _local_search(best_U, max_steps=3000)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Try more local searches from different starting points
    for base in [2, 3, 4]:
        if _elapsed() > _DEADLINE - 10:
            break
        for n in range(4, 8):
            if _elapsed() > _DEADLINE - 10:
                break
            U = _try_geometric(base, n)
            U2, score2 = _local_search(U, max_steps=1000)
            if score2 > best_score:
                best_score = score2
                best_U = U2
    
    return sorted(best_U)
# EVOLVE-BLOCK-END
