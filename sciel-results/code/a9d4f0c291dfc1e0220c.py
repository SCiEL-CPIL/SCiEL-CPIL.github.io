import time
import itertools
import random

def _score(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    U = set(U)
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    U_plus = {a + b for a in U for b in U}
    U_minus = {a - b for a in U for b in U}
    import math
    val = len(U_minus) / len(U_plus)
    if val <= 0:
        return 0.0
    return 1.0 + math.log(val) / math.log(2 * max_u + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin before 120s limit
    
    best_set = [0, 1, 3]
    best_score = _score(best_set)
    
    # Try parametric families
    # Family 1: {0, a, b, c, ...} small sets
    # Try all sets of size up to 6 with max element up to 20
    for n in range(2, 7):
        if time.time() > deadline:
            break
        # Generate candidate sets containing 0
        # Use combinations from 1..20
        for combo in itertools.combinations(range(1, 21), n - 1):
            if time.time() > deadline:
                break
            U = [0] + list(combo)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Family 2: Arithmetic progressions with perturbations
    # Family 3: Powers-based sets
    for base in range(2, 10):
        for k in range(2, 8):
            if time.time() > deadline:
                break
            U = [base ** i for i in range(k)]
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Hill climbing on best set found
    def hill_climb(U, max_iters=5000):
        U = list(U)
        current_score = _score(U)
        max_u = max(U)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try adding a new element
            for new_el in range(1, max_u + 5):
                if new_el not in U:
                    U_new = U + [new_el]
                    s = _score(U_new)
                    if s > current_score:
                        U = U_new
                        current_score = s
                        improved = True
                        max_u = max(U)
                        break
            if improved:
                continue
            # Try removing an element (except 0)
            for i, el in enumerate(U):
                if el == 0:
                    continue
                U_new = U[:i] + U[i+1:]
                if len(U_new) < 2:
                    continue
                s = _score(U_new)
                if s > current_score:
                    U = U_new
                    current_score = s
                    improved = True
                    max_u = max(U)
                    break
            if improved:
                continue
            # Try replacing an element
            for i, el in enumerate(U):
                if el == 0:
                    continue
                for new_el in range(1, max_u + 5):
                    if new_el not in U or new_el == el:
                        U_new = U[:]
                        U_new[i] = new_el
                        s = _score(U_new)
                        if s > current_score:
                            U = U_new
                            current_score = s
                            improved = True
                            max_u = max(U)
                            break
                if improved:
                    break
            if not improved:
                break
        return U, current_score
    
    # Hill climb from best found
    U_hc, score_hc = hill_climb(best_set)
    if score_hc > best_score:
        best_set = U_hc
        best_score = score_hc
    
    # Try multiple random starts with hill climbing
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        # Random set containing 0
        size = random.randint(3, 12)
        max_el = random.randint(5, 50)
        U_rand = [0] + random.sample(range(1, max_el + 1), min(size - 1, max_el))
        U_hc, score_hc = hill_climb(U_rand, max_iters=2000)
        if score_hc > best_score:
            best_set = U_hc
            best_score = score_hc
    
    # Final polish: try small modifications
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        max_u = max(best_set)
        # Try adding
        for new_el in range(1, max_u + 3):
            if new_el not in best_set:
                U_new = best_set + [new_el]
                s = _score(U_new)
                if s > best_score:
                    best_set = U_new
                    best_score = s
                    improved = True
                    break
        if improved:
            continue
        # Try removing
        for i, el in enumerate(best_set):
            if el == 0:
                continue
            U_new = best_set[:i] + best_set[i+1:]
            if len(U_new) >= 2:
                s = _score(U_new)
                if s > best_score:
                    best_set = U_new
                    best_score = s
                    improved = True
                    break
        if improved:
            continue
        # Try replacing
        for i, el in enumerate(best_set):
            if el == 0:
                continue
            for new_el in range(1, max_u + 3):
                if new_el not in best_set:
                    U_new = best_set[:]
                    U_new[i] = new_el
                    s = _score(U_new)
                    if s > best_score:
                        best_set = U_new
                        best_score = s
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return sorted(best_set)
