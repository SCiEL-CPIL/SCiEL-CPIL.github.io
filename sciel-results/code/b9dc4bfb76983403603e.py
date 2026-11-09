import math
import time
import itertools
import random

def _score(U):
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
    n_plus = len(plus)
    n_minus = len(minus)
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _lacunary(n, base):
    """Lacunary sequence: powers of base, truncated to n elements."""
    U = [0]
    for i in range(1, n):
        U.append(base ** i)
    return U

def _geometric(n, ratio):
    """Geometric progression with floating ratio, rounded to integers."""
    U = [0]
    for i in range(1, n):
        U.append(int(round(ratio ** i)))
    # Ensure uniqueness and non-negative
    seen = set()
    result = [0]
    for x in U[1:]:
        if x > 0 and x not in seen:
            seen.add(x)
            result.append(x)
    return sorted(result)

def _random_set(size, max_val, rng):
    """Generate a random set of given size containing 0."""
    if size <= 1:
        return [0]
    candidates = list(range(1, max_val + 1))
    chosen = rng.sample(candidates, min(size - 1, len(candidates)))
    return sorted([0] + chosen)

def _local_search(U, max_val, max_steps, rng, deadline):
    """Local search: try adding, removing, or replacing elements."""
    U = list(U)
    current_score = _score(U)
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        
        # Generate a random mutation
        mutation_type = rng.random()
        
        if mutation_type < 0.4:
            # Try adding a new element
            if len(U) < 20:
                new_el = rng.randint(1, max_val)
                if new_el not in U:
                    new_U = sorted(U + [new_el])
                    new_score = _score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
        elif mutation_type < 0.7:
            # Try removing an element (keep 0)
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    remove_idx = rng.choice(removable)
                    new_U = [x for x in U if x != remove_idx]
                    new_score = _score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
        else:
            # Try replacing a non-zero element
            if len(U) > 2:
                non_zero = [x for x in U if x != 0]
                if non_zero:
                    old_el = rng.choice(non_zero)
                    new_el = rng.randint(1, max_val)
                    if new_el not in U:
                        new_U = sorted([x for x in U if x != old_el] + [new_el])
                        new_score = _score(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
    
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = random.Random(42)
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Phase 1: Exhaustive search for small sets
    for size in range(2, 7):
        for combo in itertools.combinations(range(1, 30), size - 1):
            if time.time() > deadline:
                return best_set
            U = sorted([0] + list(combo))
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Phase 2: Structured families
    for base in range(2, 15):
        for n in range(3, 12):
            if time.time() > deadline:
                return best_set
            U = _lacunary(n, base)
            if max(U) > 100000:
                continue
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    for ratio in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0]:
        for n in range(3, 15):
            if time.time() > deadline:
                return best_set
            U = _geometric(n, ratio)
            if len(U) < 2 or max(U) > 100000:
                continue
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 3: Random sets + local search
    max_val = 200
    for _ in range(50):
        if time.time() > deadline:
            break
        
        size = rng.randint(3, 15)
        U = _random_set(size, max_val, rng)
        s = _score(U)
        
        # Local search from this starting point
        U_opt, s_opt = _local_search(U, max_val, 100, rng, deadline)
        if s_opt > best_score:
            best_score = s_opt
            best_set = list(U_opt)
    
    # Phase 4: Local search from best found so far
    if time.time() < deadline - 10:
        U_opt, s_opt = _local_search(best_set, max_val, 500, rng, deadline)
        if s_opt > best_score:
            best_score = s_opt
            best_set = list(U_opt)
    
    # Ensure 0 is in the set and return
    if 0 not in best_set:
        best_set = sorted([0] + best_set)
    
    return best_set
# EVOLVE-BLOCK-END
