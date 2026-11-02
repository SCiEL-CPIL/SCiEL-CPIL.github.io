"""Rewritten program: modular lattice construction for sumset/difference-set exponent."""
import time
import math
import itertools
import random

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    U = list(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute U+U and U-U
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    # The score is 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    # Note: |U-U| can be larger than |U+U| since differences include negatives
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    
    score = 1 + math.log(ratio) / denom
    return score


def _exhaustive_small(max_size=5, max_val=20):
    """Exhaustive search over small sets to find good configurations."""
    best_set = [0]
    best_score = 0.0
    
    # Try all subsets of {0, 1, ..., max_val} of size up to max_size
    # But we need to include 0
    candidates = list(range(1, max_val + 1))
    
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(candidates, size):
            U = [0] + list(combo)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    return best_set, best_score


def _modular_lattice(m, density, max_iter=10000):
    """
    Construct U using a modular lattice approach.
    Select elements that form a dense sublattice in Z_m.
    """
    # Start with 0
    U = {0}
    
    # Try to build a set where differences are large but sums collapse mod m
    # Key insight: if U is contained in a coset of a subgroup of Z_m,
    # then U-U is contained in that subgroup, limiting |U-U|.
    # Instead, we want U-U to be large (spread out) but U+U to be small.
    
    # Alternative: use a geometric progression or sparse set
    # For now, let's try random local search
    
    best = [0]
    best_score = 0.0
    
    random.seed(42)
    
    for attempt in range(max_iter):
        # Randomly add or remove elements
        if len(U) < 50 and random.random() < 0.7:
            # Add a random element
            x = random.randint(1, 2 * m)
            U.add(x)
        elif len(U) > 1:
            # Remove a random non-zero element
            non_zero = [x for x in U if x != 0]
            if non_zero:
                U.remove(random.choice(non_zero))
        
        U_list = sorted(U)
        s = _score_set(U_list)
        if s > best_score:
            best_score = s
            best = U_list[:]
    
    return best, best_score


def _geometric_progression(ratio, terms):
    """Generate a geometric progression set."""
    U = [0]
    val = 1
    for i in range(terms):
        U.append(val)
        val *= ratio
    return U


def _fibonacci_set(n_terms):
    """Generate a set based on Fibonacci numbers."""
    fibs = [1, 1]
    for i in range(n_terms - 2):
        fibs.append(fibs[-1] + fibs[-2])
    return [0] + fibs


def _local_search(start_set, max_iter=5000, max_val=100):
    """Local search to improve a set."""
    U = list(set(start_set))
    if 0 not in U:
        U.append(0)
    U = sorted(U)
    
    best = U[:]
    best_score = _score_set(U)
    
    random.seed(123)
    
    for i in range(max_iter):
        # Try a random perturbation
        new_U = U[:]
        
        # Add, remove, or swap an element
        action = random.random()
        if action < 0.4 and len(new_U) < 30:
            # Add
            x = random.randint(1, max_val)
            if x not in new_U:
                new_U.append(x)
        elif action < 0.7 and len(new_U) > 1:
            # Remove
            non_zero = [x for x in new_U if x != 0]
            if non_zero:
                new_U.remove(random.choice(non_zero))
        else:
            # Swap
            if len(new_U) > 1:
                non_zero = [x for x in new_U if x != 0]
                if non_zero:
                    old = random.choice(non_zero)
                    new_U.remove(old)
                    new_val = random.randint(1, max_val)
                    if new_val not in new_U:
                        new_U.append(new_val)
        
        new_U = sorted(set(new_U))
        if 0 not in new_U:
            new_U.append(0)
        
        s = _score_set(new_U)
        if s > best_score:
            best_score = s
            best = new_U[:]
            U = new_U[:]
    
    return best, best_score


def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Phase 1: Exhaustive search for small sets
    if time.time() < deadline:
        ex_set, ex_score = _exhaustive_small(max_size=5, max_val=25)
        if ex_score > best_score:
            best_score = ex_score
            best_set = ex_set
    
    # Phase 2: Try structured constructions
    structured_candidates = []
    
    # Geometric progressions
    for ratio in range(2, 8):
        for terms in range(3, 15):
            gp = _geometric_progression(ratio, terms)
            structured_candidates.append(gp)
    
    # Fibonacci-like
    for n in range(5, 15):
        structured_candidates.append(_fibonacci_set(n))
    
    # Powers of 2
    for n in range(2, 15):
        structured_candidates.append([0] + [2**i for i in range(n)])
    
    # Sparse arithmetic progressions
    for step in range(2, 10):
        for length in range(3, 12):
            structured_candidates.append([0] + [step * i for i in range(1, length)])
    
    # Mixed constructions
    structured_candidates.append([0, 1, 2, 5, 10, 20, 40])
    structured_candidates.append([0, 1, 3, 7, 15, 31, 63])
    structured_candidates.append([0, 1, 2, 4, 8, 16, 32, 64])
    
    for cand in structured_candidates:
        if time.time() >= deadline:
            break
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_set = cand[:]
    
    # Phase 3: Local search from best candidate
    if time.time() < deadline:
        ls_set, ls_score = _local_search(best_set, max_iter=3000, max_val=50)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    # Phase 4: Modular lattice random search
    if time.time() < deadline:
        for m in [7, 11, 13, 17, 19, 23, 31]:
            if time.time() >= deadline:
                break
            ml_set, ml_score = _modular_lattice(m, 0.5, max_iter=2000)
            if ml_score > best_score:
                best_score = ml_score
                best_set = ml_set
    
    # Phase 5: Final local search polish
    if time.time() < deadline - 10:
        final_set, final_score = _local_search(best_set, max_iter=2000, max_val=80)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
    
    return best_set
# EVOLVE-BLOCK-END
