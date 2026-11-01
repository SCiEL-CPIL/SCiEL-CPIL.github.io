import time
import itertools
import random
import math
import numpy as np

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
    val = len(U_minus) / len(U_plus)
    if val <= 0:
        return 0.0
    return 1.0 + math.log(val) / math.log(2 * max_u + 1)

def _score_fast(U):
    """Fast score computation using numpy for larger sets."""
    U = list(U)
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    n = len(U)
    U_arr = np.array(U, dtype=np.int64)
    # Compute all pairwise sums
    sums = U_arr[:, None] + U_arr[None, :]
    # Compute all pairwise differences
    diffs = U_arr[:, None] - U_arr[None, :]
    n_plus = len(np.unique(sums))
    n_minus = len(np.unique(diffs))
    val = n_minus / n_plus
    if val <= 0:
        return 0.0
    return 1.0 + math.log(val) / math.log(2 * max_u + 1)

def _lfsr_sequence(taps, n_bits, length):
    """Generate a sequence using LFSR with given taps."""
    # Initialize with non-zero state
    state = 1
    seq = []
    for _ in range(length):
        seq.append(state & 1)
        # Compute feedback
        feedback = 0
        for t in taps:
            feedback ^= (state >> t) & 1
        state = (state >> 1) | (feedback << (n_bits - 1))
    return seq

def _de_bruijn_from_lfsr(taps, k, offset=0):
    """Generate positions of 1s from LFSR-based de Bruijn-like sequence."""
    n_bits = k
    # Maximum period for LFSR with k bits is 2^k - 1
    max_period = (1 << n_bits) - 1
    seq = _lfsr_sequence(taps, n_bits, max_period)
    # Find positions of 1s
    positions = [i for i, bit in enumerate(seq) if bit == 1]
    # Apply offset and ensure 0 is in the set
    positions = [(p + offset) for p in positions]
    # Ensure 0 is present
    if 0 not in positions:
        positions.append(0)
    return sorted(set(positions))

def _gen_lfsr_seeds():
    """Generate multiple LFSR-based seeds."""
    seeds = []
    # Try different tap configurations for various k values
    for k in range(4, 9):
        # Generate all possible primitive-like tap combinations
        # For k bits, taps are subsets of {1, ..., k-1}
        all_taps = list(itertools.combinations(range(1, k), 2))
        # Also try 3-tap configurations
        if k >= 4:
            all_taps.extend(itertools.combinations(range(1, k), 3))
        
        for taps in all_taps:
            for offset in range(0, 5):
                try:
                    U = _de_bruijn_from_lfsr(taps, k, offset)
                    if len(U) >= 2 and 0 in U:
                        seeds.append(U)
                except:
                    pass
    return seeds

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
    
    # NEW: LFSR-based de Bruijn seeds
    lfsr_seeds = _gen_lfsr_seeds()
    for U_seed in lfsr_seeds:
        if time.time() > deadline:
            break
        s = _score(U_seed)
        if s > best_score:
            best_score = s
            best_set = U_seed[:]
    
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
                    s = _score_fast(U_new) if len(U_new) > 15 else _score(U_new)
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
                s = _score_fast(U_new) if len(U_new) > 15 else _score(U_new)
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
                        s = _score_fast(U_new) if len(U_new) > 15 else _score(U_new)
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
    
    # NEW: Hill climb from LFSR seeds
    for U_seed in lfsr_seeds[:10]:  # Limit to avoid timeout
        if time.time() > deadline:
            break
        U_hc, score_hc = hill_climb(U_seed, max_iters=3000)
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
                s = _score_fast(U_new) if len(U_new) > 15 else _score(U_new)
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
                s = _score_fast(U_new) if len(U_new) > 15 else _score(U_new)
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
                    s = _score_fast(U_new) if len(U_new) > 15 else _score(U_new)
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
