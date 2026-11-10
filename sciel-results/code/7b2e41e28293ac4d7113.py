"""Optimized solver for no-dividing-3-sum task."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_MAX = 48
_DEADLINE = time.time() + 85.0  # Safety margin before 90s kill


def _precompute_pair_sums(S_list):
    """Precompute all pair sums and their indices for fast validity checks."""
    n = len(S_list)
    pair_sums = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_sums.append((S_list[i] + S_list[j], i, j))
    return pair_sums


def _is_valid_addition(S, x):
    """Check if adding x to valid set S keeps it valid.
    Only need to check:
    1. x does not divide sum of any 3 in S
    2. No a in S divides sum of x + 2 others from S
    """
    S_list = list(S)
    n = len(S_list)
    if n < 3:
        # Check 2 only: for each a in S, a does not divide x + 2 others
        # If n < 2, no pairs exist, so trivially valid
        if n >= 2:
            for i, a in enumerate(S_list):
                others = [S_list[j] for j in range(n) if j != i]
                if len(others) >= 2:
                    for t in combinations(others, 2):
                        if (x + t[0] + t[1]) % a == 0:
                            return False
        return True
    
    # Precompute pair sums
    pair_sums = _precompute_pair_sums(S_list)
    
    # Check 1: x does not divide sum of any 3 in S
    # Sum of 3 = S[i] + S[j] + S[k] = S[i] + (S[j] + S[k])
    # So check if any (pair_sum + remaining) % x == 0
    for k in range(n):
        rem = S_list[k]
        for ps, i, j in pair_sums:
            if i == k or j == k:
                continue
            if (ps + rem) % x == 0:
                return False
    
    # Check 2: for each a in S, a does not divide sum of x + 2 others from S
    for i, a in enumerate(S_list):
        for ps, p, q in pair_sums:
            if p == i or q == i:
                continue
            if (x + ps) % a == 0:
                return False
    
    return True


def _is_valid_swap(S, remove_val, add_val):
    """Check if swapping remove_val with add_val in valid set S keeps it valid."""
    if add_val in S:
        return False
    if remove_val not in S:
        return False
    
    S_without = [v for v in S if v != remove_val]
    n = len(S_without)
    
    if n < 3:
        if n >= 2:
            for i, a in enumerate(S_without):
                others = [S_without[j] for j in range(n) if j != i]
                if len(others) >= 2:
                    for t in combinations(others, 2):
                        if (add_val + t[0] + t[1]) % a == 0:
                            return False
        return True
    
    pair_sums = _precompute_pair_sums(S_without)
    
    # Check: add_val does not divide sum of any 3 in S_without
    for k in range(n):
        rem = S_without[k]
        for ps, i, j in pair_sums:
            if i == k or j == k:
                continue
            if (ps + rem) % add_val == 0:
                return False
    
    # Check: for each a in S_without, a does not divide sum of add_val + 2 others
    for i, a in enumerate(S_without):
        for ps, p, q in pair_sums:
            if p == i or q == i:
                continue
            if (add_val + ps) % a == 0:
                return False
    
    return True


def _greedy(descending=True):
    """Greedy construction."""
    selected = []
    if descending:
        order = range(_N, 0, -1)
    else:
        order = range(1, _N + 1)
    for x in order:
        if time.time() > _DEADLINE:
            break
        if _is_valid_addition(selected, x):
            selected.append(x)
    return selected


def _local_search(S, time_budget=30.0):
    """Local search with swaps and insertions to improve set size."""
    S = list(S)
    start_time = time.time()
    best = list(S)
    best_len = len(S)
    
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget and time.time() < _DEADLINE:
        improved = False
        
        # Try to add a random element not in S
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            for x in candidates[:100]:
                if time.time() > _DEADLINE:
                    return best
                if _is_valid_addition(S, x):
                    S.append(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # Try swaps: remove one from S, add one not in S
        if len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            
            for a in to_remove[:60]:
                if time.time() > _DEADLINE:
                    return best
                for x in candidates[:60]:
                    if time.time() > _DEADLINE:
                        return best
                    if _is_valid_swap(S, a, x):
                        S = [v for v in S if v != a] + [x]
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = list(S)
                        break
                if improved:
                    break
        
        # If no improvement, perturb
        if not improved:
            if len(S) > 3:
                to_remove = random.sample(S, min(2, len(S) - 2))
                S = [v for v in S if v not in to_remove]
    
    return best


def _multi_start_greedy(num_starts=30, time_budget=20.0):
    """Try multiple greedy orders with randomization."""
    best = []
    start_time = time.time()
    
    for _ in range(num_starts):
        if time.time() - start_time > time_budget or time.time() > _DEADLINE:
            break
        
        # Randomized greedy with different strategies
        strategy = random.random()
        order = list(range(1, _N + 1))
        
        if strategy < 0.3:
            # Bias toward larger values
            weights = [random.random() * 0.5 * _N + v * 1.5 for v in order]
            order.sort(key=lambda v: -weights[v - 1])
        elif strategy < 0.6:
            # Bias toward smaller values
            weights = [random.random() * 0.5 * _N + (_N + 1 - v) * 1.5 for v in order]
            order.sort(key=lambda v: -weights[v - 1])
        else:
            # Fully random
            random.shuffle(order)
        
        selected = []
        for x in order:
            if time.time() > _DEADLINE:
                break
            if _is_valid_addition(selected, x):
                selected.append(x)
        
        if len(selected) > len(best):
            best = list(selected)
    
    return best


def _hill_climbing(S, time_budget=20.0):
    """Hill climbing: try to increase size by removing minimal elements."""
    S = list(S)
    start_time = time.time()
    best = list(S)
    best_len = len(S)
    
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget and time.time() < _DEADLINE:
        improved = False
        
        # Try all possible single additions
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            for x in candidates[:100]:
                if time.time() > _DEADLINE:
                    return best
                if _is_valid_addition(S, x):
                    S.append(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # Try swaps
        if len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            
            for a in to_remove[:50]:
                if time.time() > _DEADLINE:
                    return best
                for x in candidates[:50]:
                    if time.time() > _DEADLINE:
                        return best
                    if _is_valid_swap(S, a, x):
                        S = [v for v in S if v != a] + [x]
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = list(S)
                        break
                if improved:
                    break
        
        if not improved:
            # Perturb: remove 1-2 random elements
            if len(S) > 2:
                k = random.randint(1, 2)
                to_remove = random.sample(S, k)
                S = [v for v in S if v not in to_remove]
    
    return best


def construct():
    """Main entry point: find maximum valid set."""
    
    # Phase 1: Basic greedy
    S1 = _greedy(descending=True)
    S2 = _greedy(descending=False)
    best = max([S1, S2], key=len)
    
    # Phase 2: Multi-start greedy
    if time.time() < _DEADLINE - 30:
        S3 = _multi_start_greedy(num_starts=50, time_budget=18.0)
        if len(S3) > len(best):
            best = S3
    
    # Phase 3: Local search from best
    if time.time() < _DEADLINE - 18:
        S4 = _local_search(best, time_budget=15.0)
        if len(S4) > len(best):
            best = S4
    
    # Phase 4: Hill climbing
    if time.time() < _DEADLINE - 8:
        S5 = _hill_climbing(best, time_budget=8.0)
        if len(S5) > len(best):
            best = S5
    
    # Phase 5: Another local search round if time permits
    if time.time() < _DEADLINE - 3:
        S6 = _local_search(best, time_budget=4.0)
        if len(S6) > len(best):
            best = S6
    
    return best
# EVOLVE-BLOCK-END
