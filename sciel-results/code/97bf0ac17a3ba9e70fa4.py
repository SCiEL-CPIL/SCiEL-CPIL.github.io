"""Rewritten program with stronger algorithm for the no-dividing-3-sum task."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_MAX = 48
_DEADLINE = time.time() + 85.0  # Safety margin before 90s kill


def _is_valid_addition(S, x):
    """Check if adding x to valid set S keeps it valid.
    Only need to check:
    1. x does not divide sum of any 3 in S
    2. No a in S divides sum of x + 2 others from S
    """
    S_list = S
    n = len(S_list)
    
    # Check 1: x does not divide sum of any 3 in S
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (S_list[i] + S_list[j] + S_list[k]) % x == 0:
                        return False
    
    # Check 2: for each a in S, a does not divide sum of x + 2 others from S
    for i, a in enumerate(S_list):
        for j in range(n):
            if j == i:
                continue
            for k in range(j+1, n):
                if k == i:
                    continue
                if (x + S_list[j] + S_list[k]) % a == 0:
                    return False
    
    return True


def _is_valid_swap(S, remove_idx, x):
    """Check if swapping S[remove_idx] with x keeps the set valid.
    We remove S[remove_idx] and add x.
    """
    a = S[remove_idx]
    S_list = S
    n = len(S_list)
    
    # After swap, the set is S' = S \\ {a} ∪ {x}
    # We need to verify no element in S' divides sum of any 3 distinct others in S'
    
    # Elements in S': all of S except a, plus x
    
    # For each element e in S':
    #   - if e == x: check x doesn't divide sum of any 3 from (S \\ {a})
    #   - if e == S[i] (i != remove_idx): check S[i] doesn't divide sum of any 3 from (S \\ {a, S[i]}) ∪ {x}
    
    # Check for x: x doesn't divide sum of any 3 from S \\ {a}
    others = [S_list[j] for j in range(n) if j != remove_idx]
    m = len(others)
    if m >= 3:
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    if (others[i] + others[j] + others[k]) % x == 0:
                        return False
    
    # Check for each S[i] (i != remove_idx): S[i] doesn't divide sum of any 3 from (others \\ {S[i]}) ∪ {x}
    for i in range(n):
        if i == remove_idx:
            continue
        e = S_list[i]
        # The "others" for e are: all elements in S' except e
        # S' = (S \\ {a}) ∪ {x}
        # So others for e = (S \\ {a, e}) ∪ {x}
        # We need to check all triples from this set
        
        # Build the list of others for e
        others_for_e = []
        for j in range(n):
            if j != i and j != remove_idx:
                others_for_e.append(S_list[j])
        others_for_e.append(x)
        
        m2 = len(others_for_e)
        if m2 >= 3:
            for i2 in range(m2):
                for j2 in range(i2+1, m2):
                    for k2 in range(j2+1, m2):
                        if (others_for_e[i2] + others_for_e[j2] + others_for_e[k2]) % e == 0:
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
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try to add a random element not in S
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            for x in candidates[:80]:
                if _is_valid_addition(S, x):
                    S.append(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # Try swaps: remove one from S, add one not in S
        if len(S) >= 1:
            to_remove = list(range(len(S)))
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            
            for idx in to_remove[:50]:
                for x in candidates[:50]:
                    if _is_valid_swap(S, idx, x):
                        S[idx] = x
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = list(S)
                        break
                if improved:
                    break
        
        # If no improvement after several rounds, perturb
        if not improved:
            # Random perturbation: remove 2-3 random elements
            if len(S) > 3:
                to_remove = random.sample(range(len(S)), min(3, len(S)-1))
                to_remove.sort(reverse=True)
                for idx in to_remove:
                    S.pop(idx)
    
    return best


def _multi_start_greedy(num_starts=30, time_budget=20.0):
    """Try multiple greedy orders with randomization."""
    best = []
    start_time = time.time()
    
    for _ in range(num_starts):
        if time.time() - start_time > time_budget:
            break
        
        # Randomized greedy: shuffle order with bias toward larger
        order = list(range(1, _N + 1))
        weights = [random.random() * _N + v for v in order]
        order.sort(key=lambda v: -weights[v-1])
        
        selected = []
        for x in order:
            if _is_valid_addition(selected, x):
                selected.append(x)
        
        if len(selected) > len(best):
            best = list(selected)
    
    return best


def _hill_climbing(S, time_budget=20.0):
    """Hill climbing: try all possible single swaps to improve."""
    S = list(S)
    start_time = time.time()
    best = list(S)
    best_len = len(S)
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try to add a random element not in S
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            for x in candidates[:100]:
                if _is_valid_addition(S, x):
                    S.append(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # Try all possible swaps
        if len(S) >= 1:
            candidates = [v for v in all_vals if v not in S]
            for idx in range(len(S)):
                for x in candidates:
                    if _is_valid_swap(S, idx, x):
                        S[idx] = x
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = list(S)
                        break
                if improved:
                    break
        
        if not improved:
            # Perturb
            if len(S) > 3:
                to_remove = random.sample(range(len(S)), min(2, len(S)-1))
                to_remove.sort(reverse=True)
                for idx in to_remove:
                    S.pop(idx)
    
    return best


def construct():
    """Main entry point: find maximum valid set."""
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Phase 1: Basic greedy
    S1 = _greedy(descending=True)
    S2 = _greedy(descending=False)
    best = max([S1, S2], key=len)
    
    # Phase 2: Multi-start greedy
    if time.time() < deadline - 35:
        S3 = _multi_start_greedy(num_starts=40, time_budget=20.0)
        if len(S3) > len(best):
            best = S3
    
    # Phase 3: Local search from best
    if time.time() < deadline - 15:
        S4 = _local_search(best, time_budget=25.0)
        if len(S4) > len(best):
            best = S4
    
    # Phase 4: Hill climbing if time permits
    if time.time() < deadline - 5:
        S5 = _hill_climbing(best, time_budget=10.0)
        if len(S5) > len(best):
            best = S5
    
    # Final safety check: ensure we return a valid set
    # If best is empty or too short, fall back to greedy
    if len(best) < 10:
        best = _greedy(descending=True)
    
    return best
# EVOLVE-BLOCK-END
