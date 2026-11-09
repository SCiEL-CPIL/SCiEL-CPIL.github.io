import time
from math import gcd, lcm
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    max_n = 512
    
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return set(divs)
    
    # Precompute compatibility matrix
    compatible = np.zeros((max_n+1, max_n+1), dtype=bool)
    for i in range(1, max_n+1):
        for j in range(i, max_n+1):
            if lcm(i, j) <= max_n:
                compatible[i][j] = True
                compatible[j][i] = True
    
    # For each element, precompute the set of compatible elements as a frozenset for fast intersection
    compat_sets = [frozenset() for _ in range(max_n+1)]
    for i in range(1, max_n+1):
        s = set()
        for j in range(1, max_n+1):
            if compatible[i][j]:
                s.add(j)
        compat_sets[i] = frozenset(s)
    
    def greedy_from_seed(seed_set):
        """Greedy expansion from a seed set."""
        S = set(seed_set)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try to add any compatible element
            for x in range(1, max_n+1):
                if x in S:
                    continue
                if all(compatible[x][y] for y in S):
                    S.add(x)
                    changed = True
        return S
    
    # Start with divisors of 480 (2^5·3·5) which has 24 divisors
    S_480 = get_divisors(480)
    best_S = set(S_480)
    best_size = len(S_480)
    
    # Greedy expansion from 480 divisors
    S_expanded = greedy_from_seed(S_480)
    if len(S_expanded) > best_size:
        best_S = S_expanded
        best_size = len(best_S)
    
    # Try greedy from other seeds
    for m in [360, 420, 480, 504, 336, 240, 288, 432, 510, 468, 420, 384, 320, 448]:
        if time.time() > deadline:
            break
        S_seed = get_divisors(m)
        S_exp = greedy_from_seed(S_seed)
        if len(S_exp) > best_size:
            best_S = S_exp
            best_size = len(best_S)
    
    # Local search: try 1-swap improvements
    best_set = set(best_S)
    
    # Try to add any single element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, max_n+1):
            if x in best_set:
                continue
            if all(compatible[x][y] for y in best_set):
                best_set.add(x)
                improved = True
                best_size = len(best_set)
    
    # 1-for-1 swaps to enable further additions
    local_S = set(best_set)
    local_size = len(local_S)
    
    swap_improved = True
    while swap_improved and time.time() < deadline:
        swap_improved = False
        for y in list(local_S):
            if time.time() > deadline:
                break
            for x in range(1, max_n+1):
                if x in local_S:
                    continue
                # Check if (local_S - {y} + {x}) is valid
                test_set = set(local_S)
                test_set.discard(y)
                test_set.add(x)
                valid = True
                for a in test_set:
                    if not all(compatible[a][b] for b in test_set if b != a):
                        valid = False
                        break
                if not valid:
                    continue
                # Check if we can add more elements
                expanded = set(test_set)
                for z in range(1, max_n+1):
                    if z in expanded:
                        continue
                    if all(compatible[z][w] for w in expanded):
                        expanded.add(z)
                if len(expanded) > local_size:
                    local_S = expanded
                    local_size = len(local_S)
                    swap_improved = True
                    if local_size > best_size:
                        best_set = set(local_S)
                        best_size = local_size
                    break
        if swap_improved:
            continue
    
    if len(best_set) > best_size:
        best_size = len(best_set)
    
    # Try a few random restarts with random subsets
    import random
    random.seed(42)
    
    for restart in range(30):
        if time.time() > deadline:
            break
        r = random.randint(1, max_n)
        S_seed = get_divisors(r)
        if len(S_seed) < 10:
            continue
        S_exp = greedy_from_seed(S_seed)
        if len(S_exp) > best_size:
            best_set = S_exp
            best_size = len(best_set)
            # Local search on this
            local_S = set(best_set)
            local_size = len(local_S)
            swap_improved = True
            while swap_improved and time.time() < deadline:
                swap_improved = False
                for y in list(local_S):
                    if time.time() > deadline:
                        break
                    for x in range(1, max_n+1):
                        if x in local_S:
                            continue
                        test_set = set(local_S)
                        test_set.discard(y)
                        test_set.add(x)
                        valid = True
                        for a in test_set:
                            if not all(compatible[a][b] for b in test_set if b != a):
                                valid = False
                                break
                        if not valid:
                            continue
                        expanded = set(test_set)
                        for z in range(1, max_n+1):
                            if z in expanded:
                                continue
                            if all(compatible[z][w] for w in expanded):
                                expanded.add(z)
                        if len(expanded) > local_size:
                            local_S = expanded
                            local_size = len(local_S)
                            if local_size > best_size:
                                best_set = set(local_S)
                                best_size = local_size
                            swap_improved = True
                            break
                    if swap_improved:
                        break
    
    # Additional: try to find larger sets by starting from scratch with different strategies
    # Strategy: pick numbers that are divisors of highly composite numbers
    
    # Try all numbers from 1 to 512 as potential "center" and build around them
    for center in range(1, max_n+1):
        if time.time() > deadline:
            break
        # Start with just the center and its divisors
        S_seed = get_divisors(center)
        if len(S_seed) < 5:
            continue
        S_exp = greedy_from_seed(S_seed)
        if len(S_exp) > best_size:
            best_set = S_exp
            best_size = len(best_set)
    
    return list(best_set)
