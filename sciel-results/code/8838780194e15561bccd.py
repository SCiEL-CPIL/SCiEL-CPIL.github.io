"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _precompute_conflicts(n_max=180):
    """
    For each a in [1, n_max], find all pairs (b, c) with b != c, b != a, c != a
    such that 1/a = 1/b + 1/c.
    This is equivalent to: (b-a)*(c-a) = a^2
    """
    conflicts = {a: [] for a in range(1, n_max + 1)}
    for a in range(1, n_max + 1):
        a2 = a * a
        seen = set()
        # Case 1: both b, c > a
        # b = a + d, c = a + a2//d where d divides a2
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                d1 = d
                d2 = a2 // d
                # b = a + d1, c = a + d2
                b1 = a + d1
                c1 = a + d2
                if b1 <= n_max and c1 <= n_max and b1 != c1 and b1 != a and c1 != a:
                    pair = (min(b1, c1), max(b1, c1))
                    if pair not in seen:
                        seen.add(pair)
                        conflicts[a].append(pair)
                # b = a + d2, c = a + d1 (same pair, already handled)
        
        # Case 2: both b, c < a
        # b = a - d, c = a - a2//d where d divides a2
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                d1 = d
                d2 = a2 // d
                b1 = a - d1
                c1 = a - d2
                if b1 >= 1 and c1 >= 1 and b1 != c1 and b1 != a and c1 != a:
                    pair = (min(b1, c1), max(b1, c1))
                    if pair not in seen:
                        seen.add(pair)
                        conflicts[a].append(pair)
    
    return conflicts


def _is_valid_set(S, conflicts):
    """Check if set S has no three distinct elements forming a reciprocal Schur triple."""
    S_set = set(S)
    for a in S:
        for (b, c) in conflicts[a]:
            if b in S_set and c in S_set:
                return False
    return True


def _greedy_build(order, conflicts, n_max=180):
    """Greedily build an independent set following the given order."""
    S = set()
    for a in order:
        # Check if adding a creates a conflict
        valid = True
        for (b, c) in conflicts[a]:
            if b in S and c in S:
                valid = False
                break
        if valid:
            S.add(a)
    return S


def _local_search(S, conflicts, n_max=180, max_iter=5000):
    """Local search: try swaps and additions to improve the set."""
    S = set(S)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        # Try adding elements not in S
        for a in range(1, n_max + 1):
            if a not in S:
                valid = True
                for (b, c) in conflicts[a]:
                    if b in S and c in S:
                        valid = False
                        break
                if valid:
                    S.add(a)
                    improved = True
        
        # Try swaps: remove one, add another
        S_list = list(S)
        for i, rem in enumerate(S_list):
            for add in range(1, n_max + 1):
                if add in S:
                    continue
                # Try removing rem and adding add
                S_test = set(S)
                S_test.discard(rem)
                S_test.add(add)
                # Check validity
                valid = True
                for a in S_test:
                    if a == add:
                        for (b, c) in conflicts[a]:
                            if b in S_test and c in S_test:
                                valid = False
                                break
                        if not valid:
                            break
                    else:
                        for (b, c) in conflicts[a]:
                            if b in S_test and c in S_test:
                                valid = False
                                break
                        if not valid:
                            break
                if valid and len(S_test) > len(S):
                    S = S_test
                    improved = True
                    break
            if improved:
                break
    
    return S


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    n_max = 180
    conflicts = _precompute_conflicts(n_max)
    
    # Compute degree for each element (number of conflict pairs)
    degrees = {a: len(conflicts[a]) for a in range(1, n_max + 1)}
    
    best_set = set()
    best_size = 0
    
    # Multi-order greedy
    orders = []
    
    # Ascending
    orders.append(list(range(1, n_max + 1)))
    # Descending
    orders.append(list(range(n_max, 0, -1)))
    # By degree (ascending - low conflict first)
    orders.append(sorted(range(1, n_max + 1), key=lambda x: degrees[x]))
    # By degree (descending)
    orders.append(sorted(range(1, n_max + 1), key=lambda x: -degrees[x]))
    # By value/degree ratio
    orders.append(sorted(range(1, n_max + 1), key=lambda x: -x / (degrees[x] + 1)))
    
    # Random orders
    random.seed(42)
    for _ in range(10):
        perm = list(range(1, n_max + 1))
        random.shuffle(perm)
        orders.append(perm)
    
    for order in orders:
        if time.time() > deadline:
            break
        S = _greedy_build(order, conflicts, n_max)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Local search on best set
    if time.time() < deadline - 5:
        S_improved = _local_search(best_set, conflicts, n_max, max_iter=2000)
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved
    
    # Try local search from multiple starting points
    if time.time() < deadline - 10:
        for order in orders[:5]:
            if time.time() > deadline:
                break
            S = _greedy_build(order, conflicts, n_max)
            S_improved = _local_search(S, conflicts, n_max, max_iter=1000)
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_set = set(S_improved)
    
    # Final check
    if not _is_valid_set(best_set, conflicts):
        # Fallback to simple valid set
        best_set = set(range(91, 181))
    
    return list(best_set)

# EVOLVE-BLOCK-END
