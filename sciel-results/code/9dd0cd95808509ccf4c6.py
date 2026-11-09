"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from random import Random

DEADLINE = time.time() + 85  # safety margin before 90s kill


def _is_bad_triple(a, b, c):
    """Return True if the three pairwise gcds of a,b,c are all equal."""
    g_ab = gcd(a, b)
    g_ac = gcd(a, c)
    g_bc = gcd(b, c)
    return g_ab == g_ac == g_bc


def _build_conflict_graph(n=128):
    """
    Build a 3-uniform hypergraph of 'bad triples' over [1..n].
    We'll represent this as, for each pair (i,j), the set of k such that
    {i,j,k} is a bad triple. This helps with fast validity checks.
    Returns: pair_conflicts[i][j] = set of k where {i,j,k} is bad.
    """
    nums = list(range(1, n + 1))
    idx = {v: i for i, v in enumerate(nums)}
    # pair_conflicts[i][j] = set of indices k (k != i, k != j) such that triple is bad
    pair_conflicts = [[set() for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nums[i], nums[j]
            g_ab = gcd(a, b)
            # For the triple (a, b, c) to be bad, we need gcd(a,c) == gcd(b,c) == g_ab
            for k in range(n):
                if k == i or k == j:
                    continue
                c = nums[k]
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    pair_conflicts[i][j].add(k)
                    pair_conflicts[j][i].add(k)
    return nums, pair_conflicts


def _check_addition(S, x, nums, pair_conflicts):
    """Check if adding index x to set S (set of indices) is valid."""
    for i in S:
        for j in S:
            if i >= j:
                continue
            if x in pair_conflicts[i][j]:
                return False
    return True


def _greedy_with_order(order, nums, pair_conflicts):
    """Greedy: add elements in given order if valid."""
    S = set()
    for x in order:
        if x not in S and _check_addition(S, x, nums, pair_conflicts):
            S.add(x)
    return S


def _local_search_improve(S, rng, nums, pair_conflicts, max_iter=2000):
    """Try to add random elements not in S."""
    S = set(S)
    not_in = [x for x in range(len(nums)) if x not in S]
    for _ in range(max_iter):
        if time.time() > DEADLINE:
            break
        if not not_in:
            break
        x = rng.choice(not_in)
        if _check_addition(S, x, nums, pair_conflicts):
            S.add(x)
            not_in.remove(x)
    return S


def _simulate_annealing(S_init, rng, nums, pair_conflicts, steps=50000):
    """
    Simulated annealing: state is the set S. 
    Moves: add a random element, or remove a random element, or swap.
    Cost = -|S| (we want to maximize size).
    """
    S = set(S_init)
    n = len(nums)
    all_indices = list(range(n))
    current_size = len(S)
    best_S = set(S)
    best_size = current_size
    
    T = 10.0
    cooling = 0.9999
    
    for step in range(steps):
        if time.time() > DEADLINE:
            break
        
        # Choose a random move
        move_type = rng.random()
        if move_type < 0.4:
            # Try to add
            candidates = [x for x in all_indices if x not in S]
            if not candidates:
                continue
            x = rng.choice(candidates)
            if _check_addition(S, x, nums, pair_conflicts):
                S.add(x)
                new_size = len(S)
                delta = -(new_size - current_size)  # cost change
                if delta <= 0 or rng.random() < pow(2.718, -delta / T):
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
                else:
                    S.discard(x)
            # If invalid, just skip (don't add)
        elif move_type < 0.7:
            # Try to remove
            if S:
                x = rng.choice(list(S))
                S.discard(x)
                new_size = len(S)
                delta = -(new_size - current_size)
                if delta <= 0 or rng.random() < pow(2.718, -delta / T):
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
                else:
                    S.add(x)
        else:
            # Swap: remove one, try to add another
            if S:
                x_rm = rng.choice(list(S))
                S.discard(x_rm)
                candidates = [x for x in all_indices if x not in S]
                if candidates:
                    x_add = rng.choice(candidates)
                    if _check_addition(S, x_add, nums, pair_conflicts):
                        S.add(x_add)
                        new_size = len(S)
                        delta = -(new_size - current_size)
                        if delta <= 0 or rng.random() < pow(2.718, -delta / T):
                            current_size = new_size
                            if current_size > best_size:
                                best_size = current_size
                                best_S = set(S)
                        else:
                            S.discard(x_add)
                            S.add(x_rm)
                    else:
                        S.add(x_rm)
        
        T *= cooling
        if T < 0.01:
            break
    
    return best_S


def _greedy_coloring_mis(adj_matrix, n):
    """
    Greedy coloring based approach to find a large independent set.
    This is a heuristic; we use a different formulation here since
    our problem is 3-uniform, not 2-uniform.
    Instead, we'll use a different approach: iterative removal of
    elements in "bad triangles" with priority.
    """
    pass


def construct():
    best = set()
    best_len = 0
    
    nums, pair_conflicts = _build_conflict_graph(128)
    n = 128
    
    rng = Random(42)
    
    # Strategy 1: Multiple greedy orderings
    orders = [
        list(range(n)),
        list(range(n - 1, -1, -1)),
        sorted(range(n), key=lambda x: nums[x] * x),
        sorted(range(n), key=lambda x: -nums[x]),
    ]
    
    # Add prime-power priority
    def prime_power_score(idx):
        v = nums[idx]
        if v < 2:
            return 0
        d = 2
        cnt = 0
        tmp = v
        while d * d <= tmp:
            if tmp % d == 0:
                cnt += 1
                while tmp % d == 0:
                    tmp //= d
                if tmp > 1:
                    return 0
            d += 1
        if tmp > 1:
            cnt += 1
        return cnt if cnt == 1 else 0
    
    orders.append(sorted(range(n), key=lambda x: -prime_power_score(x)))
    
    for order in orders:
        S = _greedy_with_order(order, nums, pair_conflicts)
        if len(S) > best_len:
            best = S
            best_len = len(S)
    
    # Strategy 2: Random greedy + local search
    for i in range(100):
        if time.time() > DEADLINE:
            break
        order = list(range(n))
        rng.shuffle(order)
        S = _greedy_with_order(order, nums, pair_conflicts)
        S = _local_search_improve(S, rng, nums, pair_conflicts, max_iter=500)
        if len(S) > best_len:
            best = S
            best_len = len(S)
    
    # Strategy 3: Simulated annealing from multiple starts
    for i in range(5):
        if time.time() > DEADLINE:
            break
        # Start from a random greedy solution
        order = list(range(n))
        rng.shuffle(order)
        S_init = _greedy_with_order(order, nums, pair_conflicts)
        S = _simulate_annealing(S_init, rng, nums, pair_conflicts, steps=30000)
        if len(S) > best_len:
            best = S
            best_len = len(S)
    
    # Strategy 4: SA from the best found so far
    if best:
        S = _simulate_annealing(set(best), rng, nums, pair_conflicts, steps=50000)
        if len(S) > best_len:
            best = S
            best_len = len(S)
    
    # Strategy 5: Structural - try specific subsets
    # All odd numbers
    odd_order = [i for i in range(n) if nums[i] % 2 == 1]
    S_odd = _greedy_with_order(odd_order, nums, pair_conflicts)
    S_odd = _local_search_improve(S_odd, rng, nums, pair_conflicts, max_iter=300)
    if len(S_odd) > best_len:
        best = S_odd
        best_len = len(S_odd)
    
    # Numbers coprime to 6
    c6_order = [i for i in range(n) if nums[i] % 2 != 0 and nums[i] % 3 != 0]
    S_c6 = _greedy_with_order(c6_order, nums, pair_conflicts)
    S_c6 = _local_search_improve(S_c6, rng, nums, pair_conflicts, max_iter=300)
    if len(S_c6) > best_len:
        best = S_c6
        best_len = len(S_c6)
    
    # Strategy 6: More SA iterations if time allows
    if time.time() < DEADLINE - 10:
        for i in range(3):
            if time.time() > DEADLINE:
                break
            S = _simulate_annealing(set(best), Random(1000 + i), nums, pair_conflicts, steps=40000)
            if len(S) > best_len:
                best = S
                best_len = len(S)
    
    if not best:
        return [1]
    
    result = sorted(nums[i] for i in best)
    return result
# EVOLVE-BLOCK-END
