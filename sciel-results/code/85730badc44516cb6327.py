"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from math import gcd

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))


def _precompute():
    """Precompute req[x][b] = x//gcd(x,b) for fast divisibility checks."""
    req = {}
    for x in range(2, NMAX + 1):
        req[x] = {}
        for b in range(2, NMAX + 1):
            if x == b:
                continue
            req[x][b] = x // gcd(x, b)
    return req


def _valid_addition(x, S, req):
    """Check if x can be added to set S (list) without violating constraint."""
    n = len(S)
    for i in range(n):
        b = S[i]
        for j in range(n):
            if i == j:
                continue
            c = S[j]
            # Check x | b*c
            d = req[x][b]
            if c % d == 0:
                return False
            # Check b | x*c
            d2 = req[b][x]
            if c % d2 == 0:
                return False
            # Check c | x*b
            d3 = req[c][x]
            if b % d3 == 0:
                return False
    return True


def _conflict_degree(x, S, req):
    """Count number of pairs (b,c) in S where x is involved in a conflict."""
    count = 0
    n = len(S)
    for i in range(n):
        b = S[i]
        for j in range(n):
            if i == j:
                continue
            c = S[j]
            if x == b or x == c:
                continue
            # Check if x|b*c or b|x*c or c|x*b
            d = req[x][b]
            if c % d == 0:
                count += 1
                continue
            d2 = req[b][x]
            if c % d2 == 0:
                count += 1
                continue
            d3 = req[c][x]
            if b % d3 == 0:
                count += 1
    return count


def _greedy_remove_by_degree(req, deadline):
    """Start with all numbers, iteratively remove elements with highest conflict degree."""
    S = list(range(2, NMAX + 1))
    S_set = set(S)
    
    while time.time() < deadline:
        # Compute conflict degree for each element
        S_list = list(S)
        n = len(S_list)
        if n <= 2:
            break
        
        # Find max conflict degree
        max_deg = 0
        max_elem = None
        for x in S_list:
            deg = _conflict_degree(x, S_list, req)
            if deg > max_deg:
                max_deg = deg
                max_elem = x
        
        if max_deg == 0:
            break  # No conflicts, done
        
        S.remove(max_elem)
        S_set.discard(max_elem)
    
    return S


def _greedy_add_largest(req, deadline):
    """Greedy: add numbers from largest to smallest if valid."""
    chosen = []
    for x in range(NMAX, 1, -1):
        if time.time() > deadline:
            break
        if _valid_addition(x, chosen, req):
            chosen.append(x)
    return chosen


def _greedy_add_smallest(req, deadline):
    """Greedy: add numbers from smallest to largest if valid."""
    chosen = []
    for x in range(2, NMAX + 1):
        if time.time() > deadline:
            break
        if _valid_addition(x, chosen, req):
            chosen.append(x)
    return chosen


def _greedy_add_random(req, deadline, seed=42):
    """Greedy: add numbers in random order if valid."""
    rng = random.Random(seed)
    candidates = list(range(2, NMAX + 1))
    rng.shuffle(candidates)
    chosen = []
    for x in candidates:
        if time.time() > deadline:
            break
        if _valid_addition(x, chosen, req):
            chosen.append(x)
    return chosen


def _local_search(S, req, deadline, max_iters=500, seed=99999):
    """Local search: try adding, removing, swapping elements."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)
    
    for _it in range(max_iters):
        if time.time() > deadline:
            break
        
        op = rng.random()
        
        if op < 0.4 and len(S) < NMAX:
            # Try adding
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                rng.shuffle(candidates)
                for x in candidates[:30]:
                    if _valid_addition(x, S, req):
                        S.append(x)
                        S_set.add(x)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        break
        
        elif op < 0.7 and len(S) > 2:
            # Try removing one and adding one (swap)
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                rng.shuffle(candidates)
                for x in candidates[:40]:
                    if _valid_addition(x, S_temp, req):
                        S = S_temp + [x]
                        S_set = set(S)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        break
                else:
                    S = S_temp + [removed]
                    S_set = set(S)
        
        else:
            # Try removing a random element to allow re-growth
            if S:
                i = rng.randrange(len(S))
                removed = S[i]
                S.pop(i)
                S_set.discard(removed)
                candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
                if candidates:
                    rng.shuffle(candidates)
                    for x in candidates[:30]:
                        if _valid_addition(x, S, req):
                            S.append(x)
                            S_set.add(x)
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
                            break
    
    return best


def _simulated_annealing(S, req, deadline, max_iters=4000, T_start=100.0, T_end=0.5, seed=55555):
    """Simulated annealing for the set optimization."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)
    
    current_temp = T_start
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    for _it in range(max_iters):
        if time.time() > deadline:
            break
        current_temp *= cooling
        
        op = rng.random()
        
        if op < 0.35 and len(S) < NMAX:
            # Add
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition(x, S, req):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
        
        elif op < 0.65 and len(S) > 2:
            # Swap: remove one, add one
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition(x, S_temp, req):
                    S = S_temp + [x]
                    S_set = set(S)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
        
        elif op < 0.85 and len(S) > 3:
            # Remove two, try to add two different ones
            i, j = rng.sample(range(len(S)), 2)
            i, j = min(i, j), max(i, j)
            S_temp = S[:i] + S[i+1:j] + S[j+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if len(candidates) >= 2:
                x = rng.choice(candidates)
                if _valid_addition(x, S_temp, req):
                    S_temp2 = S_temp + [x]
                    S_temp2_set = S_temp_set | {x}
                    remaining = [y for y in candidates if y != x]
                    if remaining:
                        y = rng.choice(remaining)
                        if _valid_addition(y, S_temp2, req):
                            S = S_temp2 + [y]
                            S_set = set(S)
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
        
        else:
            # Remove one (to escape local optima)
            if S:
                i = rng.randrange(len(S))
                removed = S[i]
                S.pop(i)
                S_set.discard(removed)
                candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
                if candidates:
                    rng.shuffle(candidates)
                    for x in candidates[:20]:
                        if _valid_addition(x, S, req):
                            S.append(x)
                            S_set.add(x)
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
                            break
    
    return best


def _multi_start(req, deadline):
    """Run multiple strategies and return the best result."""
    best = None
    best_size = 0
    
    def consider(c):
        nonlocal best, best_size
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # 1. Greedy remove by conflict degree
    if time.time() < deadline - 20:
        consider(_greedy_remove_by_degree(req, deadline - 20))
    
    # 2. Greedy add largest first
    if time.time() < deadline - 15:
        consider(_greedy_add_largest(req, deadline - 15))
    
    # 3. Greedy add smallest first
    if time.time() < deadline - 10:
        consider(_greedy_add_smallest(req, deadline - 10))
    
    # 4. Multiple random greedy starts with local search
    if time.time() < deadline - 25:
        for seed in [1, 7, 13, 42, 99, 256, 777, 1234]:
            if time.time() > deadline - 25:
                break
            S = _greedy_add_random(req, deadline - 25, seed=seed)
            S_refined = _local_search(S, req, deadline - 25, max_iters=300, seed=seed * 1000)
            consider(S_refined)
    
    # 5. Simulated annealing from best so far
    if best and time.time() < deadline - 15:
        for sa_seed in [55555, 12345, 77777, 99999, 11111]:
            if time.time() > deadline - 10:
                break
            consider(_simulated_annealing(best, req, deadline - 10, max_iters=5000, seed=sa_seed))
    
    # 6. More local search on best
    if best and time.time() < deadline - 5:
        consider(_local_search(best, req, deadline - 5, max_iters=2000, seed=424242))
    
    # 7. Final: try to add any remaining valid elements
    if best and time.time() < deadline - 2:
        S = list(best)
        S_set = set(S)
        for x in range(2, NMAX + 1):
            if x not in S_set:
                if _valid_addition(x, S, req):
                    S.append(x)
                    S_set.add(x)
        consider(S)
    
    if best is None:
        # Fallback: all primes
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        best = [x for x in range(2, NMAX + 1) if is_prime(x)]
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    req = _precompute()
    
    best = _multi_start(req, deadline)
    
    return [int(x) for x in best]
