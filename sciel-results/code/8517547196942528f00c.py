import time
import math
import random
from itertools import combinations

def _is_prime(n):
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

def _paley_orientation(p):
    """Generate Paley tournament orientation for prime p ≡ 3 (mod 4)."""
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _check_property(n, orientation):
    """Check if every 4-set has a common dominator. Returns True if valid."""
    in_neigh = [0] * n
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                in_neigh[j] |= (1 << i)
            else:
                in_neigh[i] |= (1 << j)
            idx += 1
    
    for a, b, c, d in combinations(range(n), 4):
        if (in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]) == 0:
            return False
    return True

def _build_in_neigh(n, orientation):
    in_neigh = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                in_neigh[j] |= (1 << i)
            else:
                in_neigh[i] |= (1 << j)
            idx += 1
    return in_neigh

def _compute_cost(n, in_neigh):
    cost = 0
    for a, b, c, d in combinations(range(n), 4):
        if (in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]) == 0:
            cost += 1
    return cost

def _local_search(n, deadline, seed=42):
    """Run greedy local search for n vertices. Returns orientation if valid, else None."""
    rng = random.Random(seed)
    orientation = [rng.randint(0, 1) for _ in range(n * (n - 1) // 2)]
    in_neigh = _build_in_neigh(n, orientation)
    current_cost = _compute_cost(n, in_neigh)
    
    if current_cost == 0:
        return orientation
    
    best_orientation = orientation[:]
    best_cost = current_cost
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        idx = rng.randint(0, n * (n - 1) // 2 - 1)
        
        # Find i, j from idx
        i = 0
        rem = idx
        while i < n:
            count = n - i - 1
            if rem < count:
                j = i + 1 + rem
                break
            rem -= count
            i += 1
        
        old_val = orientation[idx]
        orientation[idx] = 1 - old_val
        
        if old_val == 0:
            in_neigh[j] &= ~(1 << i)
            in_neigh[i] |= (1 << j)
        else:
            in_neigh[i] &= ~(1 << j)
            in_neigh[j] |= (1 << i)
        
        new_cost = _compute_cost(n, in_neigh)
        
        if new_cost < current_cost:
            current_cost = new_cost
            no_improve = 0
            if new_cost < best_cost:
                best_cost = new_cost
                best_orientation = orientation[:]
                if best_cost == 0:
                    return best_orientation
        else:
            orientation[idx] = old_val
            if old_val == 0:
                in_neigh[j] |= (1 << i)
                in_neigh[i] &= ~(1 << j)
            else:
                in_neigh[i] |= (1 << j)
                in_neigh[j] &= ~(1 << i)
            no_improve += 1
    
    if best_cost == 0:
        return best_orientation
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # Try local search for small n with multiple seeds
    for n in range(31, 45):
        if time.time() > deadline - 20:
            break
        for seed in [42, 123, 777]:
            if time.time() > deadline - 20:
                break
            result = _local_search(n, deadline - 20, seed=seed)
            if result is not None:
                return {'n': n, 'orientation': result}
    
    # Fall back to Paley search
    candidates = []
    for p in range(31, 91):
        if p % 4 == 3 and _is_prime(p):
            candidates.append(p)
    
    for p in candidates:
        if time.time() > deadline:
            break
        orientation = _paley_orientation(p)
        if _check_property(p, orientation):
            return {'n': p, 'orientation': orientation}
    
    p = 83
    orientation = _paley_orientation(p)
    return {'n': p, 'orientation': orientation}
