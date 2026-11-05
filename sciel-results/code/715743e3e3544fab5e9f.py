import time
import random
import itertools

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
    """Return orientation bits for a Paley tournament of prime order p (p ≡ 3 mod 4)."""
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
    return orientation

def _build_in_neigh(n, orientation):
    """Build in-neighborhood bitsets from orientation list."""
    in_neigh = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                # i -> j, so i is in in_neigh[j]
                in_neigh[j] |= (1 << i)
            else:
                # j -> i, so j is in in_neigh[i]
                in_neigh[i] |= (1 << j)
            idx += 1
    return in_neigh

def _check_property_fast(n, orientation):
    """Check if every 4-set has a common dominator using bitsets."""
    in_neigh = _build_in_neigh(n, orientation)
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        common = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        if (common & ~mask) == 0:
            return False
    return True

def _count_bad_quads(n, orientation):
    """Count number of 4-sets lacking a common dominator."""
    in_neigh = _build_in_neigh(n, orientation)
    bad_count = 0
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        common = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        if (common & ~mask) == 0:
            bad_count += 1
    return bad_count

def _local_search(n, time_limit, deadline):
    """Local search to find a valid tournament on n vertices."""
    # Start with random tournament
    orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
    best_orientation = orientation[:]
    best_bad = _count_bad_quads(n, orientation)
    
    if best_bad == 0:
        return orientation
    
    # Simulated annealing / hill climbing with restarts
    temp = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        if best_bad == 0:
            break
        
        # Pick a random edge to flip
        idx = random.randint(0, len(orientation) - 1)
        old_val = orientation[idx]
        orientation[idx] = 1 - old_val
        
        new_bad = _count_bad_quads(n, orientation)
        delta = new_bad - best_bad
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / temp):
            best_orientation = orientation[:]
            best_bad = new_bad
        else:
            orientation[idx] = old_val
        
        temp *= cooling
        if temp < 0.001:
            temp = 1.0
            # Restart with new random tournament
            orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            cur_bad = _count_bad_quads(n, orientation)
            if cur_bad < best_bad:
                best_orientation = orientation[:]
                best_bad = cur_bad
            else:
                orientation = best_orientation[:]
    
    return best_orientation

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s margin before 150s kill
    
    # Try Paley tournaments for primes p ≡ 3 mod 4, starting from smallest
    # Primes ≡ 3 mod 4 in [31, 90]: 31, 43, 47, 59, 67, 71, 79, 83
    paley_primes = [p for p in range(31, 91) if p % 4 == 3 and _is_prime(p)]
    
    for p in paley_primes:
        if time.time() > deadline:
            break
        
        orientation = _paley_orientation(p)
        
        # Check property
        if _check_property_fast(p, orientation):
            return {'n': p, 'orientation': orientation}
    
    # If no Paley tournament worked, try local search for n < 67
    # Start with n=31 (Erdos lower bound)
    if time.time() < deadline:
        for n in [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
            if time.time() > deadline - 5:
                break
            result = _local_search(n, deadline - time.time(), deadline)
            if result is not None:
                bad = _count_bad_quads(n, result)
                if bad == 0:
                    return {'n': n, 'orientation': result}
    
    # Fallback to Paley(67) which is known to work
    p = 67
    orientation = _paley_orientation(p)
    return {'n': p, 'orientation': orientation}
