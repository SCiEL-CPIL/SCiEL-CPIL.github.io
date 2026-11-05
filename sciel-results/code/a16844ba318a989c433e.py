import time
import random
import numpy as np
from itertools import combinations

def _build_in_neigh(n, orientation):
    """Build in-neighbor bitsets from orientation list."""
    in_neigh = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                in_neigh[j] |= (1 << i)
            else:
                in_neigh[i] |= (1 << j)
            idx += 1
    return in_neigh

def _check_property_fast(n, in_neigh):
    """Check if every 4 vertices have a common dominator using bitsets."""
    for a in range(n - 3):
        ia = in_neigh[a]
        for b in range(a + 1, n - 2):
            ib = in_neigh[b]
            for c in range(b + 1, n - 1):
                ic = in_neigh[c]
                for d in range(c + 1, n):
                    id_ = in_neigh[d]
                    
                    if (ib & (1 << a)) and (ic & (1 << a)) and (id_ & (1 << a)):
                        continue
                    if (ia & (1 << b)) and (ic & (1 << b)) and (id_ & (1 << b)):
                        continue
                    if (ia & (1 << c)) and (ib & (1 << c)) and (id_ & (1 << c)):
                        continue
                    if (ia & (1 << d)) and (ib & (1 << d)) and (ic & (1 << d)):
                        continue
                    return False
    return True

def _paley_orientation(p):
    """Return orientation bits for a Paley tournament of prime order p (p ≡ 3 mod 4)."""
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    n = p
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': n, 'orientation': orientation}

def _try_paley_and_check(p, deadline):
    """Try Paley tournament on p vertices and check if valid."""
    if time.time() > deadline:
        return None
    result = _paley_orientation(p)
    n = result['n']
    in_neigh = _build_in_neigh(n, result['orientation'])
    if _check_property_fast(n, in_neigh):
        return result
    return None

def _local_search(n, time_limit, start_orientation=None):
    """Local search for a valid tournament on n vertices."""
    deadline = time.time() + time_limit
    
    if start_orientation is not None:
        orientation = start_orientation[:]
    else:
        orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
    
    in_neigh = _build_in_neigh(n, orientation)
    
    def count_violations(in_neigh, n):
        count = 0
        for a in range(n - 3):
            ia = in_neigh[a]
            for b in range(a + 1, n - 2):
                ib = in_neigh[b]
                for c in range(b + 1, n - 1):
                    ic = in_neigh[c]
                    for d in range(c + 1, n):
                        id_ = in_neigh[d]
                        
                        if (ib & (1 << a)) and (ic & (1 << a)) and (id_ & (1 << a)):
                            continue
                        if (ia & (1 << b)) and (ic & (1 << b)) and (id_ & (1 << b)):
                            continue
                        if (ia & (1 << c)) and (ib & (1 << c)) and (id_ & (1 << c)):
                            continue
                        if (ia & (1 << d)) and (ib & (1 << d)) and (ic & (1 << d)):
                            continue
                        count += 1
        return count

    violations = count_violations(in_neigh, n)
    if violations == 0:
        return {'n': n, 'orientation': orientation}
    
    best_violations = violations
    best_orientation = orientation[:]
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Pick a random edge to flip
        edge_idx = random.randint(0, len(orientation) - 1)
        
        # Find (i, j) using precomputed mapping
        i = j = -1
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if idx == edge_idx:
                    break
                idx += 1
            if idx == edge_idx:
                break
        
        # Flip the edge
        orientation[edge_idx] = 1 - orientation[edge_idx]
        
        # Update in_neigh
        if orientation[edge_idx] == 0:
            # i -> j
            in_neigh[i] &= ~(1 << j)
            in_neigh[j] |= (1 << i)
        else:
            # j -> i
            in_neigh[j] &= ~(1 << i)
            in_neigh[i] |= (1 << j)
        
        # Count violations every 50 steps
        if step % 50 == 0:
            violations = count_violations(in_neigh, n)
            if violations < best_violations:
                best_violations = violations
                best_orientation = orientation[:]
                if violations == 0:
                    return {'n': n, 'orientation': orientation}
    
    if best_violations == 0:
        return {'n': n, 'orientation': best_orientation}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5 seconds safety margin
    
    # Primes p = 3 mod 4
    # Smallest known valid is 67.
    # Let's try 31, 43, 47, 59, 67.
    primes_to_try = [31, 43, 47, 59, 67]
    
    best_result = None
    best_n = 91
    
    for p in primes_to_try:
        if time.time() > deadline:
            break
        
        result = _try_paley_and_check(p, deadline)
        if result is not None:
            best_result = result
            best_n = p
            break  # Found the smallest valid Paley
    
    # If we found a valid Paley tournament, try to beat it with local search
    if best_result is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            # Try to find a valid tournament with n < best_n
            # Start from best_n - 1 down to max(5, best_n - 3)
            for target_n in range(best_n - 1, max(4, best_n - 3), -1):
                if time.time() > deadline - 5:
                    break
                time_for_this = min(15.0, (deadline - time.time()) / 2)
                if time_for_this < 1:
                    break
                result = _local_search(target_n, time_for_this)
                if result is not None:
                    best_result = result
                    best_n = target_n
                    continue
                else:
                    break
    
    # If no Paley worked, fall back to 67 (known valid)
    if best_result is None:
        if time.time() < deadline:
            result = _try_paley_and_check(67, deadline)
            if result is not None:
                return result
        # Absolute fallback
        return _paley_orientation(67)
    
    return best_result
