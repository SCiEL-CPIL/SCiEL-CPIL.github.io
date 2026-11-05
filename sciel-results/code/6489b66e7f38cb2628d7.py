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

def _build_edge_map(n):
    """Precompute edge index -> (i, j) mapping."""
    edge_map = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_map.append((i, j))
    return edge_map

def _check_property_fast(n, in_neigh):
    """Check if every 4 vertices have a common dominator using bitsets."""
    for a in range(n - 3):
        ia = in_neigh[a]
        for b in range(a + 1, n - 2):
            ia_b = ia & in_neigh[b]
            for c in range(b + 1, n - 1):
                ia_b_c = ia_b & in_neigh[c]
                for d in range(c + 1, n):
                    common = ia_b_c & in_neigh[d]
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    if (common & ~mask) == 0:
                        return False
    return True

def _check_property(n, orientation):
    in_neigh = _build_in_neigh(n, orientation)
    return _check_property_fast(n, in_neigh)

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

def _count_violations_fast(n, in_neigh):
    """Count violations efficiently with early termination of inner loops."""
    count = 0
    for a in range(n - 3):
        ia = in_neigh[a]
        for b in range(a + 1, n - 2):
            ia_b = ia & in_neigh[b]
            if ia_b == 0:
                for c in range(b + 1, n - 1):
                    for d in range(c + 1, n):
                        count += 1
                continue
            for c in range(b + 1, n - 1):
                ia_b_c = ia_b & in_neigh[c]
                if ia_b_c == 0:
                    for d in range(c + 1, n):
                        count += 1
                    continue
                for d in range(c + 1, n):
                    common = ia_b_c & in_neigh[d]
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    if (common & ~mask) == 0:
                        count += 1
    return count

def _local_search(n, time_limit, edge_map):
    """Local search with simulated annealing for a valid tournament on n vertices."""
    deadline = time.time() + time_limit
    
    # Initialize random tournament
    orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
    in_neigh = _build_in_neigh(n, orientation)
    
    violations = _count_violations_fast(n, in_neigh)
    if violations == 0:
        return {'n': n, 'orientation': orientation}
    
    best_violations = violations
    best_orientation = orientation[:]
    
    step = 0
    initial_temp = 5.0
    final_temp = 0.01
    cooling_rate = 0.9999
    
    temp = initial_temp
    
    while time.time() < deadline:
        step += 1
        temp *= cooling_rate
        if temp < final_temp:
            temp = final_temp
        
        # Pick a random edge to flip
        edge_idx = random.randint(0, len(orientation) - 1)
        i, j = edge_map[edge_idx]
        
        old_bit = orientation[edge_idx]
        
        # Flip the edge
        orientation[edge_idx] = 1 - old_bit
        
        # Update in_neigh
        if old_bit == 0:
            # Was i -> j, now j -> i
            in_neigh[j] &= ~(1 << i)
            in_neigh[i] |= (1 << j)
        else:
            # Was j -> i, now i -> j
            in_neigh[i] &= ~(1 << j)
            in_neigh[j] |= (1 << i)
        
        # Count violations
        new_violations = _count_violations_fast(n, in_neigh)
        
        delta = new_violations - best_violations
        
        if delta <= 0:
            # Accept improvement
            best_violations = new_violations
            best_orientation = orientation[:]
            if new_violations == 0:
                return {'n': n, 'orientation': orientation}
        else:
            # Accept worse with probability exp(-delta/temp)
            if random.random() < np.exp(-delta / max(temp, 1e-10)):
                pass  # Accept worse
            else:
                # Revert
                orientation[edge_idx] = old_bit
                if old_bit == 0:
                    in_neigh[j] |= (1 << i)
                    in_neigh[i] &= ~(1 << j)
                else:
                    in_neigh[i] |= (1 << j)
                    in_neigh[j] &= ~(1 << i)
        
        # Safety check
        if time.time() > deadline:
            break
    
    if best_violations == 0:
        return {'n': n, 'orientation': best_orientation}
    return None

def _local_search_multi_start(n, time_limit, edge_map, num_starts=3):
    """Run local search multiple times with different random seeds."""
    per_start_time = time_limit / num_starts
    for _ in range(num_starts):
        if time.time() > time.time() - time_limit + per_start_time * (num_starts - 1) + 1:
            break
        result = _local_search(n, per_start_time, edge_map)
        if result is not None:
            return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5 seconds safety margin
    
    # First, try Paley tournaments for small primes p ≡ 3 (mod 4)
    primes_to_try = [31, 43, 47, 59, 67, 71, 79, 83]
    
    best_result = None
    best_n = 91
    
    for p in primes_to_try:
        if time.time() > deadline:
            break
        
        result = _try_paley_and_check(p, deadline)
        if result is not None:
            best_result = result
            best_n = p
            break
    
    # If we found a valid Paley tournament, try to beat it with local search
    if best_result is not None:
        # Try local search for n = best_n - 1, best_n - 2, ... down to max(5, best_n - 10)
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            for target_n in range(best_n - 1, max(4, best_n - 10), -1):
                if time.time() > deadline - 5:
                    break
                time_for_this = min(20.0, (deadline - time.time()) / 2)
                if time_for_this < 1:
                    break
                edge_map = _build_edge_map(target_n)
                result = _local_search(target_n, time_for_this, edge_map)
                if result is not None:
                    best_result = result
                    best_n = target_n
                    continue
                else:
                    break
    
    # If no Paley worked, fall back to 83
    if best_result is None:
        if time.time() < deadline:
            result = _try_paley_and_check(83, deadline)
            if result is not None:
                return result
        return _paley_orientation(83)
    
    return best_result
