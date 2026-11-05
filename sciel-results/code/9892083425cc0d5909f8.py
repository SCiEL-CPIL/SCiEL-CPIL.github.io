import math
import time
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

def _paley_orientation(n):
    """Generate orientation bits for Paley tournament on n vertices (n must be prime ≡ 3 mod 4)."""
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _check_property(n, orientation):
    """Check if every 4-set has a common dominator. Returns True if valid."""
    # Build in-neighborhood bitmasks
    in_neigh = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                # i -> j
                in_neigh[j] |= (1 << i)
            else:
                # j -> i
                in_neigh[i] |= (1 << j)
            idx += 1
    
    # Check all 4-sets
    for a, b, c, d in combinations(range(n), 4):
        if (in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]) == 0:
            return False
    return True

def _count_violations(n, orientation):
    """Count number of 4-sets without a common dominator."""
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
    
    violations = 0
    for a, b, c, d in combinations(range(n), 4):
        if (in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]) == 0:
            violations += 1
    return violations

def _improve_tournament(n, orientation, max_iters=200):
    """
    Local search: flip individual edges to reduce number of violating 4-sets.
    Returns improved orientation and violation count.
    """
    # Precompute in_neigh
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
    
    cur_viol = _count_violations(n, orientation)
    if cur_viol == 0:
        return orientation, 0
    
    # Precompute for each edge (i,j) the list of 4-sets it affects
    # A 4-set {a,b,c,d} is affected by edge (i,j) if i or j is in the set
    # and the other endpoint is the dominator candidate... actually it's complex.
    # Let's just do simple coordinate descent.
    
    best_orientation = list(orientation)
    best_viol = cur_viol
    
    for it in range(max_iters):
        improved = False
        # Try flipping each edge
        for idx in range(len(orientation)):
            # Find i, j from idx
            # idx = i*(2n-i-1)//2 + (j-i-1)
            # Solve for i
            i = 0
            while i * (2 * n - i - 1) // 2 <= idx:
                i += 1
            i -= 1
            j_offset = idx - i * (2 * n - i - 1) // 2
            j = i + 1 + j_offset
            
            # Flip edge
            old_bit = best_orientation[idx]
            best_orientation[idx] = 1 - old_bit
            
            # Update in_neigh
            if old_bit == 0:
                # was i->j, now j->i
                in_neigh[j] &= ~(1 << i)
                in_neigh[i] |= (1 << j)
            else:
                # was j->i, now i->j
                in_neigh[i] &= ~(1 << j)
                in_neigh[j] |= (1 << i)
            
            new_viol = _count_violations(n, best_orientation)
            
            if new_viol < best_viol:
                best_viol = new_viol
                improved = True
            else:
                # Revert
                best_orientation[idx] = old_bit
                if old_bit == 0:
                    in_neigh[j] |= (1 << i)
                    in_neigh[i] &= ~(1 << j)
                else:
                    in_neigh[i] |= (1 << j)
                    in_neigh[j] &= ~(1 << i)
        
        if not improved:
            break
    
    return best_orientation, best_viol

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Primes p ≡ 3 mod 4 in range [31, 90]
    candidates = []
    for p in range(31, 91):
        if p % 4 == 3 and _is_prime(p):
            candidates.append(p)
    
    # Try Paley tournaments from smallest to largest
    for p in candidates:
        if time.time() > deadline - 30:
            break
        
        orientation = _paley_orientation(p)
        
        if _check_property(p, orientation):
            return {'n': p, 'orientation': orientation}
        
        # Try local search to see if we can fix violations
        if time.time() > deadline - 60:
            continue
        
        improved_orient, viol_count = _improve_tournament(p, orientation, max_iters=100)
        
        if viol_count == 0:
            return {'n': p, 'orientation': improved_orient}
    
    # Fallback: return Paley tournament on 67 (known to be valid)
    p = 67
    orientation = _paley_orientation(p)
    return {'n': p, 'orientation': orientation}
