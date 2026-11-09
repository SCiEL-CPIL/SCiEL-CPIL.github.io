"""
3-progression-free set on [1,256] maximizing reciprocal weight.

Strategy:
- Use a bitset-based representation for fast 3-AP checking.
- Precompute for each x the set of "conflicts" (pairs that would form a 3-AP with x).
- Use simulated annealing with bitmask operations for speed.
- Multiple restarts with different initializations.
"""
import time
import random
from fractions import Fraction
import sys

N = 256
TIME_LIMIT = 85.0  # safety margin before 90s kill

def _build_conflict_masks():
    """
    For each x in 1..256, build a bitmask of all y such that {x,y} is part of some
    3-term AP (i.e., there exists z where x,y,z form a 3-AP).
    Actually, we need: when we have a set S, x can be added iff no y in S conflicts with x.
    
    Two elements a, b conflict if there exists c such that {a,b,c} is a 3-AP.
    This happens when:
    - c = 2a - b (a is middle) -> need 1 <= c <= 256, c != a, c != b
    - c = 2b - a (b is middle) -> need 1 <= c <= 256, c != a, c != b  
    - a and b are endpoints: c = (a+b)/2 must be integer, 1 <= c <= 256, c != a, c != b
    
    So conflict(a,b) exists iff:
    (2a-b in [1,256] and 2a-b != a and 2a-b != b) or
    (2b-a in [1,256] and 2b-a != a and 2b-a != b) or
    ((a+b) even and (a+b)//2 in [1,256] and (a+b)//2 != a and (a+b)//2 != b)
    """
    conflict = [0] * (N + 1)  # conflict[x] is bitmask of all y that conflict with x
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            has_conflict = False
            # a is middle: c = 2a - b
            c = 2 * a - b
            if 1 <= c <= N and c != a and c != b:
                has_conflict = True
            # b is middle: c = 2b - a
            c = 2 * b - a
            if 1 <= c <= N and c != a and c != b:
                has_conflict = True
            # a,b are endpoints: c = (a+b)/2
            if (a + b) % 2 == 0:
                c = (a + b) // 2
                if 1 <= c <= N and c != a and c != b:
                    has_conflict = True
            
            if has_conflict:
                conflict[a] |= (1 << b)
                conflict[b] |= (1 << a)
    return conflict

def _is_3free(mask, conflict):
    """Check if the set represented by mask is 3-AP-free."""
    for x in range(1, N + 1):
        if mask & (1 << x):
            # Check if any other element in mask conflicts with x
            if conflict[x] & mask:
                return False
    return True

def _can_add_bitset(x, mask, conflict):
    """Check if x can be added to set represented by mask."""
    if mask & (1 << x):
        return True  # already there
    return (conflict[x] & mask) == 0

def _recip_sum_from_mask(mask):
    total = Fraction(0, 1)
    for x in range(1, N + 1):
        if mask & (1 << x):
            total += Fraction(1, x)
    return total

def _recip_float_from_mask(mask, recip_floats):
    """Fast float approximation of reciprocal sum."""
    s = 0.0
    for x in range(1, N + 1):
        if mask & (1 << x):
            s += recip_floats[x]
    return s

def _greedy_initial(conflict, recip_floats):
    """Greedy: add elements in order of decreasing 1/x."""
    mask = 0
    for x in range(1, N + 1):
        if _can_add_bitset(x, mask, conflict):
            mask |= (1 << x)
    return mask

def _greedy_randomized(conflict, recip_floats, rng):
    """Randomized greedy with noise."""
    mask = 0
    candidates = list(range(1, N + 1))
    weights = [recip_floats[x] * (0.5 + rng.random()) for x in candidates]
    order = sorted(candidates, key=lambda x: weights[x - 1], reverse=True)
    for x in order:
        if _can_add_bitset(x, mask, conflict):
            mask |= (1 << x)
    return mask

def _local_search_conflict(mask, conflict, recip_floats, deadline):
    """
    Local search: try all single swaps (remove one, add one) to find best improvement.
    Uses conflict masks for fast checking.
    """
    current = mask
    current_float = _recip_float_from_mask(current, recip_floats)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0.0
        best_remove = -1
        best_add = -1
        
        # Try all removals
        for remove in range(1, N + 1):
            if not (current & (1 << remove)):
                continue
            
            temp_mask = current & ~(1 << remove)
            
            # Try all additions
            for add in range(1, N + 1):
                if current & (1 << add):
                    continue
                if not _can_add_bitset(add, temp_mask, conflict):
                    continue
                
                delta = recip_floats[add] - recip_floats[remove]
                if delta > best_delta:
                    best_delta = delta
                    best_remove = remove
                    best_add = add
        
        if best_delta > 0 and best_remove != -1:
            current = current & ~(1 << best_remove)
            current |= (1 << best_add)
            current_float += best_delta
            improved = True
    
    return current

def _simulated_annealing(start_mask, conflict, recip_floats, deadline, seed):
    """
    Simulated annealing with bitmask operations.
    Moves: add one element, remove one element, or swap one element.
    """
    rng = random.Random(seed)
    current = start_mask
    T = 0.01
    alpha = 0.999
    best_mask = start_mask
    best_float = _recip_float_from_mask(start_mask, recip_floats)
    
    step = 0
    while time.time() < deadline and T > 1e-7:
        step += 1
        move_type = rng.random()
        
        if move_type < 0.3:
            # Remove a random element
            elements = [x for x in range(1, N + 1) if current & (1 << x)]
            if not elements:
                break
            remove = rng.choice(elements)
            new_mask = current & ~(1 << remove)
            delta = -recip_floats[remove]
        elif move_type < 0.6:
            # Add a random element
            candidates = [x for x in range(1, N + 1) if not (current & (1 << x))]
            if not candidates:
                break
            add = rng.choice(candidates)
            if not _can_add_bitset(add, current, conflict):
                T *= alpha
                continue
            new_mask = current | (1 << add)
            delta = recip_floats[add]
        else:
            # Swap: remove one, add one
            elements = [x for x in range(1, N + 1) if current & (1 << x)]
            if not elements:
                break
            remove = rng.choice(elements)
            temp_mask = current & ~(1 << remove)
            candidates = [x for x in range(1, N + 1) 
                         if not (current & (1 << x)) 
                         and _can_add_bitset(x, temp_mask, conflict)]
            if not candidates:
                T *= alpha
                continue
            add = rng.choice(candidates)
            new_mask = temp_mask | (1 << add)
            delta = recip_floats[add] - recip_floats[remove]
        
        # Acceptance criterion
        if delta > 0 or rng.random() < (2.71828 ** (delta / T)):
            current = new_mask
        
        if current_float + delta > best_float:
            best_float = current_float + delta
            best_mask = current
        else:
            # Recompute best_float if needed (this is a simplification)
            pass
        
        T *= alpha
    
    # Final check
    final_float = _recip_float_from_mask(best_mask, recip_floats)
    if final_float > best_float:
        best_float = final_float
    
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5.0  # 5s safety margin
    
    # Precompute
    conflict = _build_conflict_masks()
    recip_floats = [0.0] + [1.0 / x for x in range(1, N + 1)]
    
    # Initial solutions
    best_mask = _greedy_initial(conflict, recip_floats)
    best_val = _recip_sum_from_mask(best_mask)
    
    # Try several randomized greedy starts
    rng = random.Random(42)
    for i in range(20):
        if time.time() > deadline - 10:
            break
        candidate = _greedy_randomized(conflict, recip_floats, rng)
        val = _recip_sum_from_mask(candidate)
        if val > best_val:
            best_val = val
            best_mask = candidate
    
    # Local search from best
    if time.time() < deadline - 5:
        improved = _local_search_conflict(best_mask, conflict, recip_floats, deadline - 5)
        val = _recip_sum_from_mask(improved)
        if val > best_val:
            best_val = val
            best_mask = improved
    
    # Simulated annealing with multiple restarts
    sa_time_budget = max(5, deadline - time.time() - 3)
    n_sa_restarts = max(1, int(sa_time_budget / 5))
    
    for restart in range(n_sa_restarts):
        if time.time() > deadline - 3:
            break
        
        # Start SA from current best or a random perturbation
        if restart == 0:
            sa_start = best_mask
        else:
            # Perturb best: remove 2-3 random elements
            sa_start = best_mask
            elements = [x for x in range(1, N + 1) if sa_start & (1 << x)]
            for _ in range(min(3, len(elements))):
                if not elements:
                    break
                rem = rng.choice(elements)
                sa_start &= ~(1 << rem)
                elements.remove(rem)
        
        per_restart_deadline = min(deadline - 3, time.time() + max(2, sa_time_budget / n_sa_restarts))
        sa_result = _simulated_annealing(sa_start, conflict, recip_floats, per_restart_deadline, seed=100 + restart)
        
        val = _recip_sum_from_mask(sa_result)
        if val > best_val:
            best_val = val
            best_mask = sa_result
    
    # Final local search
    if time.time() < deadline - 2:
        improved = _local_search_conflict(best_mask, conflict, recip_floats, deadline - 1)
        val = _recip_sum_from_mask(improved)
        if val > best_val:
            best_val = val
            best_mask = improved
    
    # Convert mask to list
    result = [x for x in range(1, N + 1) if best_mask & (1 << x)]
    
    if not result:
        result = [1]
    
    return result
