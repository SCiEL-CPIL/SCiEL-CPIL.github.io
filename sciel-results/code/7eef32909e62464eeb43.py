import time
import random
from fractions import Fraction

_N = 256
_KILL = 90.0
_MARGIN = 5.0

# Precompute reciprocals for fast comparison
_RECIP = [0.0] + [1.0 / i for i in range(1, _N + 1)]


def _get_blocked(S):
    """Compute the blocked set for a given set S.
    blocked contains all x that would form a 3-AP with two elements of S.
    """
    blocked = set()
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            # a, b, x => x = 2b - a
            x = 2 * b - a
            if 1 <= x <= _N:
                blocked.add(x)
            # a, x, b => x = (a+b)//2 if even
            s = a + b
            if s % 2 == 0:
                x = s // 2
                if 1 <= x <= _N:
                    blocked.add(x)
            # x, a, b => x = 2a - b
            x = 2 * a - b
            if 1 <= x <= _N:
                blocked.add(x)
    return blocked


def _add_to_blocked(blocked, S, x):
    """Update blocked set when adding x to S. Mutates blocked in place."""
    for a in S:
        if a == x:
            continue
        # x, a, 2a-x
        q = 2 * a - x
        if 1 <= q <= _N:
            blocked.add(q)
        # a, x, 2x-a
        q = 2 * x - a
        if 1 <= q <= _N:
            blocked.add(q)
        # (a+x)/2 if even
        s = a + x
        if s % 2 == 0:
            q = s // 2
            if 1 <= q <= _N:
                blocked.add(q)


def _greedy_ascending():
    """Greedy: add elements in order of smallest x (largest reciprocal) if valid."""
    S = set()
    blocked = set()
    for x in range(1, _N + 1):
        if x not in blocked:
            S.add(x)
            _add_to_blocked(blocked, S, x)
    return S


def _greedy_descending():
    """Greedy: add elements in order of largest x (smallest reciprocal) if valid."""
    S = set()
    blocked = set()
    for x in range(_N, 0, -1):
        if x not in blocked:
            S.add(x)
            _add_to_blocked(blocked, S, x)
    return S


def _random_init(rng, size_range=(30, 80)):
    """Generate a random valid initial set."""
    S = set()
    blocked = set()
    elements = list(range(1, _N + 1))
    rng.shuffle(elements)
    
    target_size = rng.randint(*size_range)
    for x in elements:
        if len(S) >= target_size:
            break
        if x not in blocked:
            S.add(x)
            _add_to_blocked(blocked, S, x)
    return S


def _simulated_annealing(S_init, deadline, rng):
    """Simulated Annealing to optimize the set."""
    S = set(S_init)
    if not S:
        S.add(1)
    
    blocked = _get_blocked(S)
    weight = sum(_RECIP[x] for x in S)
    
    T = 0.1
    alpha = 0.9995
    min_T = 1e-5
    
    # Precompute list of all elements for random sampling
    all_elements = list(range(1, _N + 1))
    
    while time.time() < deadline and T > min_T:
        T *= alpha
        
        move_type = rng.random()
        
        if move_type < 0.4:
            # Add move
            # Pick a random element not in S
            # To avoid infinite loops if S is full, pick from a random subset
            candidates = [x for x in all_elements if x not in S]
            if not candidates:
                continue
            x = rng.choice(candidates)
            
            if x in blocked:
                continue
                
            delta_w = _RECIP[x]
            
            # Always accept improvement
            # If not improvement, accept with prob exp(delta/T)
            if delta_w > 0 or rng.random() < pow(2.718281828, delta_w / T):
                S.add(x)
                weight += delta_w
                _add_to_blocked(blocked, S, x)
                
        elif move_type < 0.7:
            # Remove move
            if not S:
                continue
            y = rng.choice(list(S))
            delta_w = -_RECIP[y]
            
            if delta_w > 0 or rng.random() < pow(2.718281828, delta_w / T):
                S.remove(y)
                weight += delta_w
                # Recompute blocked
                blocked = _get_blocked(S)
                
        else:
            # Swap move
            if not S:
                continue
            y = rng.choice(list(S))
            candidates = [x for x in all_elements if x not in S]
            if not candidates:
                continue
            x = rng.choice(candidates)
            
            # Check validity of x in S \ {y}
            # First check if x is in blocked(S). If not, it is definitely valid in S\{y}.
            # If x is in blocked(S), it might be valid in S\{y} if the conflict was with y.
            # We need to check precisely.
            
            # Optimization: If x is not in blocked(S), we can proceed without recompute.
            # If x is in blocked(S), we must recompute blocked for S\{y} to check.
            
            temp_S = S - {y}
            
            if x not in blocked:
                # x is not blocked by S, so it is not blocked by S\{y}
                is_valid = True
                new_blocked = None # We will update incrementally later if accepted
            else:
                # x is blocked by S. Check if it is still blocked by S\{y}
                # We need to recompute blocked for S\{y}
                temp_blocked = _get_blocked(temp_S)
                is_valid = (x not in temp_blocked)
                new_blocked = temp_blocked if is_valid else None
            
            if not is_valid:
                continue
                
            delta_w = _RECIP[x] - _RECIP[y]
            
            if delta_w > 0 or rng.random() < pow(2.718281828, delta_w / T):
                # Accept move
                S = temp_S
                S.add(x)
                weight += delta_w
                
                if new_blocked is not None:
                    # We already have the blocked set for S\{y}
                    blocked = new_blocked
                    _add_to_blocked(blocked, S, x)
                else:
                    # x was not in blocked(S), so we can just update from old blocked?
                    # No, blocked(S) contains blocks from y.
                    # We removed y, so some blocks might be gone.
                    # But we didn't recompute.
                    # However, if x was NOT in blocked(S), it means no pair in S blocks x.
                    # So no pair in S\{y} blocks x.
                    # But we need the correct `blocked` set for future moves.
                    # Since we didn't recompute, we must recompute now to be safe.
                    blocked = _get_blocked(S)
                    
    return S


def _reciprocal_sum_exact(S):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total


def construct():
    start_time = time.time()
    deadline = start_time + _KILL - _MARGIN
    rng = random.Random(20240517)
    
    best_set = None
    best_weight = -1.0
    
    # Generate diverse initial solutions
    inits = []
    
    # 1. Greedy ascending
    S1 = _greedy_ascending()
    inits.append(S1)
    
    # 2. Greedy descending
    S2 = _greedy_descending()
    inits.append(S2)
    
    # 3-10. Random initializations
    for i in range(8):
        S_rand = _random_init(rng, size_range=(20, 100))
        inits.append(S_rand)
    
    # Run Simulated Annealing on each initial solution
    for S_init in inits:
        if time.time() > deadline:
            break
        
        result = _simulated_annealing(S_init, deadline, rng)
        w = sum(_RECIP[x] for x in result)
        
        if w > best_weight:
            best_weight = w
            best_set = result.copy()
            
        # Additional local search polish on the SA result
        # (Optional, but SA should be good enough)
    
    # Additional restarts from best solution with perturbations
    for restart in range(20):
        if time.time() > deadline:
            break
        
        S = set(best_set)
        
        # Randomly remove some elements
        n_remove = rng.randint(1, min(5, len(S)))
        to_remove = rng.sample(list(S), n_remove)
        for x in to_remove:
            S.discard(x)
        
        # Add some random elements
        available = [x for x in range(1, _N + 1) if x not in S]
        if available:
            n_add = rng.randint(0, min(3, len(available)))
            to_add = rng.sample(available, n_add)
            for x in to_add:
                if x not in _get_blocked(S):
                    S.add(x)
        
        result = _simulated_annealing(S, deadline, rng)
        w = sum(_RECIP[x] for x in result)
        
        if w > best_weight:
            best_weight = w
            best_set = result.copy()
    
    return list(best_set)
