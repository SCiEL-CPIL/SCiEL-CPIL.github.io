"""Combined program: fast integer LCM-based dissociation check + incremental backtracking search."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _is_dissociated_int(S):
    """Fast check: all subset sums of {1/x for x in S} distinct, via integer LCM."""
    if not S:
        return True
    L = 1
    for x in S:
        L = math.lcm(L, x)
    vals = [L // x for x in S]
    sums = {0}
    for v in vals:
        new = {s + v for s in sums}
        if new & sums:
            return False
        sums |= new
    return True

def _greedy_construct(universe):
    """
    Greedily construct a dissociated set.
    Tries to add elements in a specific order.
    """
    S = []
    L = 1
    sums = {0}
    for x in universe:
        new_lcm = math.lcm(L, x)
        scale = new_lcm // L
        delta = new_lcm // x
        
        valid = True
        if delta % scale == 0:
            diff = delta // scale
            if any((s + diff) in sums for s in sums):
                valid = False
        
        if valid:
            S.append(x)
            L = new_lcm
            if scale == 1:
                new_sums = set()
                for s in sums:
                    new_sums.add(s)
                    new_sums.add(s + delta)
            else:
                new_sums = set()
                for s in sums:
                    s_scaled = s * scale
                    new_sums.add(s_scaled)
                    new_sums.add(s_scaled + delta)
            sums = new_sums
    return S

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill
    universe = list(range(1, 23))
    
    # 1. Greedy Initialization to set a strong lower bound
    # Try a few greedy orders
    best = [1]
    best_size = 1
    
    greedy_orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        [x for x in range(1, 23) if x % 2 == 1] + [x for x in range(1, 23) if x % 2 == 0] # Odd first
    ]
    
    for order in greedy_orders:
        S_greedy = _greedy_construct(order)
        if len(S_greedy) > best_size:
            best = S_greedy
            best_size = len(S_greedy)
            
    # Fallback if greedy failed (shouldn't happen, [1] is always valid)
    if not best:
        best = [1]
        best_size = 1

    def backtrack(current, remaining, current_lcm, current_sums, current_size):
        nonlocal best, best_size
        
        if time.time() > deadline - 2:
            return
        
        # Prune: even if we add all remaining, can we beat best?
        if current_size + len(remaining) <= best_size:
            return
        
        if not remaining:
            if current_size > best_size:
                best = list(current)
                best_size = current_size
            return
        
        x = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include x
        new_lcm = math.lcm(current_lcm, x)
        scale = new_lcm // current_lcm
        delta = new_lcm // x
        
        valid = True
        if delta % scale == 0:
            diff = delta // scale
            if any((s + diff) in current_sums for s in current_sums):
                valid = False
        
        if valid:
            # Construct new sums
            if scale == 1:
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s)
                    new_sums.add(s + delta)
            else:
                new_sums = set()
                for s in current_sums:
                    s_scaled = s * scale
                    new_sums.add(s_scaled)
                    new_sums.add(s_scaled + delta)
            
            current.append(x)
            backtrack(current, rest, new_lcm, new_sums, current_size + 1)
            current.pop()
        
        # Option 2: exclude x
        backtrack(current, rest, current_lcm, current_sums, current_size)

    # 2. Backtracking with diverse orderings
    # We use the best_size from greedy to prune effectively.
    
    # Order 1: Descending
    backtrack([], sorted(universe, reverse=True), 1, {0}, 0)
    
    if time.time() > deadline - 15:
        if _is_dissociated_int(best):
            return sorted(best)
        return [1]

    # Order 2: Ascending
    backtrack([], sorted(universe), 1, {0}, 0)
    
    if time.time() > deadline - 15:
        if _is_dissociated_int(best):
            return sorted(best)
        return [1]

    # Order 3: Primes first
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
        
    primes = [x for x in range(1, 23) if is_prime(x)]
    composites = [x for x in range(1, 23) if not is_prime(x)]
    backtrack([], primes + composites, 1, {0}, 0)
    
    if time.time() > deadline - 10:
        if _is_dissociated_int(best):
            return sorted(best)
        return [1]

    # Order 4: Random shuffles, looping until time is nearly up
    random.seed(42)
    while time.time() < deadline - 2:
        order4 = list(universe)
        random.shuffle(order4)
        backtrack([], order4, 1, {0}, 0)
        
    if not best or not _is_dissociated_int(best):
        return [1]
        
    return sorted(best)
# EVOLVE-BLOCK-END
