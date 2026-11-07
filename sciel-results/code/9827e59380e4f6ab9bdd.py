"""Large sets with distinct subset products."""
# EVOLVE-BLOCK-START
import time
import random

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Uses greedy construction with local search to maximize set size.
    """
    candidates = list(range(2, 41))
    
    def can_add(prods, x):
        """Check if x can be added to the set with product set 'prods' without creating collisions."""
        for p in prods:
            if p % x == 0:
                q = p // x
                if q in prods:
                    return False
        return True
    
    def try_add(S, prods, x):
        """Try to add x. Return (new_S, new_prods) or (None, None)."""
        if x in S:
            return None, None
        if not can_add(prods, x):
            return None, None
        new_prods = set(prods)
        for p in prods:
            new_prods.add(p * x)
        new_S = S | {x}
        return new_S, new_prods
    
    def greedy_construct(ordering):
        """Build a set greedily in the given order."""
        S = set()
        prods = {1}
        for x in ordering:
            if time.time() > _DEADLINE:
                break
            new_S, new_prods = try_add(S, prods, x)
            if new_S is not None:
                S = new_S
                prods = new_prods
        return S, prods
    
    def local_search(S, prods, max_iterations=5000):
        """Improve set via add, swap, and remove operations."""
        best_S = S
        best_prods = prods
        best_size = len(S)
        remaining = [x for x in candidates if x not in S]
        
        for _ in range(max_iterations):
            if time.time() > _DEADLINE:
                break
            
            improved = False
            
            # Try to add one element
            for x in remaining:
                new_S, new_prods = try_add(S, prods, x)
                if new_S is not None:
                    S = new_S
                    prods = new_prods
                    remaining = [y for y in remaining if y != x]
                    improved = True
                    break
            
            if improved:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S
                    best_prods = prods
                continue
            
            # Try swaps: remove one, add one or two
            S_list = list(S)
            random.shuffle(S_list)
            found_swap = False
            for remove_x in S_list:
                # Remove remove_x
                S_minus = S - {remove_x}
                prods_minus = set()
                for p in prods:
                    if p % remove_x == 0:
                        prods_minus.add(p // remove_x)
                    else:
                        prods_minus.add(p)
                
                # Try adding each remaining element
                for add_x in remaining:
                    new_S, new_prods = try_add(S_minus, prods_minus, add_x)
                    if new_S is not None and len(new_S) > len(S):
                        S = new_S
                        prods = new_prods
                        remaining = [y for y in remaining if y != add_x]
                        found_swap = True
                        break
                    elif new_S is not None and len(new_S) == len(S):
                        # Same size swap, might help for future additions
                        S = new_S
                        prods = new_prods
                        remaining = [y for y in remaining if y != add_x]
                        found_swap = True
                        break
                
                if found_swap:
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S
                        best_prods = prods
                    break
            
            if not found_swap:
                # Try double swaps: remove two, add one or two
                S_list = list(S)
                random.shuffle(S_list)
                found_double = False
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        rx1, rx2 = S_list[i], S_list[j]
                        S_minus = S - {rx1, rx2}
                        prods_minus = set()
                        for p in prods:
                            q = p
                            if q % rx1 == 0:
                                q = q // rx1
                            if q % rx2 == 0:
                                q = q // rx2
                            prods_minus.add(q)
                        
                        # Try adding each remaining element
                        for add_x in remaining:
                            new_S, new_prods = try_add(S_minus, prods_minus, add_x)
                            if new_S is not None and len(new_S) > len(S):
                                S = new_S
                                prods = new_prods
                                remaining = [y for y in remaining if y != add_x]
                                found_double = True
                                break
                        
                        if found_double:
                            break
                    if found_double:
                        break
                
                if found_double:
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S
                        best_prods = prods
            
            if not found_swap and not found_double:
                # No improvement possible with single/double swaps, try random restarts
                break
        
        return best_S, best_prods
    
    best_size = 0
    best_S = set()
    best_prods = {1}
    
    # Multiple strategies with random restarts
    num_restarts = 0
    
    while time.time() < _DEADLINE - 3:
        num_restarts += 1
        
        # Different ordering strategies
        strategy = num_restarts % 5
        if strategy == 0:
            ordering = sorted(candidates, key=lambda x: x)
        elif strategy == 1:
            ordering = sorted(candidates, key=lambda x: -x)
        elif strategy == 2:
            ordering = sorted(candidates, key=lambda x: (x % 2, x))
        elif strategy == 3:
            ordering = candidates[:]
            random.shuffle(ordering)
        else:
            # Sort by number of prime factors
            def num_prime_factors(n):
                count = 0
                d = 2
                while d * d <= n:
                    while n % d == 0:
                        count += 1
                        n //= d
                    d += 1
                if n > 1:
                    count += 1
                return count
            ordering = sorted(candidates, key=lambda x: (num_prime_factors(x), x))
        
        S, prods = greedy_construct(ordering)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            best_prods = prods
        
        # Local search refinement
        if time.time() < _DEADLINE - 3:
            S2, prods2 = local_search(S, prods)
            if len(S2) > best_size:
                best_size = len(S2)
                best_S = S2
                best_prods = prods2
        
        if best_size >= 18:
            break
    
    # Final verification
    if best_S:
        prods_check = {1}
        valid = True
        for x in sorted(best_S):
            new_prods = set()
            for p in prods_check:
                np = p * x
                if np in prods_check:
                    valid = False
                    break
                new_prods.add(np)
            if not valid:
                break
            prods_check.update(new_prods)
        
        if not valid:
            # Fallback: return empty set (shouldn't happen)
            best_S = set()
    
    return [int(x) for x in best_S]
# EVOLVE-BLOCK-END
