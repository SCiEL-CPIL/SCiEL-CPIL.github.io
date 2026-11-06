import time
import random
from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # Precompute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    weights = {s: L // s for s in range(1, 23)}

    # Fast pre-filter using modular arithmetic
    # If two subset sums are equal mod M, they could be equal (but not guaranteed)
    # Use multiple moduli for better discrimination
    moduli = [1000003, 1000033, 1000037]
    
    def is_dissociated_fast(S):
        """Quick check using modular arithmetic as a necessary condition."""
        if not S:
            return True
        w = [weights[s] for s in S]
        for mod in moduli:
            sums = {0}
            for val in w:
                v = val % mod
                new_sums = set()
                for s in sums:
                    ns = (s + v) % mod
                    if ns in sums or ns in new_sums:
                        return False
                    new_sums.add(ns)
                sums.update(new_sums)
        return True

    def is_dissociated(S):
        if not S:
            return True
        w = [weights[s] for s in S]
        sums = {0}
        for val in w:
            new_sums = set()
            for s in sums:
                ns = s + val
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    def try_add_set(base_set, additions, removals):
        """Try adding 'additions' and removing 'removals' from base_set."""
        new_set = [x for x in base_set if x not in removals] + additions
        if len(new_set) != len(set(new_set)):
            return None
        if is_dissociated_fast(new_set) and is_dissociated(new_set):
            return new_set
        return None

    best_set = []
    best_size = 0
    candidate_sets = []  # Keep multiple good candidates

    # Strategy 1: Greedy with random restarts
    for _ in range(80):
        if time.time() > deadline - 15:  # Reserve 15s for local search
            break
        
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        
        current_set = []
        for x in cand_order:
            new_set = current_set + [x]
            if is_dissociated_fast(new_set) and is_dissociated(new_set):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            candidate_sets = [list(current_set)]
        elif len(current_set) == best_size:
            candidate_sets.append(list(current_set))
            if len(candidate_sets) > 5:
                candidate_sets.pop(0)

    # Strategy 2: Local Search with multiple moves
    # Start from the best set
    current_best = list(best_set) if best_set else []
    
    # If we have multiple candidates, also try local search from them
    search_candidates = [list(best_set)] if best_set else []
    for cs in candidate_sets[1:]:
        if cs != best_set:
            search_candidates.append(list(cs))
    
    for cand in search_candidates:
        if time.time() > deadline - 2:
            break
        
        current = list(cand)
        if not current:
            continue
            
        stall_count = 0
        max_stall = 50
        
        while time.time() < deadline - 2 and stall_count < max_stall:
            improved = False
            missing = [x for x in range(1, 23) if x not in current]
            
            # Try to add a random missing element
            if missing:
                x = random.choice(missing)
                new_set = current + [x]
                if is_dissociated_fast(new_set) and is_dissociated(new_set):
                    current = new_set
                    improved = True
            
            # Try single swap: remove one, add another
            if not improved and len(current) > 1 and missing:
                x_remove = random.choice(current)
                x_add = random.choice(missing)
                if x_remove != x_add:
                    new_set = [x for x in current if x != x_remove] + [x_add]
                    if is_dissociated_fast(new_set) and is_dissociated(new_set):
                        current = new_set
                        improved = True
            
            # Try double swap: remove 2, add 2
            if not improved and len(current) > 2 and len(missing) >= 2:
                # Pick 2 to remove
                remove_candidates = random.sample(current, min(2, len(current)))
                # Pick 2 to add
                add_candidates = random.sample(missing, min(2, len(missing)))
                
                new_set = [x for x in current if x not in remove_candidates] + add_candidates
                if len(new_set) == len(set(new_set)):
                    if is_dissociated_fast(new_set) and is_dissociated(new_set):
                        current = new_set
                        improved = True
            
            # Try triple swap: remove 3, add 3 (more aggressive)
            if not improved and len(current) > 3 and len(missing) >= 3:
                remove_candidates = random.sample(current, min(3, len(current)))
                add_candidates = random.sample(missing, min(3, len(missing)))
                
                new_set = [x for x in current if x not in remove_candidates] + add_candidates
                if len(new_set) == len(set(new_set)):
                    if is_dissociated_fast(new_set) and is_dissociated(new_set):
                        current = new_set
                        improved = True
            
            if improved:
                stall_count = 0
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = list(current)
            else:
                stall_count += 1
        
        # Update best if this candidate improved
        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)

    if not best_set:
        return [1]
        
    return best_set
