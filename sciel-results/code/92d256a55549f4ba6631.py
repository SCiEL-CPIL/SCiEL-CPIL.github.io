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

    def is_dissociated(S):
        if not S:
            return True
        w = sorted([weights[s] for s in S], reverse=True)
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

    def is_dissociated_incremental(S, base_sums):
        """Check if adding elements of S to a set with existing sums base_sums is dissociated."""
        w = sorted([weights[s] for s in S], reverse=True)
        current_sums = set(base_sums)
        for val in w:
            new_sums = set()
            for s in current_sums:
                ns = s + val
                if ns in current_sums or ns in new_sums:
                    return None  # collision
                new_sums.add(ns)
            current_sums.update(new_sums)
        return current_sums

    best_set = []
    best_size = 0

    # Strategy 1: Multiple greedy passes with different orderings
    for _ in range(80):
        if time.time() > deadline - 15:  # Reserve 15s for local search
            break
        
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        
        current_set = []
        current_sums = {0}
        for x in cand_order:
            w = weights[x]
            new_sums = set()
            collision = False
            for s in current_sums:
                ns = s + w
                if ns in current_sums or ns in new_sums:
                    collision = True
                    break
                new_sums.add(ns)
            if not collision:
                current_set.append(x)
                current_sums.update(new_sums)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    # Strategy 2: Local Search with 1-swap, 2-swap, and add moves
    current_best = list(best_set)
    best_sums = {0}
    for x in current_best:
        w = weights[x]
        new_sums = set()
        for s in best_sums:
            new_sums.add(s + w)
        best_sums.update(new_sums)

    # Track recent failures to avoid cycling
    failed_moves = set()
    failed_moves_ttl = {}  # move -> time when it failed

    while time.time() < deadline - 2:
        if not current_best:
            break
            
        improved = False
        missing = [x for x in range(1, 23) if x not in current_best]
        
        # Try to add a random missing element
        if missing:
            # Try all missing elements, pick first that works
            for x in missing:
                move_key = ('add', x)
                if move_key in failed_moves and time.time() - failed_moves_ttl.get(move_key, 0) < 5:
                    continue
                w = weights[x]
                collision = False
                for s in best_sums:
                    if s + w in best_sums:
                        collision = True
                        break
                if not collision:
                    new_sums = best_sums | {s + w for s in best_sums}
                    current_best.append(x)
                    best_sums = new_sums
                    improved = True
                    if len(current_best) > best_size:
                        best_size = len(current_best)
                        best_set = list(current_best)
                    break
                else:
                    failed_moves.add(move_key)
                    failed_moves_ttl[move_key] = time.time()
        
        # Try 1-swap: remove one, add another
        if not improved and len(current_best) > 1 and missing:
            for x_remove in current_best:
                for x_add in missing:
                    if x_remove == x_add:
                        continue
                    move_key = ('swap1', x_remove, x_add)
                    if move_key in failed_moves and time.time() - failed_moves_ttl.get(move_key, 0) < 5:
                        continue
                    
                    # Compute sums without x_remove
                    sums_without = {0}
                    for x in current_best:
                        if x == x_remove:
                            continue
                        w = weights[x]
                        new_sums = set()
                        for s in sums_without:
                            ns = s + w
                            if ns in sums_without or ns in new_sums:
                                # This shouldn't happen if current_best is dissociated
                                break
                            new_sums.add(ns)
                        sums_without.update(new_sums)
                    
                    # Check if adding x_add to sums_without is dissociated
                    w_add = weights[x_add]
                    collision = False
                    for s in sums_without:
                        if s + w_add in sums_without:
                            collision = True
                            break
                    if not collision:
                        new_sums = sums_without | {s + w_add for s in sums_without}
                        current_best = [x for x in current_best if x != x_remove] + [x_add]
                        best_sums = new_sums
                        improved = True
                        break
                    else:
                        failed_moves.add(move_key)
                        failed_moves_ttl[move_key] = time.time()
                if improved:
                    break
        
        # Try 2-swap: remove two, add two (only if we're stuck)
        if not improved and len(current_best) > 2 and len(missing) >= 2:
            # Sample a few 2-swap moves
            for _ in range(10):
                if time.time() > deadline - 2:
                    break
                if len(current_best) < 2 or len(missing) < 2:
                    break
                x_remove1, x_remove2 = random.sample(current_best, 2)
                x_add1, x_add2 = random.sample(missing, 2)
                
                move_key = ('swap2', tuple(sorted([x_remove1, x_remove2])), tuple(sorted([x_add1, x_add2])))
                if move_key in failed_moves and time.time() - failed_moves_ttl.get(move_key, 0) < 5:
                    continue
                
                # Compute sums without x_remove1 and x_remove2
                sums_without = {0}
                for x in current_best:
                    if x == x_remove1 or x == x_remove2:
                        continue
                    w = weights[x]
                    new_sums = set()
                    for s in sums_without:
                        ns = s + w
                        if ns in sums_without or ns in new_sums:
                            break
                        new_sums.add(ns)
                    sums_without.update(new_sums)
                
                # Check if adding x_add1 and x_add2 is dissociated
                w1 = weights[x_add1]
                w2 = weights[x_add2]
                collision = False
                for s in sums_without:
                    if s + w1 in sums_without:
                        collision = True
                        break
                if not collision:
                    sums_after1 = sums_without | {s + w1 for s in sums_without}
                    for s in sums_after1:
                        if s + w2 in sums_after1:
                            collision = True
                            break
                if not collision:
                    new_sums = sums_after1 | {s + w2 for s in sums_after1}
                    current_best = [x for x in current_best if x != x_remove1 and x != x_remove2] + [x_add1, x_add2]
                    best_sums = new_sums
                    improved = True
                    break
                else:
                    failed_moves.add(move_key)
                    failed_moves_ttl[move_key] = time.time()
        
        # If no improvement after many attempts, do a random restart from best_set
        if not improved and random.random() < 0.1:
            if len(best_set) > len(current_best):
                current_best = list(best_set)
                best_sums = {0}
                for x in current_best:
                    w = weights[x]
                    new_sums = set()
                    for s in best_sums:
                        new_sums.add(s + w)
                    best_sums.update(new_sums)
            # Clear some failed moves
            if len(failed_moves) > 100:
                failed_moves.clear()
                failed_moves_ttl.clear()

    if not best_set:
        return [1]
        
    return best_set
