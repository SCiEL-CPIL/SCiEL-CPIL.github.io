"""
Optimized search for maximum-cardinality reciprocal-dissociated subset of [1..22].
Uses greedy construction + randomized local search (simulated annealing).
"""
import time
import random

# LCM of 1..22
_LCM = 232792560

def _check_collision(sums, v):
    """Check if adding v to current sums causes any collision."""
    for s in sums:
        if (s + v) in sums:
            return True
    return False

def _add_element(sums, v):
    """Add element v to sums, return new sums set."""
    new_sums = sums | {s + v for s in sums}
    return new_sums

def _remove_element(sums, v, current_set_indices, idx):
    """
    Remove element at idx from the set. We need to rebuild sums from scratch
    for the remaining elements.
    """
    return None  # We'll rebuild instead

def _rebuild_sums(subset):
    """Rebuild all subset sums for a given subset."""
    sums = {0}
    for i in subset:
        v = _LCM // i
        new_sums = {s + v for s in sums}
        sums = sums | new_sums
    return sums

def _verify_dissociated(subset):
    """
    Return True if all reciprocal subset sums are distinct.
    Uses integer arithmetic with LCM.
    """
    if not subset:
        return True
    
    values = [_LCM // i for i in subset]
    
    sums = {0}
    
    for v in values:
        if _check_collision(sums, v):
            return False
        new_sums = [s + v for s in sums]
        sums.update(new_sums)
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Strict safety margin
    
    all_elements = list(range(1, 23))
    best_set = [1]  # Fallback
    best_size = 1
    
    random.seed(42)
    
    # Phase 1: Multiple greedy attempts to find a good initial solution
    while time.time() < deadline - 20.0:
        # Generate a random permutation
        perm = all_elements[:]
        random.shuffle(perm)
        
        # Occasionally try sorted orders
        r = random.random()
        if r < 0.15:
            perm.sort(reverse=True)  # Large numbers first (safer)
        elif r < 0.30:
            perm.sort()  # Small numbers first
        elif r < 0.45:
            perm.sort(key=lambda x: _LCM // x, reverse=True)  # By reciprocal value desc
        elif r < 0.60:
            perm.sort(key=lambda x: _LCM // x)  # By reciprocal value asc
        
        current_set = []
        current_sums = {0}
        
        # Incremental greedy construction
        for i in perm:
            if time.time() > deadline - 20.0:
                break
            
            v = _LCM // i
            
            if not _check_collision(current_sums, v):
                current_set.append(i)
                new_sums = [s + v for s in current_sums]
                current_sums.update(new_sums)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
        
        if best_size == 22:
            break
    
    # Phase 2: Randomized local search with simulated annealing
    # Try to improve on the best set found so far
    if best_size < 22 and time.time() < deadline - 5.0:
        # Initialize from best_set
        current_set = best_set[:]
        current_sums = _rebuild_sums(current_set)
        current_size = len(current_set)
        
        # Temperature for simulated annealing
        temp = 1.0
        cooling_rate = 0.999
        min_temp = 0.01
        
        in_set = set(current_set)
        out_set = set(all_elements) - in_set
        
        while time.time() < deadline - 5.0 and temp > min_temp:
            move_type = random.random()
            
            if move_type < 0.4 and out_set:
                # Try adding an element from outside
                candidates = list(out_set)
                random.shuffle(candidates)
                
                for i in candidates:
                    v = _LCM // i
                    if not _check_collision(current_sums, v):
                        # Accept the addition
                        current_set.append(i)
                        new_sums = [s + v for s in current_sums]
                        current_sums.update(new_sums)
                        in_set.add(i)
                        out_set.discard(i)
                        current_size += 1
                        
                        if current_size > best_size:
                            best_size = current_size
                            best_set = current_set[:]
                        
                        break  # Only add one element per iteration
                    else:
                        # Check if we should accept a collision (unlikely, but for SA)
                        # Actually, we can't accept collisions - they violate the constraint
                        pass
            
            elif move_type < 0.7 and in_set and current_size > 1:
                # Try removing an element from inside
                candidates = list(in_set)
                random.shuffle(candidates)
                
                for i in candidates:
                    # Remove i and check if we can add something else
                    current_set.remove(i)
                    in_set.discard(i)
                    out_set.add(i)
                    current_size -= 1
                    
                    # Rebuild sums after removal
                    current_sums = _rebuild_sums(current_set)
                    
                    # Try to add a different element
                    if out_set:
                        add_candidates = list(out_set - {i})
                        random.shuffle(add_candidates)
                        
                        added = False
                        for j in add_candidates:
                            v = _LCM // j
                            if not _check_collision(current_sums, v):
                                current_set.append(j)
                                new_sums = [s + v for s in current_sums]
                                current_sums.update(new_sums)
                                in_set.add(j)
                                out_set.discard(j)
                                current_size += 1
                                added = True
                                
                                if current_size > best_size:
                                    best_size = current_size
                                    best_set = current_set[:]
                                break
                        
                        if not added:
                            # Keep the removal if we couldn't replace
                            pass
                    break
            
            elif move_type < 0.9 and in_set and out_set:
                # Swap: remove one from in, add one from out
                remove_candidates = list(in_set)
                add_candidates = list(out_set)
                random.shuffle(remove_candidates)
                random.shuffle(add_candidates)
                
                swapped = False
                for i in remove_candidates:
                    for j in add_candidates:
                        # Try swapping i for j
                        current_set.remove(i)
                        in_set.discard(i)
                        out_set.add(i)
                        current_size -= 1
                        
                        # Rebuild sums
                        current_sums = _rebuild_sums(current_set)
                        
                        v = _LCM // j
                        if not _check_collision(current_sums, v):
                            current_set.append(j)
                            new_sums = [s + v for s in current_sums]
                            current_sums.update(new_sums)
                            in_set.add(j)
                            out_set.discard(j)
                            current_size += 1
                            swapped = True
                            
                            if current_size > best_size:
                                best_size = current_size
                                best_set = current_set[:]
                            break
                    
                    if swapped:
                        break
                
                if not swapped:
                    # Try just swapping without size change
                    for i in remove_candidates:
                        for j in add_candidates:
                            if i == j:
                                continue
                            # Remove i
                            current_set.remove(i)
                            in_set.discard(i)
                            out_set.add(i)
                            current_size -= 1
                            
                            current_sums = _rebuild_sums(current_set)
                            
                            v = _LCM // j
                            if not _check_collision(current_sums, v):
                                current_set.append(j)
                                new_sums = [s + v for s in current_sums]
                                current_sums.update(new_sums)
                                in_set.add(j)
                                out_set.discard(j)
                                current_size += 1
                                swapped = True
                                
                                if current_size > best_size:
                                    best_size = current_size
                                    best_set = current_set[:]
                                break
                        
                        if swapped:
                            break
            
            # Cool down
            temp *= cooling_rate
            
            if best_size == 22:
                break
    
    return best_set
