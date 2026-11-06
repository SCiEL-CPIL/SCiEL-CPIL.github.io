"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _validate(a):
    """Check that a is a valid Sidon set containing {1,2,5,11} with elements in 1..284."""
    if len(a) != len(set(a)):
        return False
    for x in a:
        if x < 1 or x > 284:
            return False
    for req in (1, 2, 5, 11):
        if req not in a:
            return False
    # Check all pairwise sums a+b (a<=b) are distinct
    sums = set()
    n = len(a)
    for i in range(n):
        for j in range(i, n):
            s = a[i] + a[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_initial():
    """Greedy construction to get a valid starting set."""
    a = [1, 2, 5, 11]
    sums = set()
    for i, x in enumerate(a):
        for y in a[i:]:
            sums.add(x + y)
    for x in range(12, 285):
        new = {x + y for y in a} | {2 * x}
        if len(new) == len(a) + 1 and not new & sums:
            a.append(x)
            sums |= new
    return sorted(a)

def _can_add(a, sums, x):
    """Check if x can be added to set a with current sums."""
    if x in a:
        return False
    new_sums = {x + y for y in a} | {2 * x}
    if len(new_sums) != len(a) + 1:
        return False
    if new_sums & sums:
        return False
    return True

def _simulated_annealing(start_set, deadline):
    """SA on the set: try adding/removing/swapping elements."""
    a = sorted(start_set)
    sums = set()
    for i, x in enumerate(a):
        for j in range(i, len(a)):
            sums.add(a[i] + a[j])
    
    best_a = list(a)
    best_size = len(a)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    random.shuffle(a)
    a = sorted(a)
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        if len(a) > best_size:
            best_a = list(a)
            best_size = len(a)
        
        # Choose a random move: add, remove, or swap
        move_type = random.random()
        if move_type < 0.4:
            # Try to add a random element
            candidates = [x for x in range(1, 285) if x not in a]
            if not candidates:
                T *= cooling
                continue
            x = random.choice(candidates)
            if _can_add(a, sums, x):
                a.append(x)
                a.sort()
                new_sums = {x + y for y in a[:-1]} | {2 * x}
                sums |= new_sums
            else:
                # Accept with probability exp(-delta/T) where delta=1 (worse)
                if random.random() < 2.718281828 / (1 + T):
                    # Force add by removing a conflicting element
                    pass  # Skip for simplicity
        elif move_type < 0.7:
            # Try to remove a random element
            if len(a) > 4:
                idx = random.randint(0, len(a) - 1)
                x = a[idx]
                # Check if removing x is beneficial (frees up sums)
                # Just remove it
                removed_sums = set()
                for y in a:
                    if y != x:
                        removed_sums.add(x + y)
                removed_sums.add(2 * x)
                a.pop(idx)
                sums -= removed_sums
                # With some probability, don't add it back
                if random.random() > 0.5:
                    continue
                else:
                    a.insert(idx, x)
                    a.sort()
                    sums |= removed_sums
        else:
            # Swap: remove one, try to add another
            if len(a) > 4:
                idx = random.randint(0, len(a) - 1)
                x = a[idx]
                removed_sums = set()
                for y in a:
                    if y != x:
                        removed_sums.add(x + y)
                removed_sums.add(2 * x)
                a.pop(idx)
                sums -= removed_sums
                # Try to add a random candidate
                candidates = [c for c in range(1, 285) if c not in a]
                if candidates:
                    new_x = random.choice(candidates)
                    temp_a = list(a)
                    temp_sums = set(sums)
                    if _can_add(temp_a, temp_sums, new_x):
                        a.append(new_x)
                        a.sort()
                        new_sums = {new_x + y for y in a[:-1]} | {2 * new_x}
                        sums |= new_sums
                    else:
                        # Restore
                        a.insert(idx, x)
                        a.sort()
                        sums |= removed_sums
                else:
                    a.insert(idx, x)
                    a.sort()
                    sums |= removed_sums
        
        T *= cooling
    
    if len(a) > best_size:
        best_a = list(a)
        best_size = len(a)
    
    return sorted(best_a)

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Get initial greedy solution
    initial = _greedy_initial()
    best = list(initial)
    best_size = len(initial)
    
    # Multiple restarts with different random seeds
    random.seed(42)
    start_time = time.time()
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Create a perturbed starting point
        start = list(best)
        # Randomly remove a few elements to create diversity
        n_remove = random.randint(0, min(3, len(start) - 4))
        if n_remove > 0 and len(start) > 4:
            to_remove = set(random.sample(start, n_remove))
            to_remove = {x for x in to_remove if x not in (1, 2, 5, 11)}
            if to_remove:
                start = [x for x in start if x not in to_remove]
        
        # Run SA
        result = _simulated_annealing(start, deadline)
        
        if len(result) > best_size:
            best = result
            best_size = len(result)
            # Validate
            if not _validate(best):
                best = list(initial)
                best_size = len(initial)
        
        # If we've found a large set, try more targeted improvements
        if best_size >= 19:
            # Try to fill in with greedy
            a = list(best)
            sums = set()
            for i, x in enumerate(a):
                for j in range(i, len(a)):
                    sums.add(a[i] + a[j])
            for x in range(1, 285):
                if x not in a and _can_add(a, sums, x):
                    a.append(x)
                    sums |= {x + y for y in a[:-1]} | {2 * x}
            if len(a) > best_size:
                if _validate(a):
                    best = a
                    best_size = len(a)
    
    # Final validation
    if not _validate(best):
        best = initial
    
    return sorted(best)
# EVOLVE-BLOCK-END
