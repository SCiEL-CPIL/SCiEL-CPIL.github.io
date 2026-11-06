"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import time
import random

def _compute_sums(selected):
    """Compute all pairwise sums a+b for a<=b in selected."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + selected[j])
    return sums

def _is_sidon(selected):
    """Check if selected is a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        if 2 * a in sums:
            return False
        sums.add(2 * a)
        for j in range(i + 1, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected):
    """Check if selected is inclusion-maximal Sidon in [1,N]."""
    if not selected:
        return False
    selected_set = set(selected)
    sums = _compute_sums(selected)
    
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Adding x creates a duplicate iff 2x or x+s is already in sums
        if 2 * x in sums:
            continue
        conflict = False
        for s in selected:
            if x + s in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _greedy_construct(order):
    """Greedy Sidon set construction following the given order."""
    selected = []
    sums = set()
    for x in order:
        # Check if x can be added
        new_sums = [x + a for a in selected] + [2 * x]
        # Check for internal conflicts in new_sums
        if len(set(new_sums)) != len(new_sums):
            continue
        # Check conflicts with existing sums
        if sums.intersection(new_sums):
            continue
        
        selected.append(x)
        sums.update(new_sums)
    return selected

def _local_search_reduce(selected, deadline):
    """
    Try to remove elements while maintaining maximality.
    If removal fails, try swaps to escape local minima.
    """
    current = list(selected)
    current_set = set(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try removals
        for x in list(current):
            if time.time() > deadline:
                break
            
            new_set = [y for y in current if y != x]
            
            # A subset of a Sidon set is always Sidon
            # Check if it is still Maximal
            if _is_maximal(new_set):
                current = new_set
                current_set = set(new_set)
                improved = True
                break  # Restart scan after a removal
        
        # Phase 2: Try swaps if no removal worked
        if not improved and time.time() < deadline - 1.0:
            # Try to swap x with y
            # To save time, only try a few random swaps or a limited search
            # We want to find a y such that (S - {x}) U {y} is Sidon and Maximal.
            # This is expensive. Let's limit the number of swap attempts per iteration.
            
            # Pick a random x to remove
            if current:
                x = random.choice(current)
                base_set = [y for y in current if y != x]
                base_sums = _compute_sums(base_set)
                
                # Find candidates y that can be added to base_set
                # y must not be in base_set
                # y must not create conflicts with base_sums
                candidates = []
                for y in range(1, N + 1):
                    if y in current_set:
                        continue
                    # Check if y can be added to base_set
                    if 2 * y in base_sums:
                        continue
                    conflict = False
                    for s in base_set:
                        if y + s in base_sums:
                            conflict = True
                            break
                    if not conflict:
                        candidates.append(y)
                
                # Try a few candidates
                random.shuffle(candidates)
                for y in candidates[:20]: # Limit to 20 random candidates
                    if time.time() > deadline:
                        break
                    new_set = base_set + [y]
                    if _is_maximal(new_set):
                        current = new_set
                        current_set = set(new_set)
                        improved = True
                        break
                        
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    best = None
    best_size = N + 1
    
    # Strategy 1: Ascending greedy (baseline)
    order = list(range(1, N + 1))
    selected = _greedy_construct(order)
    if len(selected) < best_size:
        best = selected
        best_size = len(selected)
        
    # Local search to reduce size
    if time.time() < deadline:
        reduced = _local_search_reduce(best, deadline)
        if len(reduced) < best_size:
            best = reduced
            best_size = len(reduced)
            
    # Strategy 2: Try different random orders
    if time.time() < deadline - 5.0:
        for trial in range(100):
            if time.time() > deadline:
                break
            
            rng = random.Random(trial)
            order = list(range(1, N + 1))
            rng.shuffle(order)
            
            selected = _greedy_construct(order)
            
            if len(selected) < best_size:
                best = selected
                best_size = len(selected)
                
            # Refine this candidate
            if time.time() < deadline - 2.0:
                reduced = _local_search_reduce(selected, deadline)
                if len(reduced) < best_size:
                    best = reduced
                    best_size = len(reduced)
                    
    return best
# EVOLVE-BLOCK-END
