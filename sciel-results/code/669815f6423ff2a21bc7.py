import time
import random

N = 256

def _can_add(x, A, sums):
    """Check if x can be added to A while maintaining Sidon property."""
    if 2 * x in sums:
        return False
    for a in A:
        if x + a in sums:
            return False
    return True

def _get_new_sums(x, A):
    """Get the set of new sums introduced by adding x to A."""
    new_sums = {2 * x}
    for a in A:
        new_sums.add(x + a)
    return new_sums

def _count_blocked(x, A, sums, unblocked):
    """
    Count how many elements in 'unblocked' (excluding x) would become blocked
    if x were added to A.
    """
    new_sums = _get_new_sums(x, A)
    all_sums = sums | new_sums
    new_A = A + [x]
    
    blocked_count = 0
    for y in unblocked:
        if y == x:
            continue
        # Check if y is blocked by the new configuration
        # y is blocked if 2y in all_sums OR (y+a in all_sums for some a in new_A)
        if 2 * y in all_sums:
            blocked_count += 1
            continue
        for a in new_A:
            if y + a in all_sums:
                blocked_count += 1
                break
    return blocked_count

def _greedy_max_coverage(deadline):
    """
    Construct a maximal Sidon set by greedily picking the element
    that blocks the most other unblocked elements.
    """
    A = []
    sums = set()
    unblocked = set(range(1, N + 1))
    
    while unblocked and time.time() < deadline:
        best_x = None
        max_blocked = -1
        best_new_sums = None
        best_A = None
        
        # Candidate selection: iterate through unblocked elements
        for x in list(unblocked):
            if time.time() > deadline:
                break
            
            if not _can_add(x, A, sums):
                continue
            
            # Calculate blocking power
            blocked = _count_blocked(x, A, sums, unblocked)
            
            # Tie-breaking: random to explore different topologies
            if blocked > max_blocked or (blocked == max_blocked and (best_x is None or random.random() < 0.5)):
                max_blocked = blocked
                best_x = x
                best_new_sums = _get_new_sums(x, A)
                best_A = A + [x]
        
        if best_x is None:
            break
        
        # Commit the choice
        A = best_A
        sums.update(best_new_sums)
        unblocked.discard(best_x)
        
        # Update unblocked: remove any elements that are now blocked
        new_unblocked = set()
        for y in unblocked:
            if _can_add(y, A, sums):
                new_unblocked.add(y)
        unblocked = new_unblocked
        
    return A

def _local_search_reduce(A, deadline):
    """
    Try to reduce the size of a maximal Sidon set by finding elements
    whose removal still allows the set to be maximal (or can be made maximal
    with fewer additions).
    
    Strategy: For each element x in A, temporarily remove it and check if
    the resulting set is still maximal. If not, try to add back only the
    minimum needed elements. If we can achieve a smaller total size, accept.
    """
    A = list(A)
    if not A:
        return A
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        A_sorted = sorted(A)
        
        for i, x in enumerate(A_sorted):
            if time.time() > deadline:
                return A
            
            # Remove x
            A_without = [a for a in A_sorted if a != x]
            
            # Compute sums without x
            sums_without = set()
            for a in A_without:
                for b in A_without:
                    if a <= b:
                        sums_without.add(a + b)
            
            # Check which elements are now unblocked
            unblocked = set()
            for y in range(1, N + 1):
                if y in A_without:
                    continue
                if _can_add(y, A_without, sums_without):
                    unblocked.add(y)
            
            if not unblocked:
                # Removing x makes the set still maximal - great!
                A = A_without
                improved = True
                break
            
            # Try to greedily add back minimal elements to make it maximal
            # Use a greedy approach that picks elements with max blocking
            A_temp = list(A_without)
            sums_temp = set(sums_without)
            unblocked_temp = set(unblocked)
            
            # Greedily add elements to make maximal
            while unblocked_temp and time.time() < deadline:
                best_y = None
                max_blocked = -1
                
                for y in list(unblocked_temp):
                    if not _can_add(y, A_temp, sums_temp):
                        continue
                    
                    blocked = _count_blocked(y, A_temp, sums_temp, unblocked_temp)
                    
                    if blocked > max_blocked or (blocked == max_blocked and (best_y is None or random.random() < 0.5)):
                        max_blocked = blocked
                        best_y = y
                
                if best_y is None:
                    break
                
                A_temp.append(best_y)
                sums_temp.update(_get_new_sums(best_y, A_temp[:-1]))
                unblocked_temp.discard(best_y)
                
                # Update unblocked
                new_unblocked = set()
                for y in unblocked_temp:
                    if _can_add(y, A_temp, sums_temp):
                        new_unblocked.add(y)
                unblocked_temp = new_unblocked
            
            # If we ended up with a smaller set, accept
            if len(A_temp) < len(A):
                A = A_temp
                improved = True
                break
    
    return A

def construct():
    start_time = time.time()
    # 60s limit, leave 5s margin
    deadline = start_time + 55.0
    
    best_A = None
    best_size = N + 1
    
    # Run multiple trials of the greedy max-coverage algorithm
    for trial in range(80):
        if time.time() > deadline:
            break
        
        A = _greedy_max_coverage(deadline)
        
        if not A:
            continue
        
        # Try to reduce the set
        A_reduced = _local_search_reduce(A, deadline)
        
        if A_reduced and len(A_reduced) < best_size:
            best_size = len(A_reduced)
            best_A = A_reduced
    
    if best_A is None:
        # Fallback: simple greedy
        A = []
        sums = set()
        for x in range(1, N + 1):
            if _can_add(x, A, sums):
                A.append(x)
                sums.update(_get_new_sums(x, A))
        best_A = A
        best_size = len(A)
        
    return best_A
