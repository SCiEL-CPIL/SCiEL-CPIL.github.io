import time
import random

def _check_validity(seq):
    """Check if sequence is strictly increasing and has distinct consecutive sums."""
    n = len(seq)
    if n == 0:
        return True
    # Check strictly increasing
    for i in range(n - 1):
        if seq[i + 1] <= seq[i]:
            return False
    # Check distinct consecutive sums
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(seq):
    """Return the set of all consecutive block sums."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _try_insert(seq, x):
    """
    Try to insert x into the sorted sequence seq.
    Returns the new sequence if valid, else None.
    """
    # Find insertion point
    # Since seq is sorted, we can use bisect
    import bisect
    pos = bisect.bisect_left(seq, x)
    if pos < len(seq) and seq[pos] == x:
        return None # Already present
    
    # Check bounds
    if x < 1 or x > 256:
        return None
    
    # Check strict increase around insertion point
    if pos > 0 and seq[pos-1] >= x:
        return None
    if pos < len(seq) and seq[pos] <= x:
        return None
        
    new_seq = seq[:pos] + [x] + seq[pos:]
    
    # Check validity of new_seq
    if _check_validity(new_seq):
        return new_seq
    return None

def _greedy_build(order, deadline):
    """
    Build a sequence greedily.
    """
    seq = []
    used = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in used:
            continue
        new_seq = _try_insert(seq, x)
        if new_seq is not None:
            seq = new_seq
            used.add(x)
    return seq

def _local_search(seq, unused, deadline):
    """
    Improve sequence using local search (swaps and insertions).
    """
    import bisect
    best_seq = list(seq)
    best_len = len(best_seq)
    current_seq = list(best_seq)
    current_unused = set(unused)
    
    # Precompute sums for current_seq to speed up checks? 
    # _check_validity is O(N^2). N is small.
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to insert any unused element
        # Sort unused to try smaller/larger first? Random is better for diversity.
        unused_list = list(current_unused)
        random.shuffle(unused_list)
        
        for x in unused_list:
            if time.time() > deadline:
                break
            new_seq = _try_insert(current_seq, x)
            if new_seq is not None:
                current_seq = new_seq
                current_unused.remove(x)
                if len(current_seq) > best_len:
                    best_seq = list(current_seq)
                    best_len = len(best_seq)
                improved = True
                break # Restart insertion loop after one successful insertion
        
        if improved:
            continue
            
        # 2. If stuck, try a swap
        if len(current_seq) > 0 and len(current_unused) > 0:
            # Pick a random element from current_seq to remove
            idx = random.randint(0, len(current_seq) - 1)
            a = current_seq[idx]
            
            # Pick a random element from current_unused to add
            b = random.choice(list(current_unused))
            
            # Perform swap: remove a, add b
            temp_seq = current_seq[:idx] + current_seq[idx+1:]
            new_seq = _try_insert(temp_seq, b)
            
            if new_seq is not None:
                # Accept the swap
                current_seq = new_seq
                current_unused.remove(b)
                current_unused.add(a)
                if len(current_seq) > best_len:
                    best_seq = list(current_seq)
                    best_len = len(best_seq)
            else:
                # If swap failed, try removing 'a' entirely and see if we can add multiple elements
                # This is a "drop" move
                temp_seq = current_seq[:idx] + current_seq[idx+1:]
                # Try to fill temp_seq with unused elements
                # This is expensive, so only do it occasionally
                if random.random() < 0.1:
                    # Greedy fill on temp_seq
                    new_seq = _greedy_build(list(current_unused), deadline)
                    # Combine temp_seq and new_seq? No, _greedy_build builds from scratch.
                    # We need a function that tries to extend a given sequence.
                    
                    # Let's just try to add elements to temp_seq one by one
                    extended_seq = list(temp_seq)
                    extended_unused = set(current_unused)
                    # Sort extended_unused to try systematically? 
                    # Random is fine.
                    for x in list(extended_unused):
                        if time.time() > deadline:
                            break
                        res = _try_insert(extended_seq, x)
                        if res is not None:
                            extended_seq = res
                            extended_unused.remove(x)
                            
                    if len(extended_seq) > len(current_seq):
                        current_seq = extended_seq
                        current_unused = extended_unused
                        if len(current_seq) > best_len:
                            best_seq = list(current_seq)
                            best_len = len(best_seq)
                            
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_seq = []
    best_len = 0
    
    # Strategies for initial order
    strategies = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
        list(range(128, 257)) + list(range(1, 128)),
        list(range(1, 129)) + list(range(256, 127, -1)),
    ]
    
    # Add some random strategies
    rng = random.Random(42)
    for _ in range(5):
        rand_order = list(range(1, 257))
        rng.shuffle(rand_order)
        strategies.append(rand_order)
        
    for order in strategies:
        if time.time() > deadline:
            break
            
        # 1. Greedy Build
        seq = _greedy_build(order, deadline)
        
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
            
        # 2. Local Search
        used = set(seq)
        unused = set(range(1, 257)) - used
        
        improved_seq = _local_search(seq, unused, deadline)
        
        if len(improved_seq) > best_len:
            best_seq = improved_seq
            best_len = len(best_seq)
            
    # Final check: try to improve best_seq further with remaining time
    if time.time() < deadline - 5.0:
        used = set(best_seq)
        unused = set(range(1, 257)) - used
        final_seq = _local_search(best_seq, unused, deadline)
        if len(final_seq) > len(best_seq):
            best_seq = final_seq

    return best_seq
