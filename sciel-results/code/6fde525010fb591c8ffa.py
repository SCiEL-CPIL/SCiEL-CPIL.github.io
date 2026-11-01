"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def construct(n):
    """
    Construct a superpermutation over symbols 1..n.
    
    Strategy:
    1. Start with an efficient recursive "sandwich" construction.
    2. Use highly optimized local search with O(n) incremental validation
       for deletions, and efficient swap-based perturbation.
    
    For n <= 7, extensive local search within time budget.
    """
    start_time = time.time()
    deadline = 147.0  # safety margin before 150s kill
    
    # Step 1: Efficient recursive construction
    word = _sandwich_superperm(n)
    
    if n <= 4:
        return _optimize_small_n(word, n, deadline)
    
    # For larger n, use optimized local search
    return _local_search_optimized(word, n, deadline)


def _sandwich_superperm(n):
    """
    Recursive superpermutation using the sandwich pattern.
    For each step m, take the (m-1)-superpermutation, extract unique 
    (m-1)-length windows, and for each window w, emit w + [m] + w,
    merging with maximum suffix-prefix overlap.
    """
    word = [1]
    for m in range(2, n + 1):
        # Extract unique (m-1)-length windows in order of first appearance
        seen = set()
        windows = []
        for i in range(len(word) - (m - 1) + 1):
            w = tuple(word[i:i + m - 1])
            if len(set(w)) == m - 1 and w not in seen:
                seen.add(w)
                windows.append(w)
        
        new_word = []
        for w in windows:
            block = list(w) + [m] + list(w)
            # Find maximum overlap between end of new_word and start of block
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, 0, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        word = new_word
    return word


def _factorial(n):
    """Compute n!"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _perm_to_int(perm, n):
    """Encode a permutation tuple as an integer."""
    result = 0
    for val in perm:
        result = result * n + val
    return result


def _build_all_perm_ints(n):
    """Build list of all permutation integers for 1..n."""
    return [_perm_to_int(p, n) for p in permutations(range(1, n + 1))]


def _local_search_optimized(word, n, deadline):
    """
    Optimized local search with O(n) incremental validation.
    
    Key: maintain a dict of perm_int -> count. Deletion at pos i only
    affects at most n windows. Check safety in O(n). After deletion,
    update counts incrementally in O(n) rather than rebuilding.
    """
    word = list(word)
    best_word = list(word)
    best_len = len(word)
    
    all_perm_ints = _build_all_perm_ints(n)
    target_count = len(all_perm_ints)
    
    # Build initial count map
    def build_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                p_int = _perm_to_int(window, n)
                counts[p_int] = counts.get(p_int, 0) + 1
        return counts
    
    counts = build_counts(word)
    
    # Verify initial validity
    def is_valid(counts):
        for p in all_perm_ints:
            if p not in counts:
                return False
        return True
    
    if not is_valid(counts):
        # Fallback: return as-is
        return "".join(map(str, best_word))
    
    random.seed(42)
    
    # Helper: get perm ints for windows containing position pos in word w
    def get_window_perms(w, pos):
        """Return list of (perm_int, window_start) for windows containing pos."""
        result = []
        start = max(0, pos - n + 1)
        end = min(len(w) - n, pos)
        for i in range(start, end + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                result.append(_perm_to_int(window, n))
        return result
    
    # Phase 1: Iterative deletion with incremental count updates
    max_deletion_passes = 200
    for pass_num in range(max_deletion_passes):
        if time.time() > deadline - 20:
            break
        
        indices = list(range(len(word)))
        random.shuffle(indices)
        improved = False
        
        for idx in indices:
            if time.time() > deadline - 20:
                break
            if len(word) < n:
                break
            
            # Get perms in windows containing idx
            affected = get_window_perms(word, idx)
            
            # Check safety: no affected perm has count <= 1
            safe = True
            for p_int in affected:
                if counts.get(p_int, 0) <= 1:
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Perform deletion
            old_len = len(word)
            # Remove old affected windows from counts
            for p_int in affected:
                counts[p_int] -= 1
                if counts[p_int] == 0:
                    del counts[p_int]
            
            # Create new word
            word = word[:idx] + word[idx + 1:]
            new_len = len(word)
            
            # Add new windows that span the deletion point
            # Windows starting from max(0, idx-n+1) to min(new_len-n, idx-1)
            # that weren't present before (they now include the gap)
            new_start = max(0, idx - n + 1)
            new_end = min(new_len - n, idx - 1)
            for i in range(new_start, new_end + 1):
                window = tuple(word[i:i + n])
                if len(set(window)) == n:
                    p_int = _perm_to_int(window, n)
                    counts[p_int] = counts.get(p_int, 0) + 1
            
            best_word = list(word)
            best_len = len(word)
            improved = True
            break  # Restart scan with updated word
        
        if not improved:
            break
    
    # Phase 2: Swap perturbation to escape local minima
    if time.time() < deadline - 15:
        # Multiple rounds of swap + delete
        for round_num in range(20):
            if time.time() > deadline - 15:
                break
            
            # Try random adjacent swaps
            swaps_attempted = 0
            swaps_accepted = 0
            max_swaps = min(200, len(word))
            
            for _ in range(max_swaps):
                if time.time() > deadline - 15:
                    break
                if len(word) < 2:
                    break
                
                i = random.randint(0, len(word) - 2)
                
                # Compute affected windows for positions i and i+1
                old_affected = set()
                for pos in [i, i + 1]:
                    start = max(0, pos - n + 1)
                    end = min(len(word) - n, pos)
                    for k in range(start, end + 1):
                        window = tuple(word[k:k + n])
                        if len(set(window)) == n:
                            old_affected.add(_perm_to_int(window, n))
                
                # Perform swap
                word[i], word[i + 1] = word[i + 1], word[i]
                
                # Check new affected windows
                new_affected = set()
                for pos in [i, i + 1]:
                    start = max(0, pos - n + 1)
                    end = min(len(word) - n, pos)
                    for k in range(start, end + 1):
                        window = tuple(word[k:k + n])
                        if len(set(window)) == n:
                            new_affected.add(_perm_to_int(window, n))
                
                # Check validity: all old_affected perms must still exist
                # Actually we need: for each perm in old_affected, count >= 1 in new word
                # Simplified: check that no perm in old_affected has been lost
                valid = True
                for p_int in old_affected:
                    if counts.get(p_int, 0) <= 1 and p_int not in new_affected:
                        valid = False
                        break
                    elif counts.get(p_int, 0) <= 1:
                        # This perm only appears in affected windows; check if it's in new_affected
                        if p_int not in new_affected:
                            valid = False
                            break
                
                if valid:
                    # Update counts: remove old affected, add new affected
                    # More carefully: recompute counts for affected region
                    # Remove old
                    for p_int in old_affected:
                        counts[p_int] -= 1
                        if counts[p_int] == 0:
                            del counts[p_int]
                    # Add new
                    for p_int in new_affected:
                        counts[p_int] = counts.get(p_int, 0) + 1
                    
                    best_word = list(word)
                    best_len = len(word)
                    swaps_accepted += 1
                else:
                    # Revert swap
                    word[i], word[i + 1] = word[i + 1], word[i]
                
                swaps_attempted += 1
            
            # After swaps, try more deletions
            if time.time() < deadline - 10:
                for _ in range(50):
                    if time.time() > deadline - 10:
                        break
                    
                    indices = list(range(len(word)))
                    random.shuffle(indices)
                    improved = False
                    
                    for idx in indices:
                        if time.time() > deadline - 10:
                            break
                        if len(word) < n:
                            break
                        
                        affected = get_window_perms(word, idx)
                        safe = True
                        for p_int in affected:
                            if counts.get(p_int, 0) <= 1:
                                safe = False
                                break
                        
                        if not safe:
                            continue
                        
                        # Perform deletion
                        for p_int in affected:
                            counts[p_int] -= 1
                            if counts[p_int] == 0:
                                del counts[p_int]
                        
                        word = word[:idx] + word[idx + 1:]
                        new_len = len(word)
                        
                        new_start = max(0, idx - n + 1)
                        new_end = min(new_len - n, idx - 1)
                        for i2 in range(new_start, new_end + 1):
                            window = tuple(word[i2:i2 + n])
                            if len(set(window)) == n:
                                p_int = _perm_to_int(window, n)
                                counts[p_int] = counts.get(p_int, 0) + 1
                        
                        best_word = list(word)
                        best_len = len(word)
                        improved = True
                        break
                    
                    if not improved:
                        break
    
    # Phase 3: Final cleanup - try deletions one more time
    if time.time() < deadline - 3:
        # Rebuild counts to ensure consistency
        counts = build_counts(best_word)
        word = list(best_word)
        
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            
            indices = list(range(len(word)))
            random.shuffle(indices)
            improved = False
            
            for idx in indices:
                if time.time() > deadline - 3:
                    break
                if len(word) < n:
                    break
                
                affected = get_window_perms(word, idx)
                safe = True
                for p_int in affected:
                    if counts.get(p_int, 0) <= 1:
                        safe = False
                        break
                
                if not safe:
                    continue
                
                for p_int in affected:
                    counts[p_int] -= 1
                    if counts[p_int] == 0:
                        del counts[p_int]
                
                word = word[:idx] + word[idx + 1:]
                new_len = len(word)
                
                new_start = max(0, idx - n + 1)
                new_end = min(new_len - n, idx - 1)
                for i2 in range(new_start, new_end + 1):
                    window = tuple(word[i2:i2 + n])
                    if len(set(window)) == n:
                        p_int = _perm_to_int(window, n)
                        counts[p_int] = counts.get(p_int, 0) + 1
                
                best_word = list(word)
                best_len = len(word)
                improved = True
                break
            
            if not improved:
                break
    
    return "".join(map(str, best_word))


def _optimize_small_n(word, n, deadline):
    """Optimize for small n where we can be more exhaustive."""
    word = list(word)
    best_word = list(word)
    best_len = len(word)
    
    target_count = _factorial(n)
    all_perm_ints = _build_all_perm_ints(n)
    
    def check_valid(w):
        found = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                found.add(_perm_to_int(window, n))
                if len(found) == target_count:
                    return True
        return len(found) == target_count
    
    # Iterative deletion
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i in range(len(word)):
            new_word = word[:i] + word[i + 1:]
            if check_valid(new_word):
                word = new_word
                best_word = list(word)
                best_len = len(word)
                improved = True
                break
    
    return "".join(map(str, best_word))


# EVOLVE-BLOCK-END
