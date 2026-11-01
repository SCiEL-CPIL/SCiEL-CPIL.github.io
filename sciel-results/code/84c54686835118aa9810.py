"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def construct(n):
    """
    Construct a superpermutation over symbols 1..n.
    
    Strategy:
    1. Start with the classic recursive construction (length = sum_{k=1}^n k!).
    2. Use incremental local search to shorten the word while maintaining
       the property that every permutation appears as a consecutive window.
    
    For n <= 7, we can do extensive local search with efficient validation.
    """
    start_time = time.time()
    deadline = 148.0  # safety margin before 150s kill
    
    # Step 1: Classic recursive construction
    word = _classic_superperm(n)
    
    if n <= 4:
        return _optimize_small_n(word, n, start_time, deadline)
    
    # For larger n, use optimized local search
    return _local_search_optimized(word, n, start_time, deadline)


def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def _perm_to_int(perm, n):
    """Encode a permutation tuple as an integer using factorial number system."""
    result = 0
    for i, val in enumerate(perm):
        result = result * n + val
    return result


def _build_perm_set(n):
    """Build set of all permutation integers for 1..n."""
    return {_perm_to_int(p, n) for p in permutations(range(1, n + 1))}


def _local_search_optimized(word, n, start_time, deadline):
    """
    Optimized local search with incremental validation.
    
    Key insight: Track occurrence counts of each permutation. A deletion at
    position i only affects windows that include position i. We can check
    validity in O(n) by seeing if any affected permutation drops to 0.
    """
    word = list(word)
    best_word = list(word)
    best_len = len(word)
    
    target_count = _factorial(n)
    all_perms = _build_perm_set(n)
    
    # Build occurrence count map
    def build_counts(w):
        """Map from perm_int -> count of occurrences in w."""
        counts = {}
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                p_int = _perm_to_int(window, n)
                counts[p_int] = counts.get(p_int, 0) + 1
        return counts
    
    def is_valid_counts(counts):
        """Check if all permutations have count >= 1."""
        for p in all_perms:
            if p not in counts:
                return False
        return True
    
    def get_affected_perms(w, pos):
        """Get perm ints for windows that include position pos."""
        affected = []
        start = max(0, pos - n + 1)
        end = min(len(w) - n, pos)
        for i in range(start, end + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                affected.append(_perm_to_int(window, n))
        return affected
    
    # Initial count
    counts = build_counts(word)
    
    # Phase 1: Deletion search with incremental validation
    random.seed(42)
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() > deadline - 15:
            break
        
        indices = list(range(len(word)))
        random.shuffle(indices)
        improved = False
        
        for idx in indices:
            if time.time() > deadline - 15:
                break
            if len(word) < n:
                break
            
            # Get permutations affected by deleting position idx
            affected = get_affected_perms(word, idx)
            
            # Check if any affected perm would drop to 0
            safe = True
            for p_int in affected:
                if counts.get(p_int, 0) <= 1:
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Perform deletion
            old_len = len(word)
            new_word = word[:idx] + word[idx + 1:]
            
            # Update counts: remove affected perms, add new perms in the gap
            # After deletion, windows that shift into position idx need updating
            # The windows affected are those starting from max(0, idx-n+1) to min(old_len-n, idx)
            # In the new word, windows starting from max(0, idx-n+1) to min(len(new_word)-n, idx-1)
            
            # Simpler: rebuild counts for affected region
            # Remove old affected
            for p_int in affected:
                counts[p_int] -= 1
                if counts[p_int] == 0:
                    del counts[p_int]
            
            # Add new windows that now exist in the gap
            # Windows that now span the deletion point
            new_start = max(0, idx - n + 1)
            new_end = min(len(new_word) - n, idx - 1)
            for i in range(new_start, new_end + 1):
                window = tuple(new_word[i:i + n])
                if len(set(window)) == n:
                    p_int = _perm_to_int(window, n)
                    # Check if this window was already counted (it wasn't in old word at this position)
                    # Actually we need to be more careful - let's just rebuild if needed
                    pass
            
            # For correctness, rebuild counts if the update is complex
            # But for efficiency, let's use a simpler approach:
            # Just verify the new word is valid by checking only affected perms
            # Since we already checked no perm drops to 0, and we only removed one char,
            # the new windows created can only ADD to counts, never subtract.
            # So if all old affected perms had count >= 2, the new word is valid.
            
            # Update word and counts properly
            word = new_word
            # Rebuild counts (expensive but correct)
            # Optimization: only rebuild if we deleted
            counts = build_counts(word)
            best_word = list(word)
            best_len = len(word)
            improved = True
            break  # Restart with new word
        
        if not improved:
            break
    
    # Phase 2: Swap search to enable further deletions
    if time.time() < deadline - 10:
        for _ in range(500):
            if time.time() > deadline - 10:
                break
            if len(word) < 2:
                break
            
            i = random.randint(0, len(word) - 2)
            new_word = list(word)
            new_word[i], new_word[i + 1] = new_word[i + 1], new_word[i]
            
            # Quick check: does the swap preserve validity?
            # Only windows containing i or i+1 are affected
            affected_positions = set()
            for pos in [i, i + 1]:
                start = max(0, pos - n + 1)
                end = min(len(new_word) - n, pos)
                for k in range(start, end + 1):
                    affected_positions.add(k)
            
            # Check if all affected perms still have count >= 1 in new word
            valid = True
            for k in affected_positions:
                window = tuple(new_word[k:k + n])
                if len(set(window)) == n:
                    p_int = _perm_to_int(window, n)
                    # Count occurrences in new word
                    cnt = 0
                    for j in range(len(new_word) - n + 1):
                        w2 = tuple(new_word[j:j + n])
                        if len(set(w2)) == n and _perm_to_int(w2, n) == p_int:
                            cnt += 1
                    if cnt == 0:
                        valid = False
                        break
            
            if valid:
                word = new_word
                best_word = list(word)
                best_len = len(word)
    
    # Phase 3: Another deletion pass after swaps
    if time.time() < deadline - 5:
        counts = build_counts(word)
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            
            indices = list(range(len(word)))
            random.shuffle(indices)
            improved = False
            
            for idx in indices:
                if time.time() > deadline - 5:
                    break
                if len(word) < n:
                    break
                
                affected = get_affected_perms(word, idx)
                safe = True
                for p_int in affected:
                    if counts.get(p_int, 0) <= 1:
                        safe = False
                        break
                
                if not safe:
                    continue
                
                new_word = word[:idx] + word[idx + 1:]
                word = new_word
                counts = build_counts(word)
                best_word = list(word)
                best_len = len(word)
                improved = True
                break
            
            if not improved:
                break
    
    return "".join(map(str, best_word))


def _factorial(n):
    """Compute n!"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _optimize_small_n(word, n, start_time, deadline):
    """Optimize for small n where we can be exhaustive."""
    word = list(word)
    best_word = list(word)
    best_len = len(word)
    
    target_count = _factorial(n)
    all_perms = _build_perm_set(n)
    
    def check_valid(w):
        found = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                found.add(_perm_to_int(window, n))
            if len(found) == target_count:
                return True
        return len(found) == target_count
    
    # Exhaustive deletion search
    improved = True
    while improved and time.time() < deadline - 5:
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
