"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations
from collections import defaultdict

# EVOLVE-BLOCK-START

def _build_recursive_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
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
            # Find maximum overlap
            max_ov = min(len(new), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _get_all_permutations(n):
    """Generate all permutations of 1..n as tuples."""
    return set(permutations(range(1, n + 1)))

def _optimize_word_fast(word, n, time_limit):
    """
    Optimizes the word by removing redundant symbols using a sliding window count method.
    Combines deterministic greedy removal with local random restarts to escape local minima.
    """
    start_time = time.time()
    current_word = word[:]
    removed_count = 0
    
    # Precompute total permutations for validity check if needed
    total_perms = 1
    for i in range(1, n + 1):
        total_perms *= i
        
    # Initialize counts
    def _init_counts(w):
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            # Only count valid permutations? 
            # The recursive construction guarantees all windows are permutations initially.
            # After deletions, new windows might form.
            # We must ensure we only track valid permutations for the "essential" check.
            # However, the standard superpermutation property is: every perm appears at least once.
            # If we delete a char, we might break a unique perm.
            # We track counts of ALL windows. If a window is not a perm, it doesn't matter for the count of perms,
            # but it matters for validity if we require "every window is a perm" (which is NOT required, 
            # only that every PERM appears).
            # Wait, the task says: "contains every one of the 5040 permutations ... as seven consecutive symbols".
            # It does NOT say "every consecutive 7 symbols must be a permutation".
            # So we only care about the set of permutation windows.
            
            # Let's refine: We only care about windows that are permutations.
            if len(set(window)) == n:
                counts[window] += 1
        return counts

    def _try_remove(pos, w, counts):
        """
        Attempts to remove symbol at pos. Returns (success, new_word, new_counts).
        If success is False, original w and counts are returned (unchanged).
        """
        # Identify windows containing pos
        start_min = max(0, pos - n + 1)
        start_max = min(pos, len(w) - n)
        
        # Check if any window containing pos has count 1
        can_remove = True
        affected_windows = []
        for k in range(start_min, start_max + 1):
            w_tuple = tuple(w[k:k+n])
            if len(set(w_tuple)) != n:
                continue # Not a permutation, irrelevant
            if counts[w_tuple] == 1:
                can_remove = False
                break
            affected_windows.append(w_tuple)
        
        if not can_remove:
            return False, w, counts
            
        # Perform removal
        new_word = w[:pos] + w[pos+1:]
        
        # Decrement affected
        for w_tuple in affected_windows:
            counts[w_tuple] -= 1
            
        # Calculate new windows
        # New windows are formed by joining w[k]...w[pos-1] with w[pos+1]...w[k+n-1]
        # The start index k of the new window must satisfy:
        # k <= pos-1 and k+n-1 >= pos+1  => k >= pos - n + 2
        # Also k >= 0 and k+n-1 < len(new_word) => k <= len(new_word) - n
        max_new_start = len(new_word) - n
        min_new_start = max(0, pos - n + 2)
        
        for k in range(min_new_start, min(pos, max_new_start + 1)):
            new_w_tuple = tuple(new_word[k:k+n])
            if len(set(new_w_tuple)) == n:
                counts[new_w_tuple] += 1
                
        return True, new_word, counts

    # Main optimization loop
    # We use a "greedy with random restarts" approach.
    # 1. Pass through the word (left to right, then right to left) removing any removable char.
    # 2. If no progress in a full cycle, try random deletions for a short time to escape local minima.
    
    while time.time() - start_time < time_limit:
        progress = False
        
        # Phase 1: Deterministic Greedy Passes
        # Try Left-to-Right
        for pos in range(len(current_word)):
            if time.time() - start_time > time_limit:
                break
            success, current_word, counts = _try_remove(pos, current_word, counts)
            if success:
                progress = True
                removed_count += 1
                # After removal, indices shift. We should restart the pass or adjust.
                # Restarting is safer and simpler.
                break
        
        if time.time() - start_time > time_limit:
            break
            
        if progress:
            # Restart counts if we broke out early? 
            # _try_remove updates counts in place. So counts are valid.
            # But we broke out of the for loop. We need to continue checking from the start?
            # Or continue from pos-1?
            # Since removing pos shifts pos+1 to pos, we should check pos again (which is now the old pos+1).
            # But restarting the whole pass is O(N^2) per removal. 
            # Given N ~ 6000, and maybe 1000 removals, 6M ops. Fast enough.
            # However, we can be smarter: continue from pos-1.
            # But the code above breaks. Let's just restart the outer while loop which restarts the for loop.
            continue
        
        # Try Right-to-Left
        for pos in range(len(current_word) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            success, current_word, counts = _try_remove(pos, current_word, counts)
            if success:
                progress = True
                removed_count += 1
                break
        
        if time.time() - start_time > time_limit:
            break
            
        if progress:
            continue
        
        # Phase 2: No deterministic progress. Try random local search.
        # Pick a random position and try to remove it.
        # If it fails, we can't do much else with single deletion.
        # We could try swapping? No, superpermutations are fixed sequences.
        # We can try "sliding" a symbol? No, that changes the sequence.
        # The only operation is deletion.
        # If no single deletion is valid, we are stuck in a local minimum for single deletions.
        # Multi-deletion? Hard to verify.
        # Let's just try random deletions for a while to see if we hit a lucky spot (unlikely if greedy failed, 
        # but greedy is order-dependent. Random order might find a different path).
        
        random_attempts = 0
        while random_attempts < 50 and time.time() - start_time < time_limit:
            pos = random.randint(0, len(current_word) - 1)
            success, new_word, new_counts = _try_remove(pos, current_word, counts)
            if success:
                current_word = new_word
                counts = new_counts
                removed_count += 1
                progress = True
                break
            random_attempts += 1
        
        if not progress:
            # Stuck. We can try to restart with a slightly different order?
            # Or just stop?
            # For now, if we can't remove anything deterministically or randomly, we stop.
            # However, the order of removal matters.
            # Let's try a different strategy: Remove from the middle outwards?
            # Or just accept the local minimum.
            break

    return current_word, removed_count

def construct(n):
    # Build the standard recursive superpermutation
    word = _build_recursive_superperm(n)
    
    # Time budget: leave 10 seconds for safety
    start_time = time.time()
    time_limit = 140.0  # seconds (Evaluator kills at 150s)
    
    # Attempt to shorten the word
    if n <= 7:  # Only attempt optimization for reasonable n
        optimized_word, removed = _optimize_word_fast(word, n, time_limit)
        if removed > 0:
            word = optimized_word
    
    # Convert to string
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
