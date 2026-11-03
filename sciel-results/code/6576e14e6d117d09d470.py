"""Improved superpermutation constructor with efficient local search."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        # Collect all distinct (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Maximum overlap between end of new and start of block
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _count_windows(word, n):
    """Count occurrences of each permutation window in word."""
    from collections import defaultdict
    counts = defaultdict(int)
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        counts[window] += 1
    return counts

def _is_valid_counts(counts, n):
    """Check if all permutations have count >= 1."""
    for perm in permutations(range(1, n + 1)):
        if counts[perm] < 1:
            return False
    return True

def _try_remove_chars(word, n, time_limit):
    """Try to remove redundant characters using efficient window counting."""
    start_time = time.time()
    word = list(word)
    
    # Build initial window counts
    from collections import defaultdict
    counts = defaultdict(int)
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        counts[window] += 1
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Scan from right to left to make removals
        for i in range(len(word) - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                break
            if len(word) <= n:
                break
            
            # Windows affected by removing position i:
            # Windows starting at positions max(0, i-n+1) to min(i, len(word)-n)
            # A window starting at s includes position i if s <= i <= s+n-1
            # So s in [i-n+1, i] intersect [0, len(word)-n]
            
            lo = max(0, i - n + 1)
            hi = min(i, len(word) - n)
            
            # Check if removal is safe: for each affected window,
            # if count was 1, the window must still appear elsewhere
            # Actually, we need to check: after removal, do all permutations still appear?
            # The windows that disappear are those starting at lo..hi (that include position i)
            # New windows that appear: in the shifted region
            
            # Simpler: just check if any window with count 1 is among those that would disappear
            # But removal also creates new windows! So we need to be careful.
            
            # Let's just do the full check but optimized:
            # Remove position i, check validity
            test_word = word[:i] + word[i+1:]
            
            # Quick check: if len(test_word) < n, invalid
            if len(test_word) < n:
                continue
            
            # Check if all permutations still present
            # Build set of windows in test_word
            seen = set()
            for j in range(len(test_word) - n + 1):
                w = tuple(test_word[j:j+n])
                seen.add(w)
                if len(seen) == 5040:  # Early exit if we have all
                    break
            
            # Check if seen contains all permutations
            if len(seen) == 5040:
                # Verify all permutations are in seen
                valid = True
                for perm in permutations(range(1, n + 1)):
                    if perm not in seen:
                        valid = False
                        break
                if valid:
                    word = test_word
                    improved = True
                    # Rebuild counts
                    counts = defaultdict(int)
                    for j in range(len(word) - n + 1):
                        w = tuple(word[j:j+n])
                        counts[w] += 1
                    break  # Restart scan
    
    return word

def _try_remove_chars_v2(word, n, time_limit):
    """More efficient: maintain window counts and only check affected windows."""
    start_time = time.time()
    word = list(word)
    L = len(word)
    
    from collections import defaultdict
    
    # Build initial counts
    counts = defaultdict(int)
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        counts[w] += 1
    
    # Precompute: which permutations are "critical" (count == 1)
    critical = set()
    for perm in permutations(range(1, n + 1)):
        if counts[perm] <= 1:
            critical.add(perm)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(L - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                break
            if L <= n:
                break
            
            # Windows that include position i: start positions in [i-n+1, i] ∩ [0, L-n]
            lo = max(0, i - n + 1)
            hi = min(i, L - n)
            
            # These windows will disappear. Check if any critical window disappears
            # and is not recreated by the shift
            safe = True
            for s in range(lo, hi + 1):
                w = tuple(word[s:s+n])
                if counts[w] == 1:
                    # This window appears only here. After removal, it's gone unless
                    # a new window with the same content appears due to the shift
                    # New windows appear at positions s' where the window content changes
                    # Due to removal of position i, positions >= i shift left by 1
                    # New windows: for s' in [lo, hi] (approximately), the window at s' 
                    # in new word = word[s':i] + word[i+1:s'+n] (if s' < i < s'+n)
                    # This is complex. Let's just check: is w present in the new word?
                    # Quick check: is w a window in word[i+1:] or word[:i] or the merged region?
                    # The merged region is word[i-1] + word[i+1] (adjacent after removal)
                    # Windows in merged region: those that span position i
                    # A new window spanning the gap: for offset k in [0, n-1],
                    # window = word[i-1-k:i-1] + word[i+1:i+1+(n-1-k)] if valid
                    # This is getting complex. Let's just do the full validity check
                    # but with early termination.
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Removal seems safe, do full verification
            test_word = word[:i] + word[i+1:]
            if len(test_word) < n:
                continue
            
            seen = set()
            for j in range(len(test_word) - n + 1):
                w = tuple(test_word[j:j+n])
                seen.add(w)
                if len(seen) == 5040:
                    break
            
            if len(seen) == 5040:
                valid = True
                for perm in permutations(range(1, n + 1)):
                    if perm not in seen:
                        valid = False
                        break
                if valid:
                    word = test_word
                    L = len(word)
                    improved = True
                    # Rebuild counts
                    counts = defaultdict(int)
                    for j in range(L - n + 1):
                        w = tuple(word[j:j+n])
                        counts[w] += 1
                    critical = set()
                    for perm in permutations(range(1, n + 1)):
                        if counts[perm] <= 1:
                            critical.add(perm)
                    break
    
    return word

def construct(n):
    start_time = time.time()
    time_limit = 140  # Leave 10s safety margin for 150s limit
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining > 5:
        # Try to remove redundant characters
        word = _try_remove_chars_v2(word, n, remaining)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
