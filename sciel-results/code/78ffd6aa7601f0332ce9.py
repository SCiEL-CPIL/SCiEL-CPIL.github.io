import itertools
import time
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Step 1: Build standard superpermutation (length = sum(i! for i=1..n))
    # For n=7: 1+2+6+24+120+720+5040 = 5913
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
            # Find max overlap with end of new
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if time.time() > deadline - 5:
        return "".join(map(str, word))
    
    # Step 2: Build frequency map of all n-windows
    L = len(word)
    window_freq = defaultdict(int)
    window_positions = defaultdict(list)  # perm -> list of start positions
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        window_freq[p] += 1
        window_positions[p].append(i)
    
    # Verify all 5040 permutations present
    required = set(itertools.permutations(range(1, n+1)))
    present = set(window_freq.keys())
    if present != required:
        # Shouldn't happen with standard construction, but safety
        return "".join(map(str, word))
    
    # Step 3: Aggressive deletion using local verification
    # Key insight: deleting position pos only affects windows [pos-n+1, pos]
    # A deletion is safe iff for every affected window, its permutation still has freq >= 1 after removal
    
    # Sort candidate positions: prefer positions where the affected windows are "redundant"
    # (appear multiple times), as they're more likely to be safely removable
    
    # For each position, compute how many of its affected windows are "critical" (freq==1)
    # A position is "safe" if no affected window is critical
    
    # Build a list of positions to try, prioritized
    # Positions where all affected windows have freq >= 2 are ideal candidates
    
    # First pass: try all positions where no affected window is critical
    # This is O(L * n) to check all positions, but each check is O(n)
    
    # For efficiency, precompute: for each window position i, is it critical?
    # critical[i] = (window_freq[window_at_i] == 1)
    
    L = len(word)
    critical = [False] * (L - n + 1)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        critical[i] = (window_freq[p] == 1)
    
    # A position pos (0-indexed in word) affects windows starting at max(0, pos-n+1) to min(pos, L-n)
    # It's safe to delete pos iff none of those windows is critical
    
    # Build priority queue of candidate positions
    # Score: number of redundant windows affected (higher = better candidate)
    
    candidates = []
    for pos in range(1, L - 1):  # don't delete first or last
        lo = max(0, pos - n + 1)
        hi = min(pos, L - n)
        if lo > hi:
            continue
        # Count how many affected windows are non-critical
        redundant_count = 0
        all_redundant = True
        for w in range(lo, hi + 1):
            if critical[w]:
                all_redundant = False
                break
            redundant_count += 1
        
        if all_redundant and redundant_count > 0:
            candidates.append((-redundant_count, pos))
    
    # Sort by priority (most redundant windows first)
    candidates.sort()
    
    # Try deletions
    deleted_count = 0
    for _, pos in candidates:
        if time.time() > deadline - 10:
            break
        
        # Re-check: position may have shifted due to previous deletions
        # We need to track the current word state
        # Actually, let's just do iterative single deletions with re-verification
        
        # Try deleting position pos in the current word
        if pos >= len(word):
            continue
        
        test_word = word[:pos] + word[pos+1:]
        if len(test_word) < n:
            continue
        
        # Verify: check if all required permutations are still present
        # Use the local check: only windows around pos changed
        # But for safety with multiple deletions, do a full check every K deletions
        
        # Quick local check: the windows affected by deleting pos
        # are those starting at max(0, pos-n+1) to min(pos, len(test_word)-n)
        # We need to verify that each such window's permutation is still present elsewhere
        
        # For efficiency, maintain a set of present permutations
        # But rebuilding it each time is O(L*n)
        
        # Alternative: just do the full verification but break early
        test_perms = set()
        for i in range(len(test_word) - n + 1):
            p = tuple(test_word[i:i+n])
            test_perms.add(p)
            if len(test_perms) == len(required):
                break
        
        if len(test_perms) == len(required):
            word = test_word
            deleted_count += 1
            
            # Rebuild critical array and candidates periodically
            if deleted_count % 100 == 0 and time.time() < deadline - 15:
                L = len(word)
                window_freq = defaultdict(int)
                for i in range(L - n + 1):
                    p = tuple(word[i:i+n])
                    window_freq[p] += 1
                
                critical = [False] * (L - n + 1)
                for i in range(L - n + 1):
                    p = tuple(word[i:i+n])
                    critical[i] = (window_freq[p] == 1)
                
                # Rebuild candidates
                candidates = []
                for pos in range(1, L - 1):
                    lo = max(0, pos - n + 1)
                    hi = min(pos, L - n)
                    if lo > hi:
                        continue
                    all_redundant = True
                    redundant_count = 0
                    for w in range(lo, hi + 1):
                        if critical[w]:
                            all_redundant = False
                            break
                        redundant_count += 1
                    if all_redundant and redundant_count > 0:
                        candidates.append((-redundant_count, pos))
                candidates.sort()
        
        if deleted_count >= 1000:
            break
    
    # Step 4: Try end trimming
    if time.time() < deadline - 5:
        changed = True
        while changed:
            changed = False
            L = len(word)
            if L > n:
                first_perm = tuple(word[:n])
                # Count occurrences
                count = 0
                for i in range(L - n + 1):
                    if tuple(word[i:i+n]) == first_perm:
                        count += 1
                if count > 1:
                    word = word[1:]
                    changed = True
            
            if L > n:
                L = len(word)
                last_perm = tuple(word[-n:])
                count = 0
                for i in range(L - n + 1):
                    if tuple(word[i:i+n]) == last_perm:
                        count += 1
                if count > 1:
                    word = word[:-1]
                    changed = True
    
    # Step 5: Second round of internal deletions (after trimming)
    if time.time() < deadline - 5:
        L = len(word)
        window_freq = defaultdict(int)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            window_freq[p] += 1
        
        critical = [False] * (L - n + 1)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            critical[i] = (window_freq[p] == 1)
        
        # Try deleting positions from left to right, skipping critical ones
        pos = 1
        while pos < len(word) - 1:
            if time.time() > deadline - 3:
                break
            
            lo = max(0, pos - n + 1)
            hi = min(pos, len(word) - n)
            
            safe = True
            for w in range(lo, hi + 1):
                if critical[w]:
                    safe = False
                    break
            
            if safe:
                test_word = word[:pos] + word[pos+1:]
                # Quick verify
                test_perms = set()
                for i in range(len(test_word) - n + 1):
                    p = tuple(test_word[i:i+n])
                    test_perms.add(p)
                    if len(test_perms) == len(required):
                        break
                if len(test_perms) == len(required):
                    word = test_word
                    # Don't increment pos, check same position again
                    continue
            
            pos += 1
    
    # Final verification
    final_perms = set()
    for i in range(len(word) - n + 1):
        final_perms.add(tuple(word[i:i+n]))
    
    if len(final_perms) != len(required):
        # Fallback: return the standard construction
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
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
    
    return "".join(map(str, word))
