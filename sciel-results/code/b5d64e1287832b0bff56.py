import itertools
import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Step 1: Build standard superpermutation
    # Recursive construction:
    # S_1 = "1"
    # S_{m+1} = concatenation of (S_m + [m+1] + S_m) for each permutation in S_m, with overlaps removed
    # Standard efficient generation:
    word = [1]
    for m in range(2, n + 1):
        # Generate all permutations of 1..m-1 that appear as windows of length m-1
        # Actually, the standard recursive step is:
        # Let P_{m-1} be the list of permutations of 1..m-1 in the order they appear in S_{m-1}
        # S_m is formed by: for each p in P_{m-1}, append (p, m, p) but overlap with previous.
        
        # Simpler: Just generate the standard word using the known pattern or recursive definition.
        # S_n is defined recursively.
        # S_1 = [1]
        # S_{n} = for each permutation p of 1..n-1 in the order they appear in S_{n-1}:
        #          append p, then n, then p
        #          but overlap with the end of the current word.
        
        # Let's implement this directly.
        perms, seen = [], set()
        # Extract order of permutations of size m-1 from current word
        for i in range(len(word) - (m - 1) + 1):
            w = tuple(word[i:i + m - 1])
            if len(set(w)) == m - 1 and w not in seen:
                seen.add(w)
                perms.append(w)
        
        new_word = []
        for p in perms:
            # Block is p + [m] + p
            block = list(p) + [m] + list(p)
            # Find max overlap with end of new_word
            # Max possible overlap is len(block)
            # We want to find largest k such that new_word[-k:] == block[:k]
            max_ov = min(len(new_word), len(block))
            overlap = 0
            for k in range(max_ov, -1, -1):
                if k == 0 or new_word[len(new_word) - k:] == block[:k]:
                    overlap = k
                    break
            new_word += block[overlap:]
        word = new_word

    if time.time() > deadline - 10:
        return "".join(map(str, word))

    # Step 2: Optimization via Greedy Deletion
    # We maintain a list `word` and a dict `freq` mapping tuple(window) to count.
    # We try to delete indices.
    
    L = len(word)
    # Initialize freq
    freq = defaultdict(int)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        freq[p] += 1
    
    # Helper to check if index i can be deleted
    # Windows covering i: start j in [i-n+1, i]
    # All must have freq > 1
    def can_delete(i):
        if i <= 0 or i >= L - 1:
            return False
        lo = max(0, i - n + 1)
        hi = min(i, L - n)
        for j in range(lo, hi + 1):
            p = tuple(word[j:j+n])
            if freq[p] <= 1:
                return False
        return True

    # Helper to perform deletion and update freq
    def delete_index(i):
        nonlocal L, freq
        # 1. Decrement freq for windows that are removed/changed
        # Windows in old word covering i: start j in [i-n+1, i]
        lo = max(0, i - n + 1)
        hi = min(i, L - n)
        for j in range(lo, hi + 1):
            p = tuple(word[j:j+n])
            freq[p] -= 1
            if freq[p] == 0:
                del freq[p]
        
        # 2. Remove word[i]
        word.pop(i)
        L -= 1
        
        # 3. Increment freq for new merged windows
        # New windows in new word starting at k in [i-n+1, i-1]
        # Note: i in new word is the position after the deleted one? No.
        # The gap is at i.
        # New windows span the gap.
        # Start k in new word:
        # The windows that span the deletion are those that start before i and end after i (in old indices).
        # In new indices, they start at k where k < i and k+n-1 >= i.
        # So k in [i-n+1, i-1].
        # Also, we need to ensure k >= 0 and k <= L-n.
        
        new_lo = max(0, i - n + 1)
        new_hi = min(i - 1, L - n)
        
        for k in range(new_lo, new_hi + 1):
            # Window in new word at k
            # It consists of word[k..i-1] and word[i..k+n-1] (since word[i] is the element that was at i+1 before)
            # Actually, just slice the new word
            p = tuple(word[k:k+n])
            freq[p] += 1

    # Greedy Passes
    # We alternate between left-to-right and right-to-left to escape local minima
    passes = 0
    while time.time() < deadline - 20:
        changed = False
        
        # Left to Right
        i = 1
        while i < L - 1:
            if can_delete(i):
                delete_index(i)
                changed = True
                # Do not increment i, check same position again (now it's the next element)
            else:
                i += 1
        
        if not changed:
            # Right to Left
            i = L - 2
            while i > 0:
                if can_delete(i):
                    delete_index(i)
                    changed = True
                else:
                    i -= 1
        
        if not changed:
            break
        
        passes += 1
        if passes > 100: # Safety break
            break

    # Final Verification
    final_perms = set()
    for i in range(len(word) - n + 1):
        final_perms.add(tuple(word[i:i+n]))
    
    required = set(itertools.permutations(range(1, n+1)))
    if final_perms != required:
        # Fallback to original if broken
        # Reconstruct standard word
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - (m - 1) + 1):
                w = tuple(word[i:i + m - 1])
                if len(set(w)) == m - 1 and w not in seen:
                    seen.add(w)
                    perms.append(w)
            new_word = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_ov = min(len(new_word), len(block))
                overlap = 0
                for k in range(max_ov, -1, -1):
                    if k == 0 or new_word[len(new_word) - k:] == block[:k]:
                        overlap = k
                        break
                new_word += block[overlap:]
            word = new_word

    return "".join(map(str, word))
