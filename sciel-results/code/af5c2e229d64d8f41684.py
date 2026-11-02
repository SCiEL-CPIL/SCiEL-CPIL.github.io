import time
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    if n == 1:
        return "1"
    
    # Build the standard recursive superpermutation
    def get_recursive_superpermutation(k):
        if k == 1:
            return [1]
        
        prev = get_recursive_superpermutation(k - 1)
        
        # Extract unique permutations of length k-1 from prev in order
        perms = []
        seen = set()
        for i in range(len(prev) - (k - 1) + 1):
            window = tuple(prev[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        new_word = []
        for p in perms:
            block = list(p) + [k] + list(p)
            # Overlap
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
            
        return new_word

    word = get_recursive_superpermutation(n)
    
    # For small n, recursive is optimal or close enough, and deletion might be overkill/slow
    # But deletion is fast enough for n=7.
    # n=1: 1
    # n=2: 121 (len 3). Deletion: 121 -> remove 1? 21 (miss 12). remove 2? 11 (invalid). remove 1? 12 (miss 21).
    # Actually, for n<=4, recursive is optimal. 
    if n <= 4:
        return "".join(map(str, word))
        
    # For n=5, recursive is 153. Optimal is 153.
    if n == 5:
        return "".join(map(str, word))
        
    # For n=6, recursive is 873. Optimal is 872? Or 873? 
    # Known optimal for 6 is 872? No, 873 is often cited as optimal for 6 in some contexts, 
    # but 872 is the lower bound. Actually, 873 is the length of the recursive one.
    # Let's run deletion for n >= 6.
    
    # Precompute all permutations for quick lookup?
    # There are 5040 for n=7.
    # Using a set of tuples is fine.
    
    L = len(word)
    counts = defaultdict(int)
    
    # Initial count
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        # Only count if it's a valid permutation (distinct elements)
        # The recursive construction guarantees all windows are permutations?
        # No, intermediate windows might not be. But the ones we care about are the n! permutations.
        # We should only count valid permutations.
        if len(set(w)) == n:
            counts[w] += 1
            
    # Iterative deletion
    # We scan left to right. If we remove an element, we restart or continue?
    # Removing an element changes indices. 
    # It is safer to use a list and remove, then continue scanning?
    # If we remove word[i], the next element is now at i.
    # We should check index i again.
    
    i = 0
    while i < len(word):
        if time.time() > deadline - 5.0:
            break
            
        L = len(word)
        if L < n:
            break
            
        # Check if word[i] can be removed
        # Affected old windows: start j in [i-n+1, i]
        min_j = max(0, i - n + 1)
        max_j = min(L - n, i)
        
        can_remove = True
        affected_windows = []
        
        for j in range(min_j, max_j + 1):
            w = tuple(word[j:j+n])
            if len(set(w)) == n: # Only track valid perms
                if counts[w] == 1:
                    can_remove = False
                    break
                affected_windows.append(w)
                
        if not can_remove:
            i += 1
            continue
            
        # If can remove, perform removal
        # 1. Decrement counts of affected windows
        for w in affected_windows:
            counts[w] -= 1
            if counts[w] == 0:
                del counts[w]
                
        # 2. Calculate new windows formed at junction
        # Junction is between new i-1 and new i.
        # New word is word[:i] + word[i+1:]
        # New windows start at k in new coords.
        # k ranges from max(0, i-n+1) to min(i-1, L-1-n)
        # Note: L is old length. New length is L-1.
        # Max start in new word is (L-1) - n + 1 = L - n.
        
        min_k = max(0, i - n + 1)
        max_k = min(i - 1, L - n)
        
        for k in range(min_k, max_k + 1):
            # Construct new window
            # Prefix: old word[k : i]
            # Suffix: old word[i+1 : i+1 + (n - (i-k))]
            prefix = word[k:i]
            suffix_len = n - (i - k)
            suffix = word[i+1 : i+1 + suffix_len]
            
            if len(prefix) + len(suffix) != n:
                continue
                
            new_w = tuple(prefix + suffix)
            
            # Check if new_w is a valid permutation
            if len(set(new_w)) == n:
                counts[new_w] += 1
                
        # 3. Remove word[i]
        word.pop(i)
        
        # Do not increment i, check same index again (which is now the next element)
        # i remains same, but word has shrunk.
        
    return "".join(map(str, word))
