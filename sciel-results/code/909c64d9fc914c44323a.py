import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin: 15s buffer for GC, I/O, etc.
    
    # 1. Generate Classic Superpermutation
    def classic_superperm(n):
        if n == 1:
            return [1]
        prev = classic_superperm(n - 1)
        # Extract (n-1)-permutations from prev
        perms = []
        seen = set()
        for i in range(len(prev) - (n - 2)):
            window = tuple(prev[i:i + n - 1])
            if len(set(window)) == n - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [n] + list(p)
            # Overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new

    word = classic_superperm(n)
    
    if n <= 2:
        return "".join(map(str, word))

    # 2. Prepare for Greedy Pruning
    # We need to map permutations to IDs to maintain counts efficiently.
    # For n=7, 5040 perms.
    # We can use the tuple itself as the key in a dict.
    
    # Initialize counts
    counts = {}
    total_perms = 1
    for i in range(1, n + 1):
        total_perms *= i
        
    zero_count = total_perms
    
    # Pre-compute initial counts
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if w in counts:
            counts[w] += 1
            if counts[w] == 1:
                zero_count -= 1
        else:
            counts[w] = 1
            zero_count -= 1
            
    # If already valid (should be), start pruning
    # Optimization: Track which permutations have count == 1.
    # Removing a character affects at most n-1 old windows and n-1 new windows.
    # If any old window has count 1 and is not "replaced" by a new window identical to it,
    # or if a new window is not present elsewhere, validity is lost.
    # Simpler approach: Check if the set of unique permutations remains full.
    # Since we only remove characters, we never create new permutation types that weren't
    # potentially reachable? No, we can create new windows.
    # Actually, a window is valid if it's a permutation. 
    # Wait, the task requires the word to contain *every* permutation.
    # It does NOT require that every window is a permutation.
    # But the classic superpermutation has mostly permutation windows.
    # Pruning might create non-permutation windows, which is fine as long as all 5040 perms remain.
    
    # Let's optimize the inner loop.
    # Instead of copying the whole dict, we just need to know if zero_count changes.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(word)
        for i in range(L):
            # Check time periodically
            if time.time() > deadline:
                break
                
            # Identify affected windows
            old_starts = range(max(0, i - n + 1), min(i, L - n) + 1)
            new_starts = range(max(0, i - n + 1), i)
            
            # We need to verify if the removal is safe.
            # Safety condition:
            # 1. For every old window W_old with count 1, it must be "covered" by a new window W_new that is equal to W_old?
            #    No, that's not right. If W_old disappears (count goes to 0), we need W_new to be W_old?
            #    Yes. If a permutation appears exactly once in the word, and we remove the character that breaks it,
            #    that permutation is gone UNLESS the new window formed is identical to the old one.
            #    Can a new window be identical to an old one?
            #    Old window: w[i-n+1 ... i] (indices in word)
            #    New window: w[i-n+1 ... i-1] + w[i+1 ... i+n-1] (conceptually)
            #    For them to be equal, w[i] must equal w[i+n-1] (the element shifted in).
            #    Since w is a superpermutation, w[i] and w[i+n-1] are part of some structure.
            #    Usually w[i] != w[i+n-1].
            
            # So, if any old window has count 1, and w[i] != w[i+n-1] (for the specific window),
            # then that permutation is lost.
            
            # Let's implement this check efficiently.
            
            safe = True
            # Check old windows
            for s in old_starts:
                w_old = tuple(word[s:s+n])
                if counts.get(w_old, 0) == 1:
                    # This permutation is unique.
                    # It is destroyed unless a new window replaces it.
                    # The new window that "covers" this position is the one that ends at the same logical position?
                    # The window at index s in old word becomes indices s..s+n-1.
                    # In the new word, the window starting at s (if s < i) is:
                    # word'[s:s+n] = word[s:i] + word[i+1:s+n+1]
                    # Is this equal to word[s:s+n]?
                    # Only if word[i] == word[s+n].
                    # Note: s+n is the index right after the window.
                    # The new window starts at s. It includes word[s]...word[i-1] and word[i+1]...word[s+n].
                    # The old window was word[s]...word[s+n-1].
                    # They are equal iff word[i] == word[s+n].
                    
                    if word[i] != word[s+n]:
                        safe = False
                        break
            
            if not safe:
                continue
                
            # If safe so far, we must also ensure that any NEW windows created are already present
            # OR if they are new types, that's fine (we just have more of them).
            # Wait, we only care about *losing* permutations.
            # New windows don't cause loss.
            # Old windows with count > 1 don't cause loss.
            # Old windows with count == 1 only cause loss if not replaced by identical new window.
            
            # So if `safe` is True, the removal is valid!
            
            # Commit the change
            # Update counts
            for s in old_starts:
                w = tuple(word[s:s+n])
                counts[w] -= 1
                if counts[w] == 0:
                    zero_count += 1
                    del counts[w]
            
            for s in new_starts:
                w_new = tuple(word[s:i] + word[i+1:s+n+1])
                if w_new in counts:
                    counts[w_new] += 1
                else:
                    counts[w_new] = 1
                    zero_count -= 1
            
            word.pop(i)
            improved = True
            break # Restart loop from 0
            
    return "".join(map(str, word))

if __name__ == "__main__":
    # For testing
    res = construct(7)
    print(len(res))
