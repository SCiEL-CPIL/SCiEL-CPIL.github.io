import time
import sys
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    if n == 1:
        return "1"
        
    # Generate all permutations as strings
    perms = list(itertools.permutations(range(1, n + 1)))
    perms_str = [''.join(map(str, p)) for p in perms]
    
    # Initialize
    word = perms_str[0]
    unused = perms_str[1:]
    
    # Greedy Construction
    while unused:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_ov = -1
        
        # Find permutation with max overlap
        for i, p in enumerate(unused):
            # Calculate overlap
            # Check from largest possible overlap down to 0
            # Max overlap is n-1
            ov = 0
            # Optimization: If we already have n-1, we can't do better.
            # But we break out of the inner loop, not the outer.
            
            for k in range(n - 1, -1, -1):
                if k == 0:
                    break
                # Check if suffix of word of length k matches prefix of p of length k
                # Using string slicing
                if word[-k:] == p[:k]:
                    ov = k
                    break
                    
            if ov > best_ov:
                best_ov = ov
                best_idx = i
                if best_ov == n - 1:
                    # Max possible overlap found. 
                    # We could break here to save time.
                    # This might affect quality if tie-breaking is important, 
                    # but usually n-1 overlaps are abundant enough.
                    break
                    
        if best_idx == -1:
            # Should not happen if unused is not empty
            break
            
        best_p = unused[best_idx]
        # Append non-overlapping part
        word += best_p[best_ov:]
        
        # Remove from unused (swap with last and pop)
        unused[best_idx] = unused[-1]
        unused.pop()
        
    # Local Search Polish
    # Check if we have time
    if time.time() > deadline - 10.0:
        return word
        
    # Precompute all perms set
    all_perms_set = set(perms_str)
    
    # Try to remove redundant characters
    # We do a few passes
    max_passes = 5
    for _ in range(max_passes):
        if time.time() > deadline - 5.0:
            break
            
        changed = False
        L = len(word)
        # Iterate from 1 to L-2 (cannot remove first or last easily as it breaks start/end perms? 
        # Actually, removing first char breaks the first perm if it was unique. 
        # But we check validity anyway.)
        
        for i in range(1, L - 1):
            if time.time() > deadline - 5.0:
                break
                
            # Try removing word[i]
            # new_word = word[:i] + word[i+1:]
            # Check validity
            
            # Optimization: Only check if the removed char is "critical"?
            # Hard to know. Just check.
            
            # To speed up check, we can maintain a count?
            # No, just rebuild set.
            
            # Rebuilding set of size ~6000 is fast.
            new_word = word[:i] + word[i+1:]
            
            # Check if all perms are in new_word
            # We can do this by iterating windows
            present = set()
            # Use a set for O(1) lookup? No, we just need to see if len(present) == n!
            # And that all are valid.
            
            # Faster: 
            # We know the set of all perms.
            # We can check if set(windows) == all_perms_set
            
            # Even faster: 
            # Count how many perms are present.
            # But we need to ensure NO extra perms? 
            # The problem says "contains every one of the 5040 permutations".
            # It does NOT say "only" those. 
            # "contains every one of the 5040 permutations ... as seven consecutive symbols".
            # So extra windows (which are not perms of 1..n) are allowed?
            # "Shortest word over the symbols 1 to 7 that contains every one of the 5040 permutations of 1..7 as seven consecutive symbols."
            # Usually, superpermutations are defined such that *every* window of length n is a permutation.
            # But the prompt says "contains every one ... as seven consecutive symbols".
            # It doesn't explicitly forbid other windows.
            # However, the standard definition implies the word is composed of permutations.
            # If we allow non-permutation windows, we might be able to shorten it further?
            # No, the symbols are 1..n. Any window of length n with distinct symbols is a perm.
            # If a window has duplicates, it's not a perm.
            # The requirement is just that all perms appear.
            # So, if we remove a char and create a window with duplicates, that's fine, as long as all *valid* perms still appear.
            
            # So, check:
            # 1. All perms in all_perms_set are in new_word.
            
            # Efficient check:
            # We can iterate through new_word windows and add to a set.
            # If len(set) == n! and set is subset of all_perms_set? 
            # Actually, if we just check if all_perms_set is a subset of present_windows.
            
            present_windows = set()
            for j in range(len(new_word) - n + 1):
                w = new_word[j:j+n]
                # Optional: only add if it's a valid perm?
                # If it's not a valid perm, it doesn't help fill all_perms_set.
                # But checking if it's a valid perm is O(n).
                # Just adding to set is O(1).
                present_windows.add(w)
                
            if all_perms_set.issubset(present_windows):
                word = new_word
                changed = True
                break # Restart the pass with new word
                
        if not changed:
            break
            
    return word
