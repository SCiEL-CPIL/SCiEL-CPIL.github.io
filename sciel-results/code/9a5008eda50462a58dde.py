import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0

    if n <= 1:
        return "1"

    # Build the classic recursive superpermutation
    # S_1 = "1"
    # S_m = S_{m-1} with for each permutation sigma of 1..m-1 (in order of appearance in S_{m-1}),
    #        append sigma, m, sigma
    # The order of permutations in S_{m-1} is the order they appear as windows of length m-1.
    
    # We use a more efficient construction method:
    # Actually, the standard way is to know the order of permutations.
    # S_m is built from S_{m-1} where S_{m-1} contains all permutations of 1..m-1.
    # The order in S_{m-1} is determined recursively.
    
    # For efficiency, let's just build it step by step.
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract permutations of length m-1 from word in order of appearance
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap between end of new_word and start of block
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        
        word = new_word
    
    # Now we have a superpermutation of length 1! + 2! + ... + n!
    # For n=7, this is 5913
    
    # Convert to string for final return
    result = "".join(map(str, word))
    
    # Try to improve by removing redundant characters
    # A character can be removed if all n! permutations still appear as consecutive windows
    
    # For n <= 6, we can afford more aggressive optimization
    # For n = 7, the classic construction is 5913, and we know shorter ones exist (~5905)
    
    # Strategy: try removing characters from left to right
    # Maintain a set of all permutations and their counts
    
    import itertools
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    target_count = len(all_perms)
    
    # Count occurrences of each permutation
    from collections import defaultdict
    perm_counts = defaultdict(int)
    for i in range(len(word) - n + 1):
        perm_counts[tuple(word[i:i + n])] += 1
    
    # A permutation is "essential" if it appears exactly once
    # If we remove a character that's part of the only occurrence of some permutation, that's bad
    
    # Let's try a greedy deletion approach
    # For each position, check if removing it still leaves all permutations present
    
    def is_valid_fast(w):
        """Check if w is a valid superpermutation. Returns True if all perms present."""
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            t = tuple(w[i:i + n])
            seen.add(t)
            if len(seen) == target_count:
                return True
        return len(seen) == target_count
    
    # Greedy deletion: scan left to right, remove if possible
    # To speed up, we can check if the character is in a "critical" position
    
    changed = True
    while changed and (time.time() < deadline - 5.0):
        changed = False
        for i in range(len(word)):
            if time.time() > deadline - 5.0:
                break
            # Try removing word[i]
            # Quick check: if word[i] is part of a permutation that appears only once, skip
            # Actually, let's just try and validate
            # For efficiency, only try if i is not at the very start or end
            temp = word[:i] + word[i + 1:]
            if is_valid_fast(temp):
                word = temp
                changed = True
                break  # Restart scan since indices changed
    
    return "".join(map(str, word))
