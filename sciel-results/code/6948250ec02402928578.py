import time
import random

def construct(n):
    start_time = time.time()
    # 150s limit. Use 145s for construction + search to be safe.
    deadline = start_time + 145.0
    
    # 1. Standard recursive construction
    # This gives a valid superpermutation of length sum_{k=1}^n k!
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique (m-1)-permutations from the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # The block for permutation p is p + [m] + p
            # But we only add the necessary part based on overlap
            block = list(p) + [m] + list(p)
            
            # Find max overlap with the end of new
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For small n, standard is optimal or very close
    if n < 5:
        return "".join(map(str, word))

    # Pre-compute all permutations of 1..n as a set of tuples
    # This allows O(1) lookup for validity
    import itertools
    all_perms = set(itertools.permutations(range(1, n+1)))
    target_count = len(all_perms)

    def is_valid(w):
        if len(w) < n:
            return False
        found = set()
        # Optimization: Early exit if we have found all permutations
        # We can also check if remaining windows are enough to find missing ones
        for i in range(len(w) - n + 1):
            window = w[i:i+n]
            # Fast check: if all elements are unique, it's a permutation
            # Since elements are 1..n, uniqueness implies permutation
            if len(set(window)) == n:
                found.add(tuple(window))
                if len(found) == target_count:
                    return True
        return False

    current_word = word
    current_len = len(current_word)
    
    # Optimization Loop
    # Strategy: Randomized Hill Climbing (Deletion)
    # Try to delete a symbol. If valid, keep it.
    # Repeat until no more deletions are possible or time runs out.
    
    improved = True
    # To avoid getting stuck in a very shallow local minimum, we can 
    # sometimes accept a "worse" move? No, we only minimize length.
    # So it's strict hill climbing.
    
    # We can also try to delete multiple symbols at once?
    # If we delete i, the word changes.
    
    # To speed up, we can cache the result of is_valid? 
    # No, the word changes every time.
    
    # Let's run the loop until time runs out or no improvement.
    # We restart the scan after each success to handle index shifts and find new opportunities.
    
    while improved and time.time() < deadline:
        improved = False
        # Iterate through indices in a randomized order
        indices = list(range(len(current_word)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try removing current_word[i]
            # Quick heuristic: Don't remove if it's part of a unique window?
            # Hard to check quickly. Just try.
            
            new_word = current_word[:i] + current_word[i+1:]
            
            if is_valid(new_word):
                current_word = new_word
                improved = True
                # Restart the scan with a new shuffled list to handle index shifts
                break
                
    return "".join(map(str, current_word))
