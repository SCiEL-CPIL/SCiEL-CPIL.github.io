import time
from itertools import permutations

def generate_superperm(n):
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of 1..m-1 from the current word
        # The current word is a superpermutation of 1..m-1
        # We need to find all (m-1)! permutations of 1..m-1
        # They appear as windows of length m-1
        # In the standard construction, they appear in a specific order
        
        # Efficient way:
        # The standard construction inserts the new symbol m by taking the list of permutations of 1..m-1
        # and for each perm p, it appends p + [m] + p (with overlaps).
        
        # Actually, the recursive definition is:
        # S_n is formed by taking S_{n-1} and for each permutation of 1..n-1,
        # inserting n between the occurrences?
        
        # Let's stick to the window extraction method which is robust.
        
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Max overlap with current new_word
            overlap = 0
            max_ov = min(len(new_word), len(block))
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    if n <= 1:
        return "1"
        
    # 1. Generate standard superpermutation
    word = generate_superperm(n)
    
    if n < 7:
        # For small n, the standard construction is very likely optimal or the greedy deletion will finish instantly
        pass

    # 2. Greedy Deletion Optimization
    # We maintain a dictionary of permutation counts.
    # Key: tuple of the permutation (e.g., (1,2,3))
    # Value: count of occurrences
    
    L = len(word)
    
    # Initialize counts
    perm_counts = {}
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            if w in perm_counts:
                perm_counts[w] += 1
            else:
                perm_counts[w] = 1
                
    # Total permutations needed
    target_perms = len(perm_counts)
    # Note: perm_counts only contains valid permutations (all distinct elements).
    # The number of keys should be n!
    
    # Greedy deletion loop
    # We try to remove characters from left to right.
    # If we remove one, we restart from the beginning (or continue, but restart is safer for simplicity and correctness of indices).
    # Since we expect few deletions, restart is fine.
    
    deleted_any = True
    while deleted_any and time.time() < deadline:
        deleted_any = False
        L = len(word)
        
        # We need to re-calculate counts if we changed the word?
        # No, we update counts incrementally in the previous step?
        # Wait, if I restart the loop, the `perm_counts` dictionary from the previous iteration is valid for the current `word`.
        # So we just iterate.
        
        for i in range(L):
            if time.time() > deadline - 5:
                break
                
            # Check if removing word[i] is safe
            # Windows affected: start indices j such that j <= i <= j+n-1
            # j in [max(0, i-n+1), min(i, L-n)]
            
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            safe = True
            # Store the windows we "decrement" so we can revert if we fail?
            # Or just check first.
            # Check: for each j in range, if perm_counts[window_j] == 1, then it's unsafe.
            
            for j in range(start_min, start_max + 1):
                w = tuple(word[j:j+n])
                if len(set(w)) == n:
                    if perm_counts.get(w, 0) == 1:
                        safe = False
                        break
            
            if safe:
                # Perform deletion
                # 1. Decrement counts for affected windows
                for j in range(start_min, start_max + 1):
                    w = tuple(word[j:j+n])
                    if len(set(w)) == n:
                        perm_counts[w] -= 1
                        if perm_counts[w] == 0:
                            del perm_counts[w]
                
                # 2. Remove character
                word = word[:i] + word[i+1:]
                deleted_any = True
                break # Restart the loop from i=0
                
    return "".join(map(str, word))
