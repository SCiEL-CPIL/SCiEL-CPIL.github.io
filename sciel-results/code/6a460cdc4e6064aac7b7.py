import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 140
    
    # Standard superpermutation construction
    # L_n = sum(i! for i in 1..n)
    # This construction yields that length.
    
    word = [1]
    for m in range(2, n + 1):
        # Extract all unique permutations of 1..m-1 from the current word
        # They appear in the "standard order"
        perms = []
        seen = set()
        # The windows of size m-1 in word that are permutations
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new = []
        for p in perms:
            # Block is p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find max overlap with the end of 'new'
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
        if time.time() > deadline:
            break

    # Now word is the standard superpermutation (length ~5913 for n=7)
    # We try to trim it to the shortest known (5905)
    
    if time.time() > deadline:
        return "".join(map(str, word))
    
    # Calculate frequency of all n-windows
    freq = {}
    L = len(word)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        if len(set(p)) == n:
            freq[p] = freq.get(p, 0) + 1
        
    # Greedy Deletion
    # We can delete word[i] if all permutations in the windows affected by the deletion
    # appear at least twice in the current word.
    
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        L = len(word)
        if L <= n:
            break
            
        for i in range(L):
            # Windows affected by deleting word[i]
            # A window of size n contains word[i] if it starts at j where i-n+1 <= j <= i
            # So j in [max(0, i-n+1), min(i, L-n)]
            
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            valid_deletion = True
            affected_perms = []
            
            for j in range(start_min, start_max + 1):
                p = tuple(word[j:j+n])
                if len(set(p)) == n:
                    if freq.get(p, 0) <= 1:
                        valid_deletion = False
                        break
                    affected_perms.append(p)
            
            if valid_deletion:
                # Perform deletion
                # 1. Decrement freq for affected perms
                for p in affected_perms:
                    freq[p] -= 1
                
                # 2. Remove word[i]
                word.pop(i)
                
                # 3. Update freq for new windows formed
                # The new windows are at the same indices j in [start_min, start_max]
                # but the content has shifted.
                L_new = len(word)
                for j in range(start_min, min(start_max, L_new - n) + 1):
                    # Note: if i was near the end, L_new - n might be smaller
                    p_new = tuple(word[j:j+n])
                    if len(set(p_new)) == n:
                        freq[p_new] = freq.get(p_new, 0) + 1
                        
                changed = True
                break # Restart loop as indices changed
                
    return "".join(map(str, word))
