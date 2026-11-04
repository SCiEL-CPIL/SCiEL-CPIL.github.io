import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110
    
    # Standard superpermutation construction
    # L_n = sum(i! for i in 1..n) - n + 1
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
            # We can optimize this by checking from largest possible overlap
            # But for n=7, m is small, so this is fast enough.
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
        if time.time() > deadline:
            break

    # Now word is the standard superpermutation (length ~5907 for n=7)
    # We try to trim it to the shortest known (5905)
    
    if time.time() > deadline:
        return "".join(map(str, word))

    # Calculate frequency of all n-windows
    freq = {}
    L = len(word)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        freq[p] = freq.get(p, 0) + 1
        
    # Trim from ends
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # Trim start
        if len(word) > n:
            first_perm = tuple(word[:n])
            if freq.get(first_perm, 0) > 1:
                freq[first_perm] -= 1
                word = word[1:]
                changed = True
                
        # Trim end
        if len(word) > n:
            last_perm = tuple(word[-n:])
            if freq.get(last_perm, 0) > 1:
                freq[last_perm] -= 1
                word = word[:-1]
                changed = True
                
    return "".join(map(str, word))
