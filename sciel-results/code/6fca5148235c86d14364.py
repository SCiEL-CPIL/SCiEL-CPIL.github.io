"""Superpermutation construction via lexicographic greedy insertion with maximal overlap."""
# EVOLVE-BLOCK-START
def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    from itertools import permutations
    
    # Use the standard recursive construction
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
            best_overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new
    
    # Pruning: try to remove characters that aren't needed
    if n >= 4:
        target_count = 1
        for k in range(1, n + 1):
            target_count *= k
        
        # Compute initial set of permutations and their counts
        perm_count = {}
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i + n])
            if len(set(window)) == n:
                perm_count[window] = perm_count.get(window, 0) + 1
        
        # Verify we have all permutations
        if len(perm_count) != target_count:
            # Fallback: return without pruning
            return "".join(map(str, word))
        
        # Try to remove characters from left to right
        i = 0
        while i < len(word) - n:
            # Windows that include position i
            affected = []
            for j in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
                window = tuple(word[j:j + n])
                if len(set(window)) == n:
                    affected.append(window)
            
            # Check if any affected permutation would be lost
            safe = True
            for p in affected:
                if perm_count[p] == 1:
                    safe = False
                    break
            
            if safe:
                for p in affected:
                    perm_count[p] -= 1
                word = word[:i] + word[i+1:]
                # Don't increment i, check same position again
            else:
                i += 1
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
