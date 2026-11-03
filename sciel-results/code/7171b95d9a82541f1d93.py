import itertools

def construct(n):
    # 1. Generate classic superpermutation
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
            # Find max overlap
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Local Search: Greedy Deletion
    # Precompute all valid permutations
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Map permutation to a unique ID or just use the tuple in a set/dict
    # Using a dict for counts is fine.
    
    def get_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in all_perms:
                counts[p] = counts.get(p, 0) + 1
        return counts

    counts = get_counts(word)
    
    # Check if initial is valid
    if len(counts) != len(all_perms):
        return "".join(map(str, word))

    improved = True
    # Limit passes to avoid infinite loops, though greedy deletion always terminates
    # as length decreases.
    while improved:
        improved = False
        i = 0
        L = len(word)
        while i < L:
            # Try deleting word[i]
            j_min = max(0, i - n + 1)
            j_max = min(i, L - n)
            
            if j_min > j_max:
                i += 1
                continue
            
            # Check if deletion is valid
            valid_deletion = True
            lost_perms = []
            for j in range(j_min, j_max + 1):
                p = tuple(word[j:j+n])
                if p in all_perms:
                    if counts[p] == 1:
                        valid_deletion = False
                        break
                    lost_perms.append(p)
            
            if not valid_deletion:
                i += 1
                continue
            
            # Perform deletion
            for p in lost_perms:
                counts[p] -= 1
            
            # Remove from word
            word.pop(i)
            L -= 1
            
            # Update counts for new windows formed
            # The deletion was at index i.
            # New windows are those that cross the gap.
            # In the new word, the gap is between i-1 and i.
            # Windows starting at j' in [max(0, i-n+1), min(i-1, L-n)]
            j_min_new = max(0, i - n + 1)
            j_max_new = min(i - 1, L - n)
            
            for j in range(j_min_new, j_max_new + 1):
                p = tuple(word[j:j+n])
                if p in all_perms:
                    if p not in counts:
                        counts[p] = 0
                    counts[p] += 1
            
            improved = True
            # Resume scanning from the affected area
            i = max(0, i - n + 1)
            
    return "".join(map(str, word))
