import time
import itertools

def construct(n):
    # Build superpermutation using recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for i, p in enumerate(perms):
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    start_time = time.time()
    deadline = start_time + 115
    
    # Precompute: for each window position, what permutation it is
    # window_at[i] = tuple(word[i:i+n])
    # perm_occurrences[p] = list of positions where p occurs
    
    def build_occurrences(w):
        occ = {p: [] for p in all_perms}
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if window in occ:
                occ[window].append(i)
        return occ
    
    occ = build_occurrences(word)
    
    removed = 0
    while time.time() < deadline:
        progress = False
        for i in range(len(word)):
            if time.time() > deadline:
                break
            
            # Determine which windows are affected by removing position i
            # Windows starting at s include position i if s <= i <= s+n-1
            # So s is in [i-n+1, i]
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            
            # Check if any of these windows is the only occurrence of its perm
            safe = True
            for s in range(s_min, s_max + 1):
                window = tuple(word[s:s + n])
                if len(occ[window]) == 1:
                    safe = False
                    break
            
            if safe:
                # Remove position i
                word = word[:i] + word[i+1:]
                # Rebuild occurrences
                occ = build_occurrences(word)
                removed += 1
                progress = True
                break  # Restart from beginning
        
        if not progress:
            break
    
    return "".join(map(str, word))
