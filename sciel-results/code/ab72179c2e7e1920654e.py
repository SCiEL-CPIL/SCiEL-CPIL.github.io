import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 140

    # Standard superpermutation construction with greedy nearest-neighbor reordering
    word = [1]
    for m in range(2, n + 1):
        # Extract all unique permutations of 1..m-1 from the current word
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)

        # Greedy nearest-neighbor: repeatedly pick the permutation whose block
        # [p, m, p] has maximum overlap with the current word suffix
        new = list(word)
        remaining = list(perms)
        
        while remaining:
            best_idx = -1
            best_overlap = -1
            best_suffix = None
            
            for idx, p in enumerate(remaining):
                block = list(p) + [m] + list(p)
                # Find max overlap between new's suffix and block's prefix
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        if o > best_overlap:
                            best_overlap = o
                            best_idx = idx
                            best_suffix = block[o:]
                        break
            
            if best_idx == -1:
                break
            
            new += best_suffix
            remaining.pop(best_idx)
            
            if time.time() > deadline - 10:
                break
        
        word = new

        if time.time() > deadline - 10:
            break

    if time.time() > deadline - 5:
        return "".join(map(str, word))

    # Greedy deletion loop - improved version
    max_iterations = 200
    for _ in range(max_iterations):
        if time.time() > deadline - 3:
            break
            
        L = len(word)
        if L <= n:
            break
            
        # Calculate frequency of all n-windows
        freq = {}
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            freq[p] = freq.get(p, 0) + 1
        
        # Find all valid deletion positions
        deleted = False
        # Try to find multiple deletions in one pass for efficiency
        deletable = []
        for pos in range(1, L - 1):
            i_min = max(0, pos - n + 1)
            i_max = min(pos, L - n)
            
            valid = True
            for i in range(i_min, i_max + 1):
                p = tuple(word[i:i+n])
                if freq[p] == 1:
                    valid = False
                    break
            
            if valid:
                deletable.append(pos)
        
        if not deletable:
            break
        
        # Delete the last valid position (to avoid index shifting issues)
        # Actually, let's delete from right to left to keep indices stable
        for pos in reversed(deletable):
            word = word[:pos] + word[pos+1:]
            deleted = True
            # Recompute freq after deletion
            L = len(word)
            freq = {}
            for i in range(L - n + 1):
                p = tuple(word[i:i+n])
                freq[p] = freq.get(p, 0) + 1
            break  # Restart with fresh freq
        
        if not deleted:
            break
            
    return "".join(map(str, word))
