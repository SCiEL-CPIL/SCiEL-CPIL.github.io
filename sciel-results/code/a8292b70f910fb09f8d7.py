"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random

# EVOLVE-BLOCK-START
def construct(n):
    if n == 1:
        return "1"
    
    def standard_superperm(k):
        """Standard recursive superpermutation construction."""
        if k == 1:
            return [1]
        
        prev = standard_superperm(k - 1)
        
        # Extract all (k-1)-permutations in order of appearance
        perms = []
        seen = set()
        for i in range(len(prev) - (k - 2)):
            window = tuple(prev[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word
        new_word = []
        for p in perms:
            block = list(p) + [k] + list(p)
            max_overlap = min(len(new_word), len(block))
            best_overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    best_overlap = o
                    break
            new_word.extend(block[best_overlap:])
        
        return new_word
    
    # For n <= 5, standard construction is optimal
    if n <= 5:
        word = standard_superperm(n)
        return "".join(map(str, word))
    
    # For n >= 6, try to optimize
    # Build up to n-1 using standard
    base = standard_superperm(n - 1)
    
    # Extract all (n-1)-permutations
    perms = []
    seen = set()
    for i in range(len(base) - (n - 2)):
        window = tuple(base[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Standard construction for the last layer
    def build_last_layer(perms, new_symbol):
        new_word = []
        for p in perms:
            block = list(p) + [new_symbol] + list(p)
            max_overlap = min(len(new_word), len(block))
            best_overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    best_overlap = o
                    break
            new_word.extend(block[best_overlap:])
        return new_word
    
    if n == 7:
        # Try to optimize the ordering of perms
        m = len(perms)
        # Precompute overlaps
        # ov[i][j] = max overlap of suffix of perms[i] and prefix of perms[j]
        ov = [[0]*m for _ in range(m)]
        for i in range(m):
            p1 = perms[i]
            for j in range(m):
                p2 = perms[j]
                # Max possible overlap is n-2
                max_ov = n - 2
                best = 0
                for k in range(max_ov, -1, -1):
                    if k == 0 or p1[-k:] == p2[:k]:
                        best = k
                        break
                ov[i][j] = best
        
        # Initial path
        path = list(range(m))
        
        # Local search
        random.seed(42)
        iterations = 50000
        for _ in range(iterations):
            i = random.randint(0, m-2)
            
            # Swap i and i+1
            delta = 0
            if i > 0:
                delta -= ov[path[i-1]][path[i]]
                delta += ov[path[i-1]][path[i+1]]
            
            delta -= ov[path[i]][path[i+1]]
            delta += ov[path[i+1]][path[i]]
            
            if i < m-2:
                delta -= ov[path[i+1]][path[i+2]]
                delta += ov[path[i]][path[i+2]]
                
            if delta > 0:
                path[i], path[i+1] = path[i+1], path[i]
        
        # Build word using optimized path
        new_word = []
        prev_idx = -1
        for idx in path:
            p = perms[idx]
            block = list(p) + [n] + list(p)
            if prev_idx == -1:
                new_word.extend(block)
            else:
                ov_val = ov[prev_idx][idx]
                new_word.extend(block[ov_val:])
            prev_idx = idx
        return "".join(map(str, new_word))
    else:
        word = build_last_layer(perms, n)
        return "".join(map(str, word))
# EVOLVE-BLOCK-END
