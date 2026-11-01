"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    if time.time() > deadline:
        return "".join(map(str, word))
    
    # Efficient polish using coverage counts
    def build_coverage(w, n):
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            counts[tuple(w[i:i+n])] += 1
        return counts
    
    improved = list(word)
    counts = build_coverage(improved, n)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(improved)):
            if time.time() > deadline:
                break
            # Windows affected by removing position i
            j_min = max(0, i - n + 1)
            j_max = min(i, len(improved) - n)
            
            safe = True
            affected = []
            for j in range(j_min, j_max + 1):
                w = tuple(improved[j:j+n])
                affected.append(w)
                if counts[w] <= 1:
                    safe = False
                    break
            
            if safe:
                for w in affected:
                    counts[w] -= 1
                improved.pop(i)
                changed = True
                break
    
    return "".join(map(str, improved))
# EVOLVE-BLOCK-END
