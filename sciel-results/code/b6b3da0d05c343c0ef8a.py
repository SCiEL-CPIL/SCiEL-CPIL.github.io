"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import permutations
import random
import time

def construct(n):
    # Standard recursive construction
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
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Local optimization using graph-inspired heuristic search
    # We model the problem as finding a shorter walk in the permutation graph
    # by deleting redundant nodes (characters) from the path
    if n <= 7:
        # Precompute target permutations as integers for fast comparison
        base = 8
        target_set = set()
        for p in permutations(range(1, n + 1)):
            val = 0
            for k, sym in enumerate(p):
                val = val * base + sym
            target_set.add(val)
        
        def get_window_set(w):
            s = set()
            # Optimization: use local variable for speed
            nn = n
            bb = base
            for i in range(len(w) - nn + 1):
                val = 0
                for k in range(nn):
                    val = val * bb + w[i+k]
                s.add(val)
            return s
        
        def is_valid_int(w):
            if len(w) < n:
                return False
            # Check if all target permutations are present
            ws = get_window_set(w)
            return target_set.issubset(ws)
        
        # Time limit management
        start_time = time.time()
        time_limit = 110.0  # Leave 10s for overhead
        
        improved = True
        # Use randomized local search to escape local minima and speed up convergence
        # This mimics a heuristic search (A*-like) by prioritizing promising deletions
        while improved:
            if time.time() - start_time > time_limit:
                break
            improved = False
            # Try positions in random order to avoid systematic bias
            positions = list(range(len(word) - 1))
            random.shuffle(positions)
            
            for pos in positions:
                # Create trial word by removing character at pos
                # Using list slicing is O(L), which is acceptable for L~6000
                trial = word[:pos] + word[pos+1:]
                
                if is_valid_int(trial):
                    word = trial
                    improved = True
                    break
    
    s = "".join(map(str, word))
    return s
# EVOLVE-BLOCK-END
