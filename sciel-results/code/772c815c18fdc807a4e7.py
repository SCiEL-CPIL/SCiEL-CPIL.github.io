import random
import time
from itertools import permutations

def construct(n):
    # Step 1: Generate classic superpermutation
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    if n == 7:
        # Optimization for n=7
        # Precompute rank map
        perms_7 = list(permutations(range(1, 8)))
        rank_map = {p: i for i, p in enumerate(perms_7)}
        
        def is_superperm(w):
            seen = set()
            for i in range(len(w) - n + 1):
                window = tuple(w[i:i+n])
                seen.add(rank_map[window])
                if len(seen) == 5040:
                    return True
            return len(seen) == 5040
        
        start_time = time.time()
        deadline = start_time + 140.0 # Safety margin
        
        w_current = word
        # Greedy reduction
        # We try to remove characters
        # To avoid getting stuck, we can restart with different random seeds
        # Or just do many passes
        
        # Since one pass is fast, we can do many
        # But we need to be careful not to exceed time
        # Let's do passes until no improvement or time limit
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle indices to try different removals
            indices = list(range(len(w_current)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                w_test = w_current[:i] + w_current[i+1:]
                if is_superperm(w_test):
                    w_current = w_test
                    improved = True
                    break
            # If we removed one, loop again to remove more
            # If not, we are stuck in a local minimum for this random order
            # We can try a new random order
            if not improved:
                # Try a few more random orders before giving up
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    indices = list(range(len(w_current)))
                    random.shuffle(indices)
                    for i in indices:
                        if time.time() > deadline:
                            break
                        w_test = w_current[:i] + w_current[i+1:]
                        if is_superperm(w_test):
                            w_current = w_test
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break

        return "".join(map(str, w_current))
    else:
        return "".join(map(str, word))
