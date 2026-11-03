import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate Standard Superpermutation
    # S_1 = "1"
    # S_n is formed by iterating through permutations of 1..n-1 in S_{n-1}
    # and inserting n.
    
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique (m-1)-perms from current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            max_overlap = 0
            # Optimization: check only possible overlaps
            # Suffix of new, Prefix of block
            min_len = min(len(new), len(block))
            for o in range(min_len, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new
    
    best_word = word[:]
    best_len = len(word)
    
    # 2. Local Search: Randomized Greedy Deletion
    # We try to find a subsequence of length < best_len that is a valid superpermutation.
    
    def is_valid_superperm(w, n):
        # Check if all permutations are present
        # Optimization: Just check if count of all 1..n perms is > 0
        # For n=7, 5040 perms.
        # This is O(L * n) to generate windows and O(1) to add to set.
        # 5900 * 7 = 41k ops. Very fast.
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
        return len(seen) == len(range(1, n+1)) * 1 # Actually need to check if it covers all.
        # Wait, len(seen) == n! is the condition.
        # But checking len(seen) == n! is sufficient if we know it's a subset.
        # Since it's a superpermutation candidate, it should be.
        # However, to be safe, we should ensure it contains *all* permutations.
        # Generating all permutations is expensive (5040 items).
        # But we can just check len(seen) == math.factorial(n).
        
    import math
    target_perms = math.factorial(n)
    
    def try_shorten(w, n, deadline):
        # w is a list of ints
        # Maintain perm counts
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            counts[tuple(w[i:i+n])] += 1
        
        # List of valid indices to delete
        # An index i is valid if all windows covering it have count > 1
        # Windows covering i: [i-n+1, i]
        
        # We will loop until no more deletions or deadline
        while time.time() < deadline:
            valid_indices = []
            # Scan all indices
            for i in range(len(w)):
                # Check windows k in [max(0, i-n+1), min(i, len(w)-n)]
                safe = True
                for k in range(max(0, i - n + 1), min(i, len(w) - n) + 1):
                    p = tuple(w[k:k+n])
                    if counts[p] <= 1:
                        safe = False
                        break
                if safe:
                    valid_indices.append(i)
            
            if not valid_indices:
                break
            
            # Pick a random valid index
            # Heuristic: Pick one that is "most redundant" (involved in high count perms)
            # Or just random to escape local minima
            idx = random.choice(valid_indices)
            
            # Perform deletion
            # Update counts
            # Old windows affected: k in [idx-n+1, idx]
            for k in range(max(0, idx - n + 1), min(idx, len(w) - n) + 1):
                p = tuple(w[k:k+n])
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            del w[idx]
            
            # New windows affected: k in [idx-n+1, idx] in NEW word
            # Note: indices >= idx are shifted left by 1.
            # The new windows are those that now cover the "gap" or are shifted into the gap.
            # The gap is at idx.
            # Windows starting at k in [max(0, idx-n+1), min(idx, len(w)-n)]
            for k in range(max(0, idx - n + 1), min(idx, len(w) - n) + 1):
                p = tuple(w[k:k+n])
                counts[p] += 1
        
        return w

    # Run multiple restarts
    # Each restart starts from the standard word
    # We limit the number of restarts to ensure we finish in time
    max_restarts = 5000
    restart_time = (deadline - start_time) / max_restarts
    
    for r in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        w_copy = best_word[:]
        # Run shortening
        shortened = try_shorten(w_copy, n, deadline - 1)
        
        if len(shortened) < best_len:
            # Verify validity
            # Fast check: length must be >= lower bound?
            # Just verify it's a superperm
            seen = set()
            for i in range(len(shortened) - n + 1):
                seen.add(tuple(shortened[i:i+n]))
            if len(seen) == target_perms:
                best_len = len(shortened)
                best_word = shortened[:]
                # If we reached the theoretical lower bound or known best, stop?
                if best_len <= 5905:
                    break

    return "".join(map(str, best_word))
