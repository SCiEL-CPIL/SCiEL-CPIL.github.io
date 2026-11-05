import time
import random
import itertools

def max_sidon_size(S):
    """
    Returns the size of the largest subset of S where all unordered pair sums
    (including diagonals i.e. x+x) are unique.
    """
    S = sorted(list(S))
    n = len(S)
    if n <= 1:
        return n
    
    best = 1
    # Precompute sums for quick lookup? 
    # We will use a set 'used_sums' in the DFS.
    
    # To optimize DFS, we can precompute the sums of each element with all subsequent elements?
    # No, the sums depend on the current subset.
    
    # DFS with backtracking
    # State: idx (next index to consider), current_size, used_sums (set of int)
    
    def dfs(idx, size, used_sums):
        nonlocal best
        if size > best:
            best = size
        
        # Pruning: even if we take all remaining, can we beat best?
        if size + (n - idx) <= best:
            return
            
        if idx == n:
            return

        # Option 1: Skip S[idx]
        dfs(idx + 1, size, used_sums)
        
        # Option 2: Take S[idx]
        # Check if S[idx] can be added
        # Sums to check: S[idx] + S[j] for all j in current_subset?
        # We don't track current_subset indices explicitly in this signature, 
        # but we can infer them or pass them. 
        # Passing indices is better for checking sums.
        
        # Let's refactor to pass current indices.
        pass

    # Refactored DFS
    def dfs2(start, size, current_indices, used_sums):
        nonlocal best
        if size > best:
            best = size
            
        if size + (n - start) <= best:
            return
        
        for i in range(start, n):
            # Try adding S[i]
            # Check conflicts with existing elements in current_indices
            # New sums: S[i] + S[j] for j in current_indices, and S[i] + S[i]
            
            conflict = False
            new_sums = []
            
            # Check self sum
            s_self = S[i] + S[i]
            if s_self in used_sums:
                conflict = True
            else:
                new_sums.append(s_self)
                
            if not conflict:
                for j in current_indices:
                    s_pair = S[i] + S[j]
                    if s_pair in used_sums:
                        conflict = True
                        break
                    new_sums.append(s_pair)
            
            if not conflict:
                # Add to used_sums
                for s in new_sums:
                    used_sums.add(s)
                
                dfs2(i + 1, size + 1, current_indices + [i], used_sums)
                
                # Backtrack
                for s in new_sums:
                    used_sums.remove(s)
    
    dfs2(0, 0, [], set())
    return best

def _valid(c):
    if len(c) != 12 or len(set(c)) != 12:
        return False
    return all(0 <= x <= 128 for x in c)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = None
    best_val = 12  # Max possible is 12
    
    seen = set()
    
    def consider(c):
        nonlocal best_set, best_val
        key = tuple(sorted(c))
        if key in seen:
            return
        seen.add(key)
        if not _valid(c):
            return
        v = max_sidon_size(c)
        if v < best_val:
            best_val = v
            best_set = list(c)
        return v

    # 1. Try the most "collision-dense" sets: Small intervals
    # {0..11} is the densest.
    consider(list(range(12)))
    
    # 2. Try other small intervals
    for lo in range(0, 120):
        if time.time() > deadline:
            break
        c = list(range(lo, lo + 12))
        consider(c)
        
    # 3. Random search in small windows
    # Sets in small windows have more collisions.
    rng = random.Random(42)
    for _ in range(20000):
        if time.time() > deadline:
            break
        # Window size 12 to 25
        width = rng.randint(12, 25)
        lo = rng.randint(0, 128 - width)
        hi = lo + width
        c = sorted(rng.sample(range(lo, hi), 12))
        consider(c)
        
    # 4. Structured sets: Union of two small APs
    # e.g., {0, 1, 2, 3, 4, 5} U {6, 7, 8, 9, 10, 11} is just 0..11
    # Try {0, 2, 4, 6, 8, 10} U {1, 3, 5, 7, 9, 11} -> 0..11
    # Try {0, 1, 2, 3, 4, 5} U {10, 11, 12, 13, 14, 15}
    for d1 in range(1, 5):
        for d2 in range(1, 5):
            for a in range(0, 120):
                if time.time() > deadline:
                    break
                c1 = [a + i * d1 for i in range(6)]
                # Shift second AP
                for shift in range(1, 15):
                    b = a + shift
                    c2 = [b + i * d2 for i in range(6)]
                    c = sorted(set(c1 + c2))
                    if len(c) == 12:
                        consider(c)
                    if time.time() > deadline:
                        break
                        
    # 5. Local search
    if best_set is not None:
        cur = list(best_set)
        cur_val = best_val
        for _ in range(5000):
            if time.time() > deadline:
                break
            c2 = list(cur)
            pos = rng.randrange(12)
            # Try to move to a value that creates more collisions
            # Usually closer to other elements
            delta = rng.randint(-5, 5)
            nv = c2[pos] + delta
            if nv < 0 or nv > 128:
                continue
            if nv in c2:
                continue
            c2[pos] = nv
            c2.sort()
            v = max_sidon_size(c2)
            if v < cur_val:
                cur = c2
                cur_val = v
            elif v == cur_val and rng.random() < 0.2:
                cur = c2
        consider(cur)
        
    if best_set is None:
        best_set = list(range(12))
        
    return best_set
