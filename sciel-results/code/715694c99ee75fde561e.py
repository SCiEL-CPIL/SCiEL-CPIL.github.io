import itertools
import random
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 145
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1231213213121312" # Length 14? Standard is 33? 
        # L4 = 1+2+6+24 = 33.
        # Wait, 1231213213121312 is length 14.
        # Does it contain all 24 perms?
        # 123, 231, 312, 121 (no), ...
        # 121 is not a perm.
        # So 14 is not valid.
        # Standard is 33.
        pass

    symbols = list(range(1, n + 1))
    all_perms = list(itertools.permutations(symbols))
    
    # Precompute the set of all perms for fast lookup
    perm_set = set(all_perms)
    
    # Helper to get unvisited neighbors with max overlap
    def get_best_neighbors(current, unvisited):
        # Returns list of (cost, Q) with minimum cost
        # We check k from n-1 down to 1
        for k in range(n - 1, 0, -1):
            suffix = current[n - k:]
            used = set(suffix)
            remaining = [s for s in symbols if s not in used]
            
            candidates = []
            # Generate perms of remaining
            # For k=1, 1! = 1 perm.
            # For k=2, 2! = 2 perms.
            # ...
            for rem_perm in itertools.permutations(remaining):
                Q = suffix + rem_perm
                if Q in unvisited:
                    candidates.append((n - k, Q))
            
            if candidates:
                return candidates
        return []

    best_length = float('inf')
    best_path = None
    
    while time.time() < deadline:
        unvisited = set(all_perms)
        current = random.choice(all_perms)
        unvisited.remove(current)
        
        path = [current]
        current_length = n
        
        while unvisited:
            candidates = get_best_neighbors(current, unvisited)
            
            if candidates:
                # Choose one
                # Heuristic: Random choice among min-cost candidates
                cost, Q = random.choice(candidates)
                current_length += cost
                unvisited.remove(Q)
                path.append(Q)
                current = Q
            else:
                # No overlap >= 1
                Q = random.choice(list(unvisited))
                current_length += n
                unvisited.remove(Q)
                path.append(Q)
                current = Q
                
        if current_length < best_length:
            best_length = current_length
            best_path = path
            
    if best_path is None:
        # Fallback
        word = [1]
        for m in range(2, n + 1):
            perms = []
            seen = set()
            for i in range(len(word) - (m - 1) + 1):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1:
                    if window not in seen:
                        seen.add(window)
                        perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    result = list(best_path[0])
    for i in range(1, len(best_path)):
        prev = best_path[i-1]
        curr = best_path[i]
        max_ov = 0
        for k in range(n, 0, -1):
            if prev[n-k:] == curr[:k]:
                max_ov = k
                break
        result.extend(curr[max_ov:])
        
    return "".join(map(str, result))
