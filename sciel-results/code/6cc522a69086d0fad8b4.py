import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit
    
    total_words = q ** n
    # Precompute neighbors for radius 1
    # neighbors[i] is a list of indices of words within Hamming distance 1 from i
    neighbors = [[] for _ in range(total_words)]
    
    # Helper to convert index to tuple and back
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[i] = idx % q
            idx //= q
        return tuple(t)
        
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Build neighbor lists
    # For each word, add itself and all words at distance 1
    for idx in range(total_words):
        neighbors[idx].append(idx)
        t = idx_to_tuple(idx)
        for pos in range(n):
            for val in range(q):
                if val == t[pos]:
                    continue
                new_t = list(t)
                new_t[pos] = val
                neighbors[idx].append(tuple_to_idx(tuple(new_t)))
        
    # Greedy Set Cover with Restart
    best_code = None
    best_size = float('inf')
    
    # Initial seed: The known 81-word code or random
    # Let's try a few different starting points
    # 1. Random initialization
    # 2. Greedy from scratch
    
    def greedy_cover(initial_code):
        code = list(set(initial_code))
        covered = [False] * total_words
        for c in code:
            for nb in neighbors[c]:
                covered[nb] = True
                
        uncovered_count = sum(1 for c in covered if not c)
        
        # If initial code doesn't cover everything, add greedily
        while uncovered_count > 0:
            # Find candidate that covers most uncovered words
            max_covered = 0
            best_cand = -1
            for c in range(total_words):
                # Optimization: if c is already in code, skip? 
                # Actually, we want to pick from ALL possible codewords.
                # But if it's already in code, it covers 0 NEW uncovered words (by definition of covered).
                # So we can just iterate all.
                cnt = 0
                for nb in neighbors[c]:
                    if not covered[nb]:
                        cnt += 1
                if cnt > max_covered:
                    max_covered = cnt
                    best_cand = c
            if best_cand == -1:
                break # Should not happen if space is connected
            code.append(best_cand)
            for nb in neighbors[best_cand]:
                if not covered[nb]:
                    covered[nb] = True
                    uncovered_count -= 1
        
        # Redundancy removal
        # Try to remove each word if possible
        # Repeat until no more removals
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                c = code[i]
                # Check if c is redundant
                # A word w is covered by c if w in neighbors[c].
                # If we remove c, is every w in neighbors[c] still covered by some other c' in code?
                redundant = True
                for w in neighbors[c]:
                    # Check if w is covered by any other codeword
                    found = False
                    for j in range(len(code)):
                        if i == j:
                            continue
                        if w in neighbors[code[j]]:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    code.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        
        return code

    # We will run multiple restarts
    # Strategy 1: Start with empty set, pure greedy
    # Strategy 2: Start with random set of size ~75
    
    # Pure greedy is deterministic. Let's do it once.
    if time.time() - start_time < timeout:
        code = greedy_cover([])
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
    
    # Randomized restarts
    # We want to escape local minima of pure greedy.
    # Pure greedy usually gets stuck around 73-75.
    # Let's try adding random noise or starting with a random subset.
    
    for restart in range(500):
        if time.time() - start_time > timeout:
            break
        
        # Random initial set
        # Size around best known or slightly larger
        init_size = max(10, best_size - 5 if best_size > 10 else 10)
        initial_code = random.sample(range(total_words), init_size)
        
        code = greedy_cover(initial_code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            # If we hit 71 or 72, we can stop early? No, try to beat 73.
            if best_size <= 71:
                break

    if best_code is None:
        # Fallback to the original 81-word construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code_ints = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = h + list(tail)
                idx = 0
                for v in w:
                    idx = idx * 3 + v
                code_ints.append(idx)
        best_code = code_ints
        best_size = len(best_code)

    # Convert best_code (list of ints) to list of lists of ints
    result = []
    for idx in best_code:
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        result.append(w)
        
    return result
