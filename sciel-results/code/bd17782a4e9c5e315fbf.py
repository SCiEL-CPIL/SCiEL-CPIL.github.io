import time
import random
import sys

def construct(q, n, radius):
    # This problem is specific to q=3, n=6, radius=1
    # If parameters differ, fall back to a simple construction or raise error
    if q != 3 or n != 6 or radius != 1:
        # Fallback: Direct sum or similar
        # For generality, we can try a greedy construction
        pass # We will implement the specific solver for the known task
        
    # Specific solver for q=3, n=6, r=1
    total_words = 3 ** 6
    # Map word tuple to index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * 3 + w[i]
        return idx
        
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    # Precompute neighbors
    # neighbors[idx] = list of indices within distance 1
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors[idx].append(idx) # distance 0
        for i in range(n):
            orig = w[i]
            for val in range(3):
                if val == orig:
                    continue
                w[i] = val
                neighbors[idx].append(word_to_idx(w))
            w[i] = orig
            
    # Greedy Initialization
    def greedy_init():
        uncovered = [1] * total_words
        code = []
        remaining = total_words
        while remaining > 0:
            best_idx = -1
            best_cover = -1
            # Find word covering most uncovered
            for i in range(total_words):
                if uncovered[i]:
                    # Count how many of its neighbors are uncovered
                    count = 0
                    for nb in neighbors[i]:
                        if uncovered[nb]:
                            count += 1
                    if count > best_cover:
                        best_cover = count
                        best_idx = i
            if best_idx == -1:
                break
            # Add best_idx to code
            code.append(best_idx)
            # Update uncovered
            for nb in neighbors[best_idx]:
                if uncovered[nb]:
                    uncovered[nb] = 0
                    remaining -= 1
        return code

    code = greedy_init()
    best_code = list(code)
    best_size = len(code)
    
    # Coverage counts
    # cov[i] = number of codewords in code that cover word i
    cov = [0] * total_words
    for c in code:
        for nb in neighbors[c]:
            cov[nb] += 1
            
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    def try_reduce(code, cov):
        """Remove codewords that are not essential."""
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to handle deletions
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if c can be removed
                essential = False
                for nb in neighbors[c]:
                    if cov[nb] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove c
                    code.pop(i)
                    for nb in neighbors[c]:
                        cov[nb] -= 1
                    changed = True
        return code, cov

    # Run initial reduction
    code, cov = try_reduce(code, cov)
    if len(code) < best_size:
        best_size = len(code)
        best_code = list(code)
        
    # Iterated Local Search
    while time.time() < deadline:
        # Perturbation: Remove a random codeword and repair
        # Copy current state
        curr_code = list(code)
        curr_cov = list(cov)
        
        if not curr_code:
            break
            
        # Pick a random codeword to remove
        rand_idx = random.randint(0, len(curr_code) - 1)
        c_remove = curr_code[rand_idx]
        
        # Remove it
        curr_code.pop(rand_idx)
        for nb in neighbors[c_remove]:
            curr_cov[nb] -= 1
            
        # Find holes
        holes = []
        for i in range(total_words):
            if curr_cov[i] == 0:
                holes.append(i)
                
        # Repair: Add codewords to cover holes
        while holes:
            # Find best word to add
            best_add = -1
            best_cover_holes = -1
            # We can restrict search to words that cover at least one hole
            # To be fast, we might just scan all words, but that's 729 * len(holes)
            # Since holes are likely small after removing 1 word, this is fast.
            # Actually, removing 1 word creates at most 13 holes (neighbors of removed word)
            # So we only need to check words that cover these specific holes.
            
            # Optimization: Only consider candidates that cover at least one hole
            candidates = set()
            for h in holes:
                # Who covers h?
                # We don't have inverse neighbors precomputed.
                # But we can just iterate all 729 words and check intersection with holes.
                # 729 * 13 is ~9000 ops. Very fast.
                pass
                
            # Simpler: Iterate all words, count coverage of holes
            for i in range(total_words):
                count = 0
                for nb in neighbors[i]:
                    if curr_cov[nb] == 0: # If it covers a hole
                        count += 1
                if count > best_cover_holes:
                    best_cover_holes = count
                    best_add = i
                    
            if best_add == -1:
                break # Should not happen if space is connected
                
            # Add best_add
            curr_code.append(best_add)
            for nb in neighbors[best_add]:
                curr_cov[nb] += 1
            # Update holes: remove covered ones
            new_holes = []
            for h in holes:
                if curr_cov[h] == 0:
                    new_holes.append(h)
            holes = new_holes
            
        # Reduce again
        curr_code, curr_cov = try_reduce(curr_code, curr_cov)
        
        if len(curr_code) < best_size:
            best_size = len(curr_code)
            best_code = list(curr_code)
            code = list(curr_code)
            cov = list(curr_cov)
        else:
            # Keep current best as the working state? Or revert?
            # ILS usually keeps the last accepted state or reverts to best.
            # Reverting to best helps avoid drifting to bad local minima.
            # But sometimes accepting equal size with perturbation helps escape.
            # Let's stick to best for stability, but we already perturbed from 'code' which was the last accepted.
            # So 'code' remains the last accepted.
            pass

    # Convert best_code indices back to lists
    return [idx_to_word(idx) for idx in best_code]

# Fallback for other parameters
def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _construct_q3_n6_r1()
    else:
        # Simple greedy for other cases
        total = q ** n
        def w2i(w):
            idx = 0
            for x in w:
                idx = idx * q + x
            return idx
        def i2w(idx):
            w = [0] * n
            for i in range(n-1, -1, -1):
                w[i] = idx % q
                idx //= q
            return w
            
        neighbors = [[] for _ in range(total)]
        for idx in range(total):
            w = i2w(idx)
            neighbors[idx].append(idx)
            for i in range(n):
                orig = w[i]
                for val in range(q):
                    if val == orig: continue
                    w[i] = val
                    neighbors[idx].append(w2i(w))
                w[i] = orig
        
        uncovered = [1] * total
        code = []
        while any(uncovered):
            best_idx = -1
            best_c = -1
            for i in range(total):
                if uncovered[i]:
                    c = 0
                    for nb in neighbors[i]:
                        if uncovered[nb]: c += 1
                    if c > best_c:
                        best_c = c
                        best_idx = i
            if best_idx == -1: break
            code.append(best_idx)
            for nb in neighbors[best_idx]:
                if uncovered[nb]:
                    uncovered[nb] = 0
        return [i2w(idx) for idx in code]

def _construct_q3_n6_r1():
    # Move the specific logic here to keep construct clean
    import time
    import random
    
    total_words = 729
    n = 6
    q = 3
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * 3 + w[i]
        return idx
        
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors[idx].append(idx)
        for i in range(n):
            orig = w[i]
            for val in range(3):
                if val == orig:
                    continue
                w[i] = val
                neighbors[idx].append(word_to_idx(w))
            w[i] = orig
            
    def greedy_init():
        uncovered = [1] * total_words
        code = []
        remaining = total_words
        while remaining > 0:
            best_idx = -1
            best_cover = -1
            for i in range(total_words):
                if uncovered[i]:
                    count = 0
                    for nb in neighbors[i]:
                        if uncovered[nb]:
                            count += 1
                    if count > best_cover:
                        best_cover = count
                        best_idx = i
            if best_idx == -1:
                break
            code.append(best_idx)
            for nb in neighbors[best_idx]:
                if uncovered[nb]:
                    uncovered[nb] = 0
                    remaining -= 1
        return code

    code = greedy_init()
    best_code = list(code)
    best_size = len(code)
    
    cov = [0] * total_words
    for c in code:
        for nb in neighbors[c]:
            cov[nb] += 1
            
    start_time = time.time()
    deadline = 145.0
    
    def try_reduce(code, cov):
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                essential = False
                for nb in neighbors[c]:
                    if cov[nb] == 1:
                        essential = True
                        break
                if not essential:
                    code.pop(i)
                    for nb in neighbors[c]:
                        cov[nb] -= 1
                    changed = True
        return code, cov

    code, cov = try_reduce(code, cov)
    if len(code) < best_size:
        best_size = len(code)
        best_code = list(code)
        
    while time.time() < deadline:
        curr_code = list(code)
        curr_cov = list(cov)
        
        if not curr_code:
            break
            
        rand_idx = random.randint(0, len(curr_code) - 1)
        c_remove = curr_code[rand_idx]
        
        curr_code.pop(rand_idx)
        for nb in neighbors[c_remove]:
            curr_cov[nb] -= 1
            
        holes = [i for i in range(total_words) if curr_cov[i] == 0]
        
        while holes:
            best_add = -1
            best_cover_holes = -1
            for i in range(total_words):
                count = 0
                for nb in neighbors[i]:
                    if curr_cov[nb] == 0:
                        count += 1
                if count > best_cover_holes:
                    best_cover_holes = count
                    best_add = i
                    
            if best_add == -1:
                break
                
            curr_code.append(best_add)
            for nb in neighbors[best_add]:
                curr_cov[nb] += 1
            holes = [h for h in holes if curr_cov[h] == 0]
            
        curr_code, curr_cov = try_reduce(curr_code, curr_cov)
        
        if len(curr_code) < best_size:
            best_size = len(curr_code)
            best_code = list(curr_code)
            code = list(curr_code)
            cov = list(curr_cov)

    return [idx_to_word(idx) for idx in best_code]
