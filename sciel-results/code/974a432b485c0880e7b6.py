import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute all words and their spheres
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute spheres (indices of neighbors)
    spheres = []
    for w in words:
        sphere = [word_to_idx[tuple(w)]]
        for i in range(n):
            for v in range(q):
                if v != w[i]:
                    new_w = w.copy()
                    new_w[i] = v
                    sphere.append(word_to_idx[tuple(new_w)])
        spheres.append(sphere)
    
    total_words = len(words)
    
    # Precompute reverse spheres
    # reverse_spheres[w] = list of c such that w in spheres[c]
    reverse_spheres = [[] for _ in range(total_words)]
    for c_idx, s in enumerate(spheres):
        for w_idx in s:
            reverse_spheres[w_idx].append(c_idx)

    def greedy_covering_optimized(seed_val):
        random.seed(seed_val)
        
        global_cover_count = [0] * total_words
        uncovered_count = [len(s) for s in spheres]
        
        code_indices = []
        remaining_uncovered = total_words
        
        while remaining_uncovered > 0:
            max_count = max(uncovered_count)
            # Find all candidates with max_count
            candidates = []
            for i in range(total_words):
                if uncovered_count[i] == max_count:
                    candidates.append(i)
            
            if not candidates:
                break
            
            c_idx = random.choice(candidates)
            code_indices.append(c_idx)
            
            # Update
            for w_idx in spheres[c_idx]:
                if global_cover_count[w_idx] == 0:
                    remaining_uncovered -= 1
                    for c_prime in reverse_spheres[w_idx]:
                        uncovered_count[c_prime] -= 1
                global_cover_count[w_idx] += 1
        
        return [words[i] for i in code_indices]

    def greedy_deletion(code):
        if not code:
            return code
            
        code_indices = [word_to_idx[tuple(c)] for c in code]
        
        # Build coverage counts
        cover_count = [0] * total_words
        for c_idx in code_indices:
            for w_idx in spheres[c_idx]:
                cover_count[w_idx] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code_indices)):
                c_idx = code_indices[i]
                
                # Tentatively remove
                for w_idx in spheres[c_idx]:
                    cover_count[w_idx] -= 1
                
                # Check validity
                valid = True
                for w_idx in spheres[c_idx]:
                    if cover_count[w_idx] == 0:
                        valid = False
                        break
                
                if valid:
                    # Keep removed
                    code_indices.pop(i)
                    changed = True
                    break # Restart loop as indices changed
                else:
                    # Restore
                    for w_idx in spheres[c_idx]:
                        cover_count[w_idx] += 1
        
        return [words[i] for i in code_indices]

    best_code = None
    best_len = float('inf')
    
    attempts = 0
    
    while time.time() < deadline:
        seed = attempts
        code = greedy_covering_optimized(seed)
        code = greedy_deletion(code)
        
        # Local search / Swaps
        if time.time() < deadline - 10:
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                if not code: break
                idx = random.randrange(len(code))
                old_c = code[idx]
                old_c_idx = word_to_idx[tuple(old_c)]
                
                # Try a random new codeword
                new_c_idx = random.randrange(total_words)
                new_c = words[new_c_idx]
                
                if new_c_idx == old_c_idx:
                    continue
                    
                # Check if replacement is valid
                cover_count = [0] * total_words
                for c in code:
                    c_idx = word_to_idx[tuple(c)]
                    for w_idx in spheres[c_idx]:
                        cover_count[w_idx] += 1
                
                # Subtract old
                for w_idx in spheres[old_c_idx]:
                    cover_count[w_idx] -= 1
                
                # Add new
                for w_idx in spheres[new_c_idx]:
                    cover_count[w_idx] += 1
                
                # Check if all covered
                valid = True
                for w_idx in range(total_words):
                    if cover_count[w_idx] == 0:
                        valid = False
                        break
                
                if valid:
                    code[idx] = new_c
                    code = greedy_deletion(code)
                    improved = True
                    break
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
            if best_len <= 73:
                # We hit the known best. Keep searching for better if time permits.
                pass
        
        attempts += 1
        
    return best_code
