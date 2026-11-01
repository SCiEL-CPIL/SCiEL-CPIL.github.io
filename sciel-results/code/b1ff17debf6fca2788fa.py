import numpy as np
import time
import itertools

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 7 and radius == 1):
        # Simple greedy for other cases
        N = q ** n
        if N > 10000:
             # Too large for this specific heuristic, return seed-like
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             if n >= 4:
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
             else:
                 return [list(t) for t in itertools.product(range(q), repeat=n)]
        
        # General Greedy
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        spheres = []
        for w in all_words:
            s = set()
            s.add(w)
            for i in range(n):
                for val in range(q):
                    if val == w[i]: continue
                    new_w = list(w)
                    new_w[i] = val
                    s.add(tuple(new_w))
            spheres.append([word_to_idx[x] for x in s])
            
        N = len(all_words)
        M = np.zeros((N, N), dtype=bool)
        for c in range(N):
            for w in spheres[c]:
                M[c, w] = True
                
        best_code = []
        best_size = N
        
        start_time = time.time()
        deadline = start_time + 140
        
        while time.time() < deadline:
            # Greedy
            not_covered = np.ones(N, dtype=bool)
            code = []
            while np.any(not_covered):
                scores = M @ not_covered
                best_c = np.argmax(scores)
                if scores[best_c] == 0: break
                code.append(best_c)
                not_covered[M[best_c]] = False
            
            # Deletion
            count = np.zeros(N, dtype=int)
            for c in code:
                count[spheres[c]] += 1
            changed = True
            while changed:
                changed = False
                # Shuffle order for randomness?
                idxs = list(range(len(code)))
                np.random.shuffle(idxs)
                for i in idxs:
                    c = code[i]
                    count[spheres[c]] -= 1
                    if np.any(count[spheres[c]] == 0):
                        count[spheres[c]] += 1
                    else:
                        code.pop(i)
                        changed = True
                        break # Restart deletion pass after a change to be safe/efficient? 
                        # Actually, just continue. But indices shift. 
                        # Better to restart the pass.
                        
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()
                
            # Perturb
            if len(best_code) > 0:
                # Remove 2 random codewords from best_code to create a new start
                start_code = best_code.copy()
                if len(start_code) > 2:
                    remove_idx = np.random.choice(len(start_code), 2, replace=False)
                    for i in sorted(remove_idx, reverse=True):
                        start_code.pop(i)
                else:
                    start_code = []
                
                # Extend start_code
                count = np.zeros(N, dtype=int)
                not_covered = np.ones(N, dtype=bool)
                for c in start_code:
                    count[spheres[c]] += 1
                    not_covered[spheres[c]] = False
                    
                code = start_code.copy()
                while np.any(not_covered):
                    # Only consider codewords not in code? Or any?
                    # Any is fine, M @ not_covered handles it.
                    # But we should avoid re-adding recently removed ones? 
                    # The score for already added ones will be 0 for new coverage? 
                    # No, if c is in code, its sphere is already covered, so score is 0?
                    # Wait, not_covered is False for covered words.
                    # So M[c] @ not_covered will be 0 for c in code (if fully covered).
                    # So it naturally picks new ones.
                    scores = M @ not_covered
                    best_c = np.argmax(scores)
                    if scores[best_c] == 0: break
                    code.append(best_c)
                    not_covered[M[best_c]] = False
                    # Update count? Not needed for greedy, only for deletion.
                    
                # Deletion
                count = np.zeros(N, dtype=int)
                for c in code:
                    count[spheres[c]] += 1
                changed = True
                while changed:
                    changed = False
                    idxs = list(range(len(code)))
                    np.random.shuffle(idxs)
                    for i in idxs:
                        c = code[i]
                        count[spheres[c]] -= 1
                        if np.any(count[spheres[c]] == 0):
                            count[spheres[c]] += 1
                        else:
                            code.pop(i)
                            changed = True
                            break
                            
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()

        # Convert indices back to words
        return [list(all_words[c]) for c in best_code]

    # Specific optimization for 3, 7, 1
    N = 3 ** 7
    # Precompute spheres
    spheres = []
    for c in range(N):
        w = list(np.unravel_index(c, (3, 3, 3, 3, 3, 3, 3)))
        s = [c]
        for i in range(7):
            for val in range(3):
                if val == w[i]: continue
                new_w = w.copy()
                new_w[i] = val
                new_c = np.ravel_multi_index(tuple(new_w), (3, 3, 3, 3, 3, 3, 3))
                s.append(new_c)
        spheres.append(s)
        
    M = np.zeros((N, N), dtype=bool)
    for c in range(N):
        M[c, spheres[c]] = True
        
    best_code = []
    best_size = N
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Helper for greedy extension
    def greedy_extend(initial_code):
        count = np.zeros(N, dtype=int)
        not_covered = np.ones(N, dtype=bool)
        code = initial_code.copy()
        for c in code:
            count[spheres[c]] += 1
            not_covered[spheres[c]] = False
            
        while np.any(not_covered):
            scores = M @ not_covered
            best_c = np.argmax(scores)
            if scores[best_c] == 0: break
            code.append(best_c)
            not_covered[M[best_c]] = False
            # Note: count is not updated here, but not needed for greedy selection
            # We need count for deletion later, so we'll rebuild it.
            
        return code

    def local_search(code):
        count = np.zeros(N, dtype=int)
        for c in code:
            count[spheres[c]] += 1
            
        improved = True
        while improved:
            improved = False
            idxs = list(range(len(code)))
            np.random.shuffle(idxs)
            for i in idxs:
                c = code[i]
                count[spheres[c]] -= 1
                if np.any(count[spheres[c]] == 0):
                    count[spheres[c]] += 1
                else:
                    code.pop(i)
                    improved = True
                    break
        return code

    while time.time() < deadline:
        # 1. Greedy from scratch
        code = greedy_extend([])
        code = local_search(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
        # 2. Perturb best and extend
        if len(best_code) > 2:
            start_code = best_code.copy()
            remove_idx = np.random.choice(len(start_code), 2, replace=False)
            for i in sorted(remove_idx, reverse=True):
                start_code.pop(i)
                
            code = greedy_extend(start_code)
            code = local_search(code)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()
                
    return [list(np.unravel_index(c, (3, 3, 3, 3, 3, 3, 3))) for c in best_code]
