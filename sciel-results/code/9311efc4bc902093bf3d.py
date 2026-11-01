import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute all words
    words = np.array([list(map(int, np.base_repr(i, q).zfill(n))) for i in range(total)], dtype=np.int8)
    
    # Precompute balls
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for i in range(total):
        w = words[i]
        balls[i, 0] = i
        idx = 1
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                balls[i, idx] = new_idx
                idx += 1
                
    def get_counts(code_set):
        counts = np.zeros(total, dtype=np.int32)
        for c in code_set:
            counts[balls[c]] += 1
        return counts
            
    def deletion_pass(code_set, counts):
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                if np.min(counts[balls[c]]) >= 2:
                    code_set.remove(c)
                    counts[balls[c]] -= 1
                    changed = True
        return code_set, counts

    def greedy_build(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        covered = np.zeros(total, dtype=bool)
        code = []
        
        while not np.all(covered):
            new_cov = np.sum(~covered[balls], axis=1)
            max_cov = np.max(new_cov)
            if max_cov == 0:
                break
            candidates = np.where(new_cov == max_cov)[0]
            if randomize and len(candidates) > 1:
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]
            code.append(best_idx)
            covered[balls[best_idx]] = True
            
        return code
        
    def local_search(code_set):
        counts = get_counts(code_set)
        code_set, counts = deletion_pass(code_set, counts)
        
        # Swap moves
        # Precompute ball sets for fast subset check?
        # ball_size is 15. P_A size is <= 15.
        # We can just iterate B.
        
        iterations = 0
        max_iter = 200
        while iterations < max_iter and time.time() < deadline - 5.0:
            iterations += 1
            if not code_set:
                break
                
            A = random.choice(list(code_set))
            # Unique points of A
            unique_mask = counts[balls[A]] == 1
            unique_points = balls[A][unique_mask]
            
            if len(unique_points) == 0:
                # Removable, but deletion pass should have caught it.
                # Force remove
                code_set.remove(A)
                counts[balls[A]] -= 1
                code_set, counts = deletion_pass(code_set, counts)
                continue
                
            # Find B
            # B must cover all unique_points
            # And B not in code_set
            found_B = -1
            # To speed up, we can just check random B's
            for _ in range(50): # Try 50 random B's
                B = random.randrange(total)
                if B in code_set:
                    continue
                # Check if ball(B) covers unique_points
                # We can use a set for ball(B)
                set_B = set(balls[B].tolist())
                # Check if all unique_points are in set_B
                if all(p in set_B for p in unique_points):
                    found_B = B
                    break
                    
            if found_B == -1:
                continue
                
            # Perform swap
            code_set.remove(A)
            counts[balls[A]] -= 1
            code_set.add(found_B)
            counts[balls[found_B]] += 1
            
            # Run deletion pass
            old_size = len(code_set)
            code_set, counts = deletion_pass(code_set, counts)
            
            # If size didn't decrease, revert with some probability to allow exploration
            if len(code_set) > old_size:
                # Revert
                code_set.remove(found_B)
                counts[balls[found_B]] -= 1
                code_set.add(A)
                counts[balls[A]] += 1
                # Re-run deletion to stabilize
                code_set, counts = deletion_pass(code_set, counts)
                
        return code_set, counts

    best_code_set = None
    best_size = total
    
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        if restarts % 3 == 0:
            code = greedy_build(randomize=True, seed=restarts)
        else:
            code = greedy_build(randomize=False)
            
        code_set = set(code)
        code_set, _ = local_search(code_set)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code_set = set(code_set)
            
    if best_code_set is None:
        # Fallback
        best_code_set = set(greedy_build())
        
    # Final polish
    best_code_set, _ = local_search(best_code_set)
    
    result = []
    for idx in best_code_set:
        result.append(words[idx].tolist())
        
    return result
