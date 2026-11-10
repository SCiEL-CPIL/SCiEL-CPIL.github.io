import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0

    if q == 3 and n == 6 and radius == 1:
        words = list(itertools.product(range(q), repeat=n))
        num_words = len(words)
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        balls = []
        for w in words:
            w_list = list(w)
            ball_indices = [word_to_idx[w]]
            for i in range(n):
                orig_val = w_list[i]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    w_list[i] = new_val
                    ball_indices.append(word_to_idx[tuple(w_list)])
                w_list[i] = orig_val
            balls.append(ball_indices)
        
        balls_np = np.array(balls)
        
        best_code = None
        best_size = float('inf')
        
        # Try a few random restarts
        max_restarts = 20
        for _ in range(max_restarts):
            if time.time() > deadline - 10: # Keep 10s for reduction/finish
                break
                
            # Greedy with random tie-breaking
            covered = np.zeros(num_words, dtype=bool)
            code_indices = []
            
            while not covered.all():
                if time.time() > deadline - 5:
                    break
                covered_neighbors = covered[balls_np]
                sums = covered_neighbors.sum(axis=1)
                
                # Add small random noise to break ties
                noise = np.random.rand(num_words) * 0.001
                scores = sums + noise
                
                best_idx = np.argmin(scores)
                code_indices.append(best_idx)
                covered[balls_np[best_idx]] = True
                
            if not covered.all():
                continue # Timed out or failed? Shouldn't fail if space is covered.
                
            # Reduction
            code_indices = list(code_indices)
            counts = np.zeros(num_words, dtype=np.int32)
            for idx in code_indices:
                counts[balls_np[idx]] += 1
                
            changed = True
            while changed:
                if time.time() > deadline - 2:
                    break
                changed = False
                new_code_indices = []
                for idx in code_indices:
                    counts[balls_np[idx]] -= 1
                    if np.any(counts == 0):
                        counts[balls_np[idx]] += 1
                        new_code_indices.append(idx)
                    else:
                        changed = True
                
                code_indices = new_code_indices
            
            current_size = len(code_indices)
            if current_size < best_size:
                best_size = current_size
                best_code = [list(words[i]) for i in code_indices]
                
        if best_code is not None:
            return best_code
        else:
            # Fallback to seed
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
