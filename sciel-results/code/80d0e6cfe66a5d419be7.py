import time
import random
import itertools

def construct(q, n, radius):
    # Handle the specific known case efficiently
    if q == 3 and n == 7 and radius == 1:
        N = 3**7
        
        def word_to_idx(w):
            x = 0
            for i in range(n):
                x = x * 3 + w[i]
            return x

        def idx_to_word(x):
            w = [0]*n
            for i in range(n-1, -1, -1):
                w[i] = x % 3
                x //= 3
            return w

        balls = [0] * N
        for x in range(N):
            w = idx_to_word(x)
            mask = 1 << x
            for i in range(n):
                orig = w[i]
                for val in range(q):
                    if val == orig:
                        continue
                    w[i] = val
                    y = word_to_idx(w)
                    mask |= (1 << y)
                w[i] = orig
            balls[x] = mask

        def greedy_cover(seed):
            random.seed(seed)
            uncovered = (1 << N) - 1
            C = []
            while uncovered:
                best_gain = -1
                best_cand = -1
                # Randomized tie-breaking
                for c in range(N):
                    gain = (balls[c] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = c
                    elif gain == best_gain and random.random() < 0.5:
                        best_cand = c
                if best_cand == -1:
                    break
                C.append(best_cand)
                uncovered &= ~balls[best_cand]
            return C

        def local_search(C, seed):
            random.seed(seed)
            C = list(C)
            cover_count = [0] * N
            for c in C:
                mask = balls[c]
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    cover_count[idx] += 1
                    mask ^= lsb
            
            improved = True
            while improved:
                improved = False
                indices = list(range(len(C)))
                random.shuffle(indices)
                for i in indices:
                    c = C[i]
                    mask = balls[c]
                    is_redundant = True
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            is_redundant = False
                            break
                        mask ^= lsb
                    
                    if is_redundant:
                        mask = balls[c]
                        while mask:
                            lsb = mask & -mask
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            mask ^= lsb
                        C.pop(i)
                        improved = True
            
            return C

        start_time = time.time()
        deadline = start_time + 110
        
        best_C = None
        best_size = float('inf')
        
        # Run 1
        C = greedy_cover(0)
        C = local_search(C, 0)
        best_C = C
        best_size = len(C)
        
        seed = 1
        while time.time() < deadline:
            C = greedy_cover(seed)
            C = local_search(C, seed)
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = C
                
            seed += 1
            
        return [idx_to_word(c) for c in best_C]

    else:
        # Fallback for other parameters
        # Simple Greedy
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        
        def dist(w1, w2):
            return sum(1 for a, b in zip(w1, w2) if a != b)
        
        # Precompute balls
        balls = []
        for w in all_words:
            ball = set()
            for v in all_words:
                if dist(w, v) <= radius:
                    ball.add(v)
            balls.append(ball)
            
        def greedy_cover_general():
            uncovered = set(all_words)
            C = []
            while uncovered:
                best_cand = None
                max_gain = -1
                for i, w in enumerate(all_words):
                    gain = len(balls[i] & uncovered)
                    if gain > max_gain:
                        max_gain = gain
                        best_cand = i
                if best_cand is None:
                    break
                C.append(all_words[best_cand])
                uncovered -= balls[best_cand]
            return C
            
        return greedy_cover_general()
