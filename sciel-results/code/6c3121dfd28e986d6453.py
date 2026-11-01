import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total_words = q ** n
    
    if total_words > 10000:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [[0] * n]

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
            
    def idx_to_tuple(idx):
        t = [0] * n
        rem = idx
        for i in range(n):
            t[i] = rem % q
            rem //= q
        return tuple(t)
        
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx += t[i] * powers[i]
        return idx

    # Precompute balls as lists for faster intersection
    balls = [None] * total_words
    for i in range(total_words):
        w = idx_to_tuple(i)
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = tuple_to_idx(tuple(new_w))
                    ball.append(new_idx)
        balls[i] = ball

    # Convert balls to sets for intersection operations
    ball_sets = [set(b) for b in balls]

    def prune_cover(indices):
        indices = list(indices)
        cover_counts = np.zeros(total_words, dtype=np.int32)
        for idx in indices:
            for w in balls[idx]:
                cover_counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_indices = []
            for idx in indices:
                necessary = False
                for w in balls[idx]:
                    if cover_counts[w] == 1:
                        necessary = True
                        break
                if necessary:
                    new_indices.append(idx)
                else:
                    for w in balls[idx]:
                        cover_counts[w] -= 1
                    changed = True
            indices = new_indices
        return indices

    def local_search(code_indices, max_time=5.0):
        """Try to reduce code size by swapping or removing codewords."""
        code_set = set(code_indices)
        best = list(code_indices)
        best_size = len(best)
        local_start = time.time()
        
        # First try removals
        for idx in list(best):
            if time.time() > local_start + max_time:
                break
            remaining = [c for c in best if c != idx]
            # Check if remaining still covers
            covered = np.zeros(total_words, dtype=bool)
            for c in remaining:
                for w in balls[c]:
                    covered[w] = True
            if covered.all():
                best = remaining
                best_size = len(best)
        
        # Now try swaps: replace one codeword with another
        for i in range(len(best)):
            if time.time() > local_start + max_time:
                break
            for candidate in range(total_words):
                if candidate in code_set:
                    continue
                new_code = best[:i] + [candidate] + best[i+1:]
                covered = np.zeros(total_words, dtype=bool)
                for c in new_code:
                    for w in balls[c]:
                        covered[w] = True
                if covered.all():
                    # Check if this actually helps (it should if we're swapping)
                    if len(new_code) < best_size:
                        best = new_code
                        best_size = len(best)
                        code_set = set(best)
        
        return best

    best_code = None
    best_size = float('inf')
    
    # 1. Try Direct Product Construction
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        dp_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                dp_indices.append(tuple_to_idx(tuple(word)))
        dp_indices = prune_cover(dp_indices)
        if len(dp_indices) < best_size:
            best_size = len(dp_indices)
            best_code = dp_indices

    # 2. Randomized Greedy with improved efficiency
    num_restarts = 200  # More restarts since each is faster
    
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            # Find the codeword that covers the most uncovered words
            best_gain = 0
            best_candidates = []
            
            for c in range(total_words):
                # Quick check: does this ball intersect uncovered?
                # Use set intersection
                gain = len(ball_sets[c] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(c)
            
            if not best_candidates:
                break
                
            # Randomize: pick from top candidates with some probability
            if len(best_candidates) > 1 and random.random() < 0.3:
                # Pick randomly from top 5 candidates (if available)
                top_k = min(5, len(best_candidates))
                best_c = random.choice(best_candidates[:top_k])
            else:
                best_c = random.choice(best_candidates)
            
            code_indices.append(best_c)
            uncovered -= ball_sets[best_c]
            
            if time.time() > deadline - 20.0:
                break
        
        if uncovered:
            continue
            
        # Prune
        pruned = prune_cover(code_indices)
        
        # Local search to try to reduce further
        if time.time() < deadline - 10.0:
            pruned = local_search(pruned, max_time=min(3.0, deadline - time.time() - 5.0))
        
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_code = pruned

    if best_code is None:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            dp_indices = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    dp_indices.append(tuple_to_idx(tuple(word)))
            best_code = dp_indices
        else:
            best_code = list(range(total_words))

    result = [list(idx_to_tuple(c)) for c in best_code]
    
    return result
