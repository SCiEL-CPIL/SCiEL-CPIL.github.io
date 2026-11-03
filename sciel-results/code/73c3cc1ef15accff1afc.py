import time
import numpy as np
import random

def _construct_ternary_6_1():
    start_time = time.perf_counter()
    deadline = start_time + 110.0  # Safety margin
    
    q = 3
    n = 6
    total = q ** n  # 729
    
    # Precompute balls as lists of indices
    balls = []
    for idx in range(total):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        ball = [idx]
        for i in range(n):
            old_val = w[i]
            for new_val in range(q):
                if new_val != old_val:
                    power = q ** (n - 1 - i)
                    new_idx = idx + (new_val - old_val) * power
                    ball.append(new_idx)
        balls.append(ball)
    
    # Precompute distance-2 graph: edges between words at Hamming distance 2
    # For independent set on this graph
    dist2_neighbors = [[] for _ in range(total)]
    for idx in range(total):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        for i in range(n):
            for j in range(i + 1, n):
                old_i, old_j = w[i], w[j]
                for ni in range(q):
                    if ni == old_i:
                        continue
                    for nj in range(q):
                        if nj == old_j:
                            continue
                        if ni == old_i and nj == old_j:
                            continue
                        # Compute new index
                        power_i = q ** (n - 1 - i)
                        power_j = q ** (n - 1 - j)
                        new_idx = idx + (ni - old_i) * power_i + (nj - old_j) * power_j
                        if new_idx != idx:
                            dist2_neighbors[idx].append(new_idx)
    
    # Greedy independent set on distance-2 graph to get initial code
    # This ensures no two codewords are at distance 2
    in_code = np.zeros(total, dtype=bool)
    cover_count = np.zeros(total, dtype=np.int32)
    
    # Randomized greedy: repeatedly pick a random available vertex, add it, remove its neighbors
    available = set(range(total))
    random.seed(42)  # For reproducibility within the function
    
    while available:
        if time.perf_counter() > deadline:
            break
        # Pick a random available vertex
        v = random.choice(list(available))
        in_code[v] = True
        cover_count[balls[v]] += 1
        available.discard(v)
        # Remove all distance-2 neighbors from available
        for nb in dist2_neighbors[v]:
            if nb in available:
                available.discard(nb)
    
    # Check if we have a valid cover
    if np.any(cover_count == 0):
        # Add codewords to cover uncovered positions
        uncovered = np.where(cover_count == 0)[0]
        while len(uncovered) > 0:
            if time.perf_counter() > deadline:
                break
            best_idx = -1
            best_gain = 0
            for i in range(total):
                if in_code[i]:
                    continue
                gain = sum(1 for u in uncovered if u in balls[i])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0:
                break
            in_code[best_idx] = True
            cover_count[balls[best_idx]] += 1
            uncovered = np.where(cover_count == 0)[0]
    
    current_size = int(np.sum(in_code))
    best_size = current_size
    best_in_code = in_code.copy()
    
    # Local search: try to remove codewords
    while time.perf_counter() < deadline:
        current_codewords = np.where(in_code)[0]
        if len(current_codewords) == 0:
            break
        
        c_to_remove = current_codewords[np.random.randint(len(current_codewords))]
        
        # Remove it
        in_code[c_to_remove] = False
        cover_count[balls[c_to_remove]] -= 1
        
        # Check if any word is uncovered
        uncovered_words = np.where(cover_count == 0)[0]
        
        if len(uncovered_words) == 0:
            # Success! We reduced size.
            current_size -= 1
            if current_size < best_size:
                best_size = current_size
                best_in_code = in_code.copy()
        else:
            # Try to fix by adding a codeword
            if len(uncovered_words) <= 11:
                best_add = -1
                for w in range(total):
                    if in_code[w]:
                        continue
                    count = sum(1 for u in uncovered_words if u in balls[w])
                    if count == len(uncovered_words):
                        best_add = w
                        break
                if best_add != -1:
                    in_code[best_add] = True
                    cover_count[balls[best_add]] += 1
                    # Size remains same
                else:
                    # Revert
                    in_code[c_to_remove] = True
                    cover_count[balls[c_to_remove]] += 1
            else:
                # Cannot fix with single add, revert
                in_code[c_to_remove] = True
                cover_count[balls[c_to_remove]] += 1
    
    # Reconstruct code from best_in_code
    final_code = np.where(best_in_code)[0].tolist()
    
    # Convert indices to words
    result = []
    for idx in final_code:
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        result.append(w)
        
    return result

def _fallback(q, n, radius):
    total = q ** n
    if total <= 10000:
        res = []
        for i in range(total):
            w = [0] * n
            t = i
            for k in range(n - 1, -1, -1):
                w[k] = t % q
                t //= q
            res.append(w)
        return res
    return []

def construct(q, n, radius):
    # Only optimized for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1()
    # Fallback
    return _fallback(q, n, radius)
