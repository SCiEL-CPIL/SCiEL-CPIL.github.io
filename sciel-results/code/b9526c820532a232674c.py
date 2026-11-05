import time
import random
from collections import deque

N = 12

def calculate_threshold(a_trans, b_trans, max_depth=200):
    """
    Computes the exact shortest reset word length using bitmask representation.
    """
    start = (1 << N) - 1
    
    cache_a = {}
    cache_b = {}
    
    def apply_trans(mask, trans, cache):
        if mask in cache:
            return cache[mask]
        result = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            result |= (1 << trans[idx])
            m ^= lsb
        cache[mask] = result
        return result
    
    visited = {start: 0}
    q = deque([start])
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        if curr_dist > max_depth:
            return -1
        
        if (curr & (curr - 1)) == 0:
            return curr_dist
        
        next_a = apply_trans(curr, a_trans, cache_a)
        if next_a not in visited:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        next_b = apply_trans(curr, b_trans, cache_b)
        if next_b not in visited:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = -1
    best_a = None
    best_b = None
    
    def try_automaton(a, b):
        nonlocal best_score, best_a, best_b
        if time.time() > deadline:
            return
        score = calculate_threshold(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # Family 1: a = cycle (shift 1), b = identity with 1 or 2 defects
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Single defect
    for k in range(N):
        for j in range(N):
            if j == k:
                continue
            b = list(range(N))
            b[k] = j
            try_automaton(a_cycle, b)
            if time.time() > deadline:
                if best_a:
                    return [best_a, best_b]
    
    # Double defect (limit to keep within time)
    for k1 in range(N):
        for k2 in range(k1 + 1, N):
            for j1 in range(N):
                for j2 in range(N):
                    if j1 == k1 and j2 == k2:
                        continue
                    b = list(range(N))
                    b[k1] = j1
                    b[k2] = j2
                    try_automaton(a_cycle, b)
                    if time.time() > deadline:
                        if best_a:
                            return [best_a, best_b]
    
    # Family 2: a = cycle with different shifts, b = identity with single defect
    for d in range(2, N):  # d=1 already done
        a_shift = [(i + d) % N for i in range(N)]
        for k in range(N):
            for j in range(N):
                if j == k:
                    continue
                b = list(range(N))
                b[k] = j
                try_automaton(a_shift, b)
                if time.time() > deadline:
                    if best_a:
                        return [best_a, best_b]
    
    # Family 3: Classic Černý construction
    b_cherny = [0] * N
    for i in range(1, N):
        b_cherny[i] = i - 1
    try_automaton(a_cycle, b_cherny)
    
    # Family 4: Random 'b' with forced collision, 'a' = cycle
    random.seed(42)
    for _ in range(1500):
        if time.time() > deadline:
            break
        b = [random.randint(0, N - 1) for _ in range(N)]
        b[1] = b[0]
        try_automaton(a_cycle, b)
    
    # Family 5: Random single-cycle 'a', random 'b' with collision
    for _ in range(1000):
        if time.time() > deadline:
            break
        perm = list(range(N))
        random.shuffle(perm)
        a_trans = [0] * N
        for i in range(N):
            a_trans[perm[i]] = perm[(i + 1) % N]
        b = [random.randint(0, N - 1) for _ in range(N)]
        b[1] = b[0]
        try_automaton(a_trans, b)
    
    # Local improvement from best found
    if best_a and best_b and best_score > 0 and time.time() < deadline:
        a_curr = best_a[:]
        b_curr = best_b[:]
        score_curr = best_score
        
        for _ in range(30):  # Max 30 rounds of local search
            if time.time() > deadline:
                break
            improved = False
            
            # Try each position in 'a'
            for i in range(N):
                orig_a = a_curr[i]
                for new_val in range(N):
                    if new_val == orig_a:
                        continue
                    a_curr[i] = new_val
                    s = calculate_threshold(a_curr, b_curr)
                    if s > score_curr:
                        score_curr = s
                        best_a = a_curr[:]
                        best_b = b_curr[:]
                        improved = True
                        break
                a_curr[i] = orig_a
            
            # Try each position in 'b'
            for i in range(N):
                orig_b = b_curr[i]
                for new_val in range(N):
                    if new_val == orig_b:
                        continue
                    b_curr[i] = new_val
                    s = calculate_threshold(a_curr, b_curr)
                    if s > score_curr:
                        score_curr = s
                        best_a = a_curr[:]
                        best_b = b_curr[:]
                        improved = True
                        break
                b_curr[i] = orig_b
            
            if not improved:
                break
    
    if best_a is None or best_score < 0:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        for i in range(1, N):
            b[i] = i - 1
        return [a, b]
    
    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
