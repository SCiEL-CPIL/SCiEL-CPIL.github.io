import time
import random
from collections import deque

N = 12

def calculate_threshold(a_trans, b_trans):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses BFS on the subset automaton with bitmasks for efficiency.
    """
    all_states = (1 << N) - 1
    
    # Precompute transition tables for bitmask application
    # trans_a[mask] = result of applying 'a' to subset 'mask'
    # trans_b[mask] = result of applying 'b' to subset 'mask'
    # There are 2^12 = 4096 possible masks.
    
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        if mask == 0:
            continue
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            res_a |= (1 << a_trans[i])
            res_b |= (1 << b_trans[i])
            m ^= lsb
        trans_a[mask] = res_a
        trans_b[mask] = res_b

    start = all_states
    
    # Check if already singleton
    if (start & (start - 1)) == 0:
        return 0

    # BFS
    visited = [-1] * (1 << N)
    visited[start] = 0
    q = [start]
    head = 0
    
    while head < len(q):
        curr = q[head]
        head += 1
        curr_dist = visited[curr]
        
        # Check if singleton
        if (curr & (curr - 1)) == 0:
            return curr_dist
        
        next_a = trans_a[curr]
        if visited[next_a] == -1:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        next_b = trans_b[curr]
        if visited[next_b] == -1:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = -1
    best_a = None
    best_b = None
    
    # Helper to update best
    def update(a, b):
        nonlocal best_score, best_a, best_b
        score = calculate_threshold(a, b)
        if score > best_score:
            best_score = score
            best_a = list(a)
            best_b = list(b)
        return score

    # Strategy: The theoretical maximum for N=12 is (12-1)^2 = 121.
    # We search for this specific value in structured families known to produce tight examples.
    
    # 1. 'a' is a full cycle (shift by 1)
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Sub-family 1a: b is identity with 1 defect
    # This family is known to contain tight examples for many N.
    for k in range(N):
        for j in range(N):
            if j == k:
                continue
            b = [i for i in range(N)]
            b[k] = j
            update(a_cycle, b)
            if best_score == 121:
                return [best_a, best_b]
            if time.time() > deadline:
                return [best_a, best_b]
                
    # Sub-family 1b: b is identity with 2 defects
    for k1 in range(N):
        for k2 in range(k1+1, N):
            for j1 in range(N):
                for j2 in range(N):
                    if j1 == k1 and j2 == k2:
                        continue
                    b = [i for i in range(N)]
                    b[k1] = j1
                    b[k2] = j2
                    update(a_cycle, b)
                    if best_score == 121:
                        return [best_a, best_b]
                    if time.time() > deadline:
                        return [best_a, best_b]

    # 2. 'a' is a shift by d
    for d in range(1, 7):
        a_shift = [(i + d) % N for i in range(N)]
        # Try b as identity with 1 defect
        for k in range(N):
            for j in range(N):
                if j == k:
                    continue
                b = [i for i in range(N)]
                b[k] = j
                update(a_shift, b)
                if best_score == 121:
                    return [best_a, best_b]
                if time.time() > deadline:
                    return [best_a, best_b]

    # 3. If we haven't found 121, do a quick random search to ensure we return the best found
    random.seed(42)
    for _ in range(1000):
        if time.time() > deadline:
            break
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        # Ensure synchronization is likely by creating a collision
        b[0] = b[1]
        update(a, b)
        if best_score == 121:
            return [best_a, best_b]

    if best_a is None:
        best_a = a_cycle
        best_b = [0] * N

    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
