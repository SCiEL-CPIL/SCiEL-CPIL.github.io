import random
import time
from collections import deque

N = 12

def _compute_reset_length(a, b):
    size = 1 << N
    bit_a = [1 << x for x in a]
    bit_b = [1 << x for x in b]
    
    trans_a = [0] * size
    trans_b = [0] * size
    
    # DP to fill transition tables in O(2^N)
    for mask in range(1, size):
        lb = mask & -mask
        # Find index of lb
        # Since N=12, we can use a small lookup or bit_length
        i = (lb.bit_length() - 1)
        prev_mask = mask ^ lb
        trans_a[mask] = trans_a[prev_mask] | bit_a[i]
        trans_b[mask] = trans_b[prev_mask] | bit_b[i]
        
    start = size - 1
    if start & (start - 1) == 0:
        return 0
        
    visited = {start}
    q = deque([(start, 0)])
    
    while q:
        mask, dist = q.popleft()
        
        if mask & (mask - 1) == 0:
            return dist
            
        next_a = trans_a[mask]
        if next_a not in visited:
            visited.add(next_a)
            q.append((next_a, dist + 1))
            
        next_b = trans_b[mask]
        if next_b not in visited:
            visited.add(next_b)
            q.append((next_b, dist + 1))
            
    return -1

def construct():
    # Initial Pinwheel Automaton
    a = [(i + 1) % N for i in range(N)]
    b = [0 if i == 0 else i - 1 for i in range(N)]
    
    best_a = a[:]
    best_b = b[:]
    best_len = _compute_reset_length(best_a, best_b)
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    stuck_count = 0
    max_stuck = 100
    
    while time.time() < deadline:
        cur_a = best_a[:]
        cur_b = best_b[:]
        
        if stuck_count >= max_stuck:
            for _ in range(5):
                letter = random.randint(0, 1)
                state = random.randint(0, N - 1)
                new_dest = random.randint(0, N - 1)
                if letter == 0:
                    cur_a[state] = new_dest
                else:
                    cur_b[state] = new_dest
            stuck_count = 0
            max_stuck = random.randint(100, 300)
        else:
            letter = random.randint(0, 1)
            state = random.randint(0, N - 1)
            new_dest = random.randint(0, N - 1)
            if letter == 0:
                cur_a[state] = new_dest
            else:
                cur_b[state] = new_dest
        
        new_len = _compute_reset_length(cur_a, cur_b)
        
        if new_len > best_len:
            best_len = new_len
            best_a = cur_a
            best_b = cur_b
            stuck_count = 0
        else:
            stuck_count += 1
            
    return [best_a, best_b]
