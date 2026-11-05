import time
import random
from collections import deque

N = 12
FULL_STATE = (1 << N) - 1

def calculate_threshold(a_trans, b_trans):
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    for mask in range(1, 1 << N):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        
        next_a[mask] = next_a[prev_mask] | (1 << a_trans[idx])
        next_b[mask] = next_b[prev_mask] | (1 << b_trans[idx])
        
    visited = [False] * (1 << N)
    visited[FULL_STATE] = True
    queue = deque([(FULL_STATE, 0)])
    
    while queue:
        state, dist = queue.popleft()
        if state.bit_count() == 1:
            return dist
            
        next_a_state = next_a[state]
        if not visited[next_a_state]:
            visited[next_a_state] = True
            queue.append((next_a_state, dist + 1))
            
        next_b_state = next_b[state]
        if not visited[next_b_state]:
            visited[next_b_state] = True
            queue.append((next_b_state, dist + 1))
            
    return -1

def construct():
    start_time = time.time()
    timeout = 85.0
    
    best_a = [(i + 1) % N for i in range(N)]
    best_b = None
    best_score = -1
    
    curr_b = [random.randint(0, N-1) for _ in range(N)]
    curr_score = calculate_threshold(best_a, curr_b)
    if curr_score > best_score:
        best_score = curr_score
        best_b = curr_b[:]
        
    no_improve = 0
    max_no_improve = 100
    
    while time.time() - start_time < timeout:
        if no_improve > max_no_improve:
            # Kick: change 2-3 random positions
            num_changes = random.randint(2, 3)
            for _ in range(num_changes):
                i = random.randint(0, N-1)
                curr_b[i] = random.randint(0, N-1)
            
            new_score = calculate_threshold(best_a, curr_b)
            if new_score > curr_score:
                curr_score = new_score
                no_improve = 0
                if new_score > best_score:
                    best_score = new_score
                    best_b = curr_b[:]
            else:
                # If kick didn't help, try a full restart
                curr_b = [random.randint(0, N-1) for _ in range(N)]
                curr_score = calculate_threshold(best_a, curr_b)
                if curr_score > best_score:
                    best_score = curr_score
                    best_b = curr_b[:]
            no_improve = 0
            continue
            
        i = random.randint(0, N-1)
        old_val = curr_b[i]
        new_val = random.randint(0, N-1)
        if new_val == old_val:
            continue
            
        curr_b[i] = new_val
        new_score = calculate_threshold(best_a, curr_b)
        
        if new_score > curr_score:
            curr_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_b = curr_b[:]
        else:
            curr_b[i] = old_val
            no_improve += 1
            
    if best_b is None:
        best_b = [0] * N
        
    return [best_a, best_b]
