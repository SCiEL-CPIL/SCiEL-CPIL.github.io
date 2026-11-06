import time
import random
import sys
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 85.0  # safety margin before 90s kill

def _build_subset_trans(trans):
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset

def compute_threshold(trans_a, trans_b):
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)

    size = 1 << N
    dist = [-1] * size
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])

    while queue:
        cur = queue.popleft()
        if cur & (cur - 1) == 0:
            return dist[cur]

        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)

        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)

    return 0

def _valid_ping_pong():
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
    return a, b

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    random.seed(42)
    
    # Initialize with Ping-Pong
    cur_a, cur_b = _valid_ping_pong()
    cur_score = compute_threshold(cur_a, cur_b)
    
    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    
    while time.time() < deadline - 5.0:
        # Create a neighbor
        new_a = cur_a[:]
        new_b = cur_b[:]
        
        # Random mutation
        if random.random() < 0.5:
            state = random.randint(0, N - 1)
            new_a[state] = random.randint(0, N - 1)
        else:
            state = random.randint(0, N - 1)
            new_b[state] = random.randint(0, N - 1)
            
        new_score = compute_threshold(new_a, new_b)
        
        # Acceptance probability
        if new_score > cur_score:
            # Always accept improvement
            cur_a, cur_b = new_a, new_b
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_a = cur_a[:]
                best_b = cur_b[:]
        else:
            # Accept with probability exp((new_score - cur_score) / T)
            # If new_score is 0 (not synchronizing), delta is large negative, prob is ~0
            delta = new_score - cur_score
            if delta < 0:
                prob = random.expovariate(-delta / T) if T > 0 else 0
                if random.random() < prob:
                    cur_a, cur_b = new_a, new_b
                    cur_score = new_score
            # If delta is 0, accept? Usually yes, to explore.
            elif delta == 0:
                if random.random() < 0.5:
                    cur_a, cur_b = new_a, new_b
                    cur_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    if best_score == 0:
        # Fallback
        a, b = _valid_ping_pong()
        return [a, b]
        
    return [best_a, best_b]
