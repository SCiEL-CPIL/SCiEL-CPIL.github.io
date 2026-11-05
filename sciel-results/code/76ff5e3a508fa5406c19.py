import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def calculate_threshold_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using bitmask representation.
    Uses precomputed transition lookup for speed.
    """
    # Precompute transitions for all 2^N masks
    size = 1 << N
    trans_a = [0] * size
    trans_b = [0] * size
    
    for mask in range(size):
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            res_a |= (1 << a_trans[idx])
            res_b |= (1 << b_trans[idx])
            m ^= lsb
        trans_a[mask] = res_a
        trans_b[mask] = res_b
    
    start = FULL_MASK
    
    # Check for single state: power of 2
    if (start & (start - 1)) == 0:
        return 0
    
    visited = [-1] * size
    visited[start] = 0
    q = deque([start])
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        # Check if curr is a single state (power of 2)
        if (curr & (curr - 1)) == 0:
            return curr_dist
        
        # Apply letter 'a'
        next_a = trans_a[curr]
        if visited[next_a] == -1:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        # Apply letter 'b'
        next_b = trans_b[curr]
        if visited[next_b] == -1:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # 2s safety margin
    
    best_score = -1
    best_a = None
    best_b = None
    
    def try_automaton(a, b):
        nonlocal best_score, best_a, best_b
        if time.time() > deadline:
            return False
        score = calculate_threshold_fast(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
            return True
        return False

    # Strategy 1: Structured families to find a good starting point
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # 1a. a = cycle, b = identity with one defect
    for k in range(N):
        for v in range(N):
            if v == k:
                continue
            b = list(range(N))
            b[k] = v
            try_automaton(a_cycle, b)
    
    if time.time() > deadline:
        if best_a is None:
            best_a = a_cycle
            best_b = [0] * N
        return [best_a, best_b]

    # 1b. a = cycle, b = identity with two defects (sampled)
    random.seed(42)
    for _ in range(1000):
        if time.time() > deadline:
            break
        k1 = random.randint(0, N - 1)
        k2 = random.randint(0, N - 1)
        if k1 == k2:
            continue
        v1 = random.randint(0, N - 1)
        v2 = random.randint(0, N - 1)
        b = list(range(N))
        b[k1] = v1
        b[k2] = v2
        if len(set(b)) == N:
            continue
        try_automaton(a_cycle, b)

    # 1c. Random search to find diverse starting points
    for _ in range(2000):
        if time.time() > deadline:
            break
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        try_automaton(a, b)
        
        b_rand = [random.randint(0, N-1) for _ in range(N)]
        try_automaton(a_cycle, b_rand)

    if best_a is None:
        best_a = a_cycle
        best_b = [0] * N
        best_score = calculate_threshold_fast(best_a, best_b)

    # Strategy 2: Simulated Annealing
    # This is more robust than Hill Climbing against local optima.
    current_a = best_a[:]
    current_b = best_b[:]
    current_score = best_score
    
    # Initial temperature
    T = 10.0
    cooling_rate = 0.995
    
    for step in range(10000):
        if time.time() > deadline:
            break
        
        # Random neighbor
        letter = random.choice(['a', 'b'])
        idx = random.randint(0, N-1)
        new_val = random.randint(0, N-1)
        
        if letter == 'a':
            old_val = current_a[idx]
            current_a[idx] = new_val
            neighbor_score = calculate_threshold_fast(current_a, current_b)
            current_a[idx] = old_val
        else:
            old_val = current_b[idx]
            current_b[idx] = new_val
            neighbor_score = calculate_threshold_fast(current_a, current_b)
            current_b[idx] = old_val
        
        if neighbor_score == -1:
            # Non-synchronizing, penalize heavily
            delta = current_score + 100  # Large penalty
            accept_prob = 0
        else:
            delta = neighbor_score - current_score
            if delta > 0:
                accept_prob = 1.0
            else:
                accept_prob = 1.0 / (1.0 + 0.01 * (1.0 - 0.95 ** (-delta)))
                # Or standard SA: exp(-delta/T)
                # accept_prob = math.exp(-delta / T)
        
        if random.random() < accept_prob:
            if letter == 'a':
                current_a[idx] = new_val
            else:
                current_b[idx] = new_val
            current_score = neighbor_score
            
            if current_score > best_score:
                best_score = current_score
                best_a = current_a[:]
                best_b = current_b[:]
        
        T *= cooling_rate
    
    # Strategy 3: Polishing with Steepest Ascent from the best found
    # Use the best candidate from SA and do strict hill climbing to polish
    current_a = best_a[:]
    current_b = best_b[:]
    current_score = best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_neighbor_score = current_score
        best_neighbor_a = None
        best_neighbor_b = None
        
        # Try changing 'a'
        for i in range(N):
            old_val = current_a[i]
            for new_val in range(N):
                if new_val == old_val:
                    continue
                current_a[i] = new_val
                if time.time() > deadline:
                    current_a[i] = old_val
                    break
                score = calculate_threshold_fast(current_a, current_b)
                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor_a = current_a[:]
                    best_neighbor_b = current_b[:]
                current_a[i] = old_val
            if time.time() > deadline:
                break
        
        # Try changing 'b'
        if time.time() < deadline:
            for i in range(N):
                old_val = current_b[i]
                for new_val in range(N):
                    if new_val == old_val:
                        continue
                    current_b[i] = new_val
                    if time.time() > deadline:
                        current_b[i] = old_val
                        break
                    score = calculate_threshold_fast(current_a, current_b)
                    if score > best_neighbor_score:
                        best_neighbor_score = score
                        best_neighbor_a = current_a[:]
                        best_neighbor_b = current_b[:]
                    current_b[i] = old_val
                if time.time() > deadline:
                    break

        if best_neighbor_a is not None:
            current_a = best_neighbor_a
            current_b = best_neighbor_b
            current_score = best_neighbor_score
            best_a = current_a[:]
            best_b = current_b[:]
            best_score = current_score
            improved = True

    # Fallback
    if best_a is None:
        best_a = a_cycle
        best_b = [0] * N
    
    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
