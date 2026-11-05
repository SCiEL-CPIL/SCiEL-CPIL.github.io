import time
import random
import math
from collections import deque

N = 12

def calculate_threshold_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using bitmask representation.
    """
    start = (1 << N) - 1  # All states set
    
    if (start & (start - 1)) == 0:
        return 0
    
    visited = {start: 0}
    q = deque([start])
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        if (curr & (curr - 1)) == 0:
            return curr_dist
        
        # Apply letter 'a'
        next_a = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            next_a |= (1 << a_trans[idx])
            temp ^= lsb
        if next_a not in visited:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        # Apply letter 'b'
        next_b = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            next_b |= (1 << b_trans[idx])
            temp ^= lsb
        if next_b not in visited:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = 0
    best_a = None
    best_b = None
    
    def try_automaton(a, b):
        nonlocal best_score, best_a, best_b
        score = calculate_threshold_fast(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
        return score

    random.seed(42)
    
    # Initialize with a random valid automaton
    a = [(i + 1) % N for i in range(N)]
    b = [random.randint(0, N-1) for _ in range(N)]
    if len(set(b)) == N:
        b[random.randint(0, N-1)] = b[random.randint(0, N-1)]
    
    current_score = try_automaton(a, b)
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.9995
    min_T = 0.5
    
    while time.time() < deadline - 10.0 and T > min_T:
        if random.random() < 0.5:
            idx = random.randint(0, N-1)
            old_val = a[idx]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            a[idx] = new_val
            new_score = try_automaton(a, b)
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_score = new_score
            else:
                a[idx] = old_val
        else:
            idx = random.randint(0, N-1)
            old_val = b[idx]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            b[idx] = new_val
            new_score = try_automaton(a, b)
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_score = new_score
            else:
                b[idx] = old_val
        
        T *= cooling
    
    # Polish with Hill Climbing
    if best_a is not None and time.time() < deadline - 2.0:
        a = best_a[:]
        b = best_b[:]
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try flipping each bit in a
            for i in range(N):
                for new_val in range(N):
                    if new_val == a[i]: continue
                    a_old = a[i]
                    a[i] = new_val
                    new_score = try_automaton(a, b)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        a[i] = a_old
                    if time.time() > deadline - 2.0:
                        break
                if time.time() > deadline - 2.0:
                    break
            
            # Try flipping each bit in b
            if time.time() < deadline - 2.0:
                for i in range(N):
                    for new_val in range(N):
                        if new_val == b[i]: continue
                        b_old = b[i]
                        b[i] = new_val
                        new_score = try_automaton(a, b)
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                        else:
                            b[i] = b_old
                        if time.time() > deadline - 2.0:
                            break
                    if time.time() > deadline - 2.0:
                        break

    if best_a is None:
        best_a = a
        best_b = b

    return [best_a, best_b]
