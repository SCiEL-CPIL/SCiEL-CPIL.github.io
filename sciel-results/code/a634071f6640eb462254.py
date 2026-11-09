import time
import random
import sys
from collections import deque

N = 12
DEADLINE = 85.0

def compute_threshold(a_map, b_map):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Returns the length or None if not synchronizing.
    Uses precomputed transitions for speed.
    """
    size = 1 << N
    trans_a = [0] * size
    trans_b = [0] * size
    
    for s in range(size):
        na = 0
        nb = 0
        for i in range(N):
            if s & (1 << i):
                na |= (1 << a_map[i])
                nb |= (1 << b_map[i])
        trans_a[s] = na
        trans_b[s] = nb
        
    start = size - 1
    visited = [False] * size
    visited[start] = True
    queue = deque([(start, 0)])
    
    while queue:
        state, dist = queue.popleft()
        
        if state & (state - 1) == 0:
            return dist
            
        next_a = trans_a[state]
        if not visited[next_a]:
            visited[next_a] = True
            queue.append((next_a, dist + 1))
            
        next_b = trans_b[state]
        if not visited[next_b]:
            visited[next_b] = True
            queue.append((next_b, dist + 1))
            
    return None

def construct_pawlik_variant(k, j, direction=1):
    """
    Construct a Pawlik-like automaton variant.
    k: the state where b has a "fold"
    j: where b maps the fold state
    direction: 1 for forward cycle, -1 for backward
    """
    a = [(i + direction) % N for i in range(N)]
    b = [i - direction if (i - direction) % N != 0 else 0 for i in range(N)]
    # Standard b: i -> i-1 if i>0, else 0 (for direction=1)
    # For direction=-1: i -> i+1 if i<N-1, else N-1
    if direction == 1:
        b = [i - 1 if i > 0 else 0 for i in range(N)]
    else:
        b = [i + 1 if i < N - 1 else N - 1 for i in range(N)]
    
    b[k] = j
    return a, b

def construct():
    """
    Returns the transition tables for a synchronizing automaton with N=12 states,
    maximizing the shortest reset word length.
    """
    start_time = time.time()
    
    best_a = [0] * N
    best_b = [0] * N
    best_score = 0
    
    def evaluate_candidate(a, b):
        nonlocal best_a, best_b, best_score
        score = compute_threshold(a, b)
        if score is not None and score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
        return score
    
    # Strategy 1: Try all Pawlik variants (k, j) for both cycle directions
    # The Pawlik automaton has a specific structure that achieves high thresholds
    for direction in [1, -1]:
        for k in range(N):
            for j in range(N):
                if time.time() > DEADLINE - 5.0:
                    break
                a, b = construct_pawlik_variant(k, j, direction)
                evaluate_candidate(a, b)
            if time.time() > DEADLINE - 5.0:
                break
    
    # Strategy 2: Try the "dual" construction - invert the reset word structure
    # The anti-Pawlik topology uses a different fold pattern
    # Instead of b(i) = i-1, try b(i) = i+1 with a fold
    for direction in [1, -1]:
        for k in range(N):
            for j in range(N):
                if time.time() > DEADLINE - 3.0:
                    break
                a = [(i + direction) % N for i in range(N)]
                # Anti-Pawlik: b goes in opposite direction with a fold
                if direction == 1:
                    b = [i + 1 if i < N - 1 else N - 1 for i in range(N)]
                else:
                    b = [i - 1 if i > 0 else 0 for i in range(N)]
                b[k] = j
                evaluate_candidate(a, b)
            if time.time() > DEADLINE - 3.0:
                break
    
    # Strategy 3: Targeted local search around the best found
    if best_score > 0 and time.time() < DEADLINE - 2.0:
        current_a = best_a[:]
        current_b = best_b[:]
        
        # Try single-transition mutations
        for _ in range(200):
            if time.time() > DEADLINE - 1.0:
                break
            
            a = current_a[:]
            b = current_b[:]
            
            # Mutate one b transition
            idx = random.randint(0, N-1)
            b[idx] = random.randint(0, N-1)
            
            # Occasionally also mutate a
            if random.random() < 0.3:
                idx2 = random.randint(0, N-1)
                a[idx2] = random.randint(0, N-1)
            
            score = evaluate_candidate(a, b)
            if score is not None and score > best_score:
                best_score = score
                current_a = a[:]
                current_b = b[:]
                
                if best_score >= 121:
                    break
    
    # Fallback: ensure we have a valid synchronizing automaton
    if best_score == 0:
        a = [(i + 1) % N for i in range(N)]
        b = [i - 1 if i > 0 else 0 for i in range(N)]
        best_a = a
        best_b = b
        evaluate_candidate(a, b)
    
    return [best_a, best_b]

if __name__ == "__main__":
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"Time: {time.time()}")
