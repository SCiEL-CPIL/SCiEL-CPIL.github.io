"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _compute_threshold(A, B):
    """
    Computes the exact reset threshold using BFS on the subset automaton.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    num_subsets = 1 << N
    trans_a = [0] * num_subsets
    trans_b = [0] * num_subsets
    
    for s in range(num_subsets):
        next_a = 0
        next_b = 0
        for i in range(N):
            if s & (1 << i):
                next_a |= (1 << A[i])
                next_b |= (1 << B[i])
        trans_a[s] = next_a
        trans_b[s] = next_b
    
    full_set = (1 << N) - 1
    
    dist = [-1] * num_subsets
    dist[full_set] = 0
    q = deque([full_set])
    
    min_dist = float('inf')
    
    while q:
        s = q.popleft()
        d = dist[s]
        
        if s and (s & (s - 1)) == 0:
            if d < min_dist:
                min_dist = d
            
        next_a = trans_a[s]
        next_b = trans_b[s]
        
        if dist[next_a] == -1:
            dist[next_a] = d + 1
            q.append(next_a)
            
        if dist[next_b] == -1:
            dist[next_b] = d + 1
            q.append(next_b)
            
    if min_dist == float('inf'):
        return -1
    return min_dist

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Initial Candidate: Pinwheel-like
    A = [(i + 1) % N for i in range(N)]
    B = [i if i < N - 1 else 0 for i in range(N)]
    
    best_A = A[:]
    best_B = B[:]
    best_thresh = _compute_threshold(best_A, best_B)
    
    if best_thresh == -1:
        # Fallback
        A = [(i + 1) % N for i in range(N)]
        B = [0] * N
        best_A = A[:]
        best_B = B[:]
        best_thresh = _compute_threshold(best_A, best_B)
        
    if best_thresh == -1:
        return A, B

    current_A = best_A[:]
    current_B = best_B[:]
    
    while time.time() < deadline:
        letter = random.choice([0, 1])
        state = random.randint(0, N - 1)
        
        if letter == 0:
            orig_dest = current_A[state]
            new_dest = random.randint(0, N - 1)
            if new_dest == orig_dest:
                continue
            current_A[state] = new_dest
            thresh = _compute_threshold(current_A, current_B)
            current_A[state] = orig_dest
        else:
            orig_dest = current_B[state]
            new_dest = random.randint(0, N - 1)
            if new_dest == orig_dest:
                continue
            current_B[state] = new_dest
            thresh = _compute_threshold(current_A, current_B)
            current_B[state] = orig_dest
            
        if thresh != -1 and thresh > best_thresh:
            best_thresh = thresh
            if letter == 0:
                best_A[state] = new_dest
            else:
                best_B[state] = new_dest
            
            if letter == 0:
                current_A = best_A[:]
            else:
                current_B = best_B[:]
                
    return best_A, best_B
# EVOLVE-BLOCK-END
