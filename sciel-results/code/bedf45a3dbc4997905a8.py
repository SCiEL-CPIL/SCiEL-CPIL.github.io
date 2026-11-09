import time
import random
from collections import deque

N = 12

def compute_threshold(trans_a, trans_b):
    """
    Computes the exact reset threshold using BFS on the subset automaton.
    Returns the length of the shortest reset word.
    """
    full_set = (1 << N) - 1
    # BFS from full_set
    # State: (subset_mask, length)
    visited = {full_set: 0}
    queue = deque([(full_set, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        if mask.bit_count() == 1:
            return dist
        
        # Apply trans_a
        new_mask_a = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_a |= (1 << trans_a[i])
        
        if new_mask_a not in visited:
            visited[new_mask_a] = dist + 1
            queue.append((new_mask_a, dist + 1))
            
        # Apply trans_b
        new_mask_b = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_b |= (1 << trans_b[i])
        
        if new_mask_b not in visited:
            visited[new_mask_b] = dist + 1
            queue.append((new_mask_b, dist + 1))
            
    return -1 # Not synchronizing

def construct():
    """
    Uses local search to maximize the reset threshold.
    """
    start_time = time.time()
    timeout = 50.0 # Safety margin
    
    # Initial guess: Simple Černý automaton
    # a: i -> i+1 mod N
    # b: i -> i-1 if i>0 else 0
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [i - 1 if i > 0 else 0 for i in range(N)]
    
    best_score = compute_threshold(best_a, best_b)
    
    # Local Search
    current_a = best_a[:]
    current_b = best_b[:]
    current_score = best_score
    
    # To speed up BFS, we can precompute transitions for masks?
    # Or just rely on the fact that BFS is fast for N=12.
    # 4096 states. Each step iterates N=12. ~50k ops per threshold calc.
    # Python might be slow. 50k ops * 1000 iterations = 50M ops. Might take 5-10s.
    # We can do ~100-500 iterations in 50s.
    
    # Optimization: Precompute letter actions for all masks?
    # There are 4096 masks. 2 letters.
    # Precomputation takes 4096 * 12 * 2 = ~100k ops. Very fast.
    # Then BFS is just lookup.
    
    def precompute_actions(a, b):
        actions = [[0]*2 for _ in range(1 << N)]
        for mask in range(1 << N):
            # Action a
            m_a = 0
            for i in range(N):
                if mask & (1 << i):
                    m_a |= (1 << a[i])
            actions[mask][0] = m_a
            
            # Action b
            m_b = 0
            for i in range(N):
                if mask & (1 << i):
                    m_b |= (1 << b[i])
            actions[mask][1] = m_b
        return actions

    def fast_threshold(actions):
        full_set = (1 << N) - 1
        visited = {full_set: 0}
        queue = deque([(full_set, 0)])
        
        while queue:
            mask, dist = queue.popleft()
            if mask.bit_count() == 1:
                return dist
            
            for letter in range(2):
                next_mask = actions[mask][letter]
                if next_mask not in visited:
                    visited[next_mask] = dist + 1
                    queue.append((next_mask, dist + 1))
        return -1

    actions = precompute_actions(current_a, current_b)
    current_score = fast_threshold(actions)
    best_score = current_score
    best_a = current_a[:]
    best_b = current_b[:]
    
    random.seed(42)
    
    while time.time() - start_time < timeout:
        # Mutate
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        old_dest = current_a[state] if letter == 0 else current_b[state]
        new_dest = random.randint(0, N - 1)
        
        if letter == 0:
            current_a[state] = new_dest
        else:
            current_b[state] = new_dest
            
        # Recalculate
        actions = precompute_actions(current_a, current_b)
        new_score = fast_threshold(actions)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            # Revert
            if letter == 0:
                current_a[state] = old_dest
            else:
                current_b[state] = old_dest
                
    return [best_a, best_b]

# EVOLVE-BLOCK-END
