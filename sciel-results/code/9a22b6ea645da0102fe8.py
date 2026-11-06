import random
import time
from collections import deque

N = 12

def bfs_reset_length(trans):
    """
    Compute the length of the shortest reset word for a binary automaton.
    trans is a tuple (a, b) where a and b are lists of destinations.
    Returns the length, or -1 if not synchronizing.
    """
    a, b = trans
    full_set = frozenset(range(N))
    
    # If full set is not reachable or doesn't collapse, we might not sync, 
    # but BFS from full set will tell us if we reach a singleton.
    
    visited = {full_set: 0}
    queue = deque([full_set])
    
    while queue:
        subset = queue.popleft()
        dist = visited[subset]
        
        # Check if this is a singleton
        if len(subset) == 1:
            return dist
            
        # Generate next subsets
        next_a = frozenset(a[s] for s in subset)
        next_b = frozenset(b[s] for s in subset)
        
        if next_a not in visited:
            visited[next_a] = dist + 1
            queue.append(next_a)
            
        if next_b not in visited:
            visited[next_b] = dist + 1
            queue.append(next_b)
            
    return -1

def construct():
    # Start with the standard Černý automaton
    # a: i -> (i+1) % N
    # b: i -> (i-1) % N if i>0 else 0
    
    a = [(i + 1) % N for i in range(N)]
    b = [(i - 1) % N if i > 0 else 0 for i in range(N)]
    
    current_a = a[:]
    current_b = b[:]
    current_len = bfs_reset_length((current_a, current_b))
    
    best_a = current_a[:]
    best_b = current_b[:]
    best_len = current_len
    
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    # Local Search
    while time.time() - start_time < deadline:
        # Pick a random state and input
        state = random.randint(0, N - 1)
        input_type = random.choice([0, 1])  # 0 for a, 1 for b
        new_dest = random.randint(0, N - 1)
        
        # Save old state
        if input_type == 0:
            old_dest = current_a[state]
            current_a[state] = new_dest
        else:
            old_dest = current_b[state]
            current_b[state] = new_dest
            
        new_len = bfs_reset_length((current_a, current_b))
        
        # Accept if better, or if equal (to allow wandering)
        # Using a simple hill climbing with restarts or random walks
        if new_len > best_len:
            best_len = new_len
            best_a = current_a[:]
            best_b = current_b[:]
        elif new_len == best_len:
            # Keep the change to explore the plateau
            pass
        else:
            # Revert if worse
            if input_type == 0:
                current_a[state] = old_dest
            else:
                current_b[state] = old_dest
                
        # Optional: If we stuck at a local max for a while, we could restart,
        # but for N=12 the landscape is likely well-connected for the optimal value.
        
    return [best_a, best_b]
