import random
import time

N = 12

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s margin
    
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] * N
    best_score = 1  # Seed score
    
    def compute_reset_length(a, b):
        # BFS on subset automaton
        # State is a tuple of sorted unique states
        # Start with (0, 1, ..., N-1)
        # Target is any state with length 1
        
        from collections import deque
        
        start = tuple(range(N))
        if len(start) == 1:
            return 0
            
        visited = {start}
        queue = deque([(start, 0)])
        
        while queue:
            subset, dist = queue.popleft()
            
            # Apply a
            next_subset_a = tuple(sorted(set(a[i] for i in subset)))
            if len(next_subset_a) == 1:
                return dist + 1
            if next_subset_a not in visited:
                visited.add(next_subset_a)
                queue.append((next_subset_a, dist + 1))
                
            # Apply b
            next_subset_b = tuple(sorted(set(b[i] for i in subset)))
            if len(next_subset_b) == 1:
                return dist + 1
            if next_subset_b not in visited:
                visited.add(next_subset_b)
                queue.append((next_subset_b, dist + 1))
                
        return -1  # Not synchronizing

    # Initial randomization
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Generate random b
        b = [random.randint(0, N-1) for _ in range(N)]
        
        # Ensure synchronizability: often just having a sink-like behavior helps
        # But random b with cycle a is often synchronizing for N=12
        
        score = compute_reset_length(best_a, b)
        
        if score > best_score:
            best_score = score
            best_b = b[:]
            
    # Local Search: Try to improve best_b
    # We can try swapping b[i] with b[j] or changing b[i] to a random value
    
    current_b = best_b[:]
    current_score = best_score
    
    while time.time() < deadline:
        # Pick a random index to change
        idx = random.randint(0, N-1)
        old_val = current_b[idx]
        
        # Try all possible values for b[idx]
        for new_val in range(N):
            if new_val == old_val:
                continue
                
            current_b[idx] = new_val
            score = compute_reset_length(best_a, current_b)
            
            if score > current_score:
                current_score = score
                best_b = current_b[:]
                best_score = score
                # Keep the change
            else:
                # Revert if worse (or accept with prob in SA, but simple hill climb is safer)
                current_b[idx] = old_val
                
        # Also try swapping two b values
        i1 = random.randint(0, N-1)
        i2 = random.randint(0, N-1)
        if i1 != i2:
            current_b[i1], current_b[i2] = current_b[i2], current_b[i1]
            score = compute_reset_length(best_a, current_b)
            if score > current_score:
                current_score = score
                best_b = current_b[:]
                best_score = score
            else:
                current_b[i1], current_b[i2] = current_b[i2], current_b[i1]

    return [best_a, best_b]
