import time
import random

N = 12

def get_reset_length(a, b):
    # Returns shortest reset word length, or -1 if not synchronizing
    start_mask = (1 << N) - 1
    if start_mask == 0: return 0
    
    # BFS
    from collections import deque
    queue = deque([(start_mask, 0)])
    visited = {start_mask}
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return dist
            
        # Next states for 'a' and 'b'
        for letter in (a, b):
            next_mask = 0
            for i in range(N):
                if (mask >> i) & 1:
                    next_mask |= (1 << letter[i])
            if next_mask not in visited:
                visited.add(next_mask)
                queue.append((next_mask, dist + 1))
                
    return -1

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    best_a = None
    best_b = None
    best_len = -1
    
    # Multi-start local search
    while time.time() < deadline:
        # Initialize random automaton
        # To increase chance of synchronization, make 'b' somewhat contracting
        # e.g., map many states to 0
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        
        # Ensure it's not trivially non-synchronizing? 
        # Random ones often don't synchronize.
        # Let's try a structured initialization:
        # a is a random permutation (cycle)
        # b maps most states to 0, some to others
        a = list(range(N))
        random.shuffle(a)
        # Ensure a is a single cycle? Not strictly necessary for sync, but helps.
        # Let's just use random permutations for a, and biased random for b.
        
        a = list(range(N))
        random.shuffle(a)
        b = [0] * N
        for i in range(1, N):
            b[i] = random.randint(0, i) # This ensures b is contracting? No.
            # Better: b[i] = random.randint(0, N-1)
        
        # Let's use a specific heuristic initialization:
        # a: cycle 0->1->...->N-1->0
        # b: 0->0, i->i-1 for i>0 (This we know loops for N=3, but might work for larger N with modifications?)
        # Actually, let's just do pure random restarts with local search.
        
        cur_a = [random.randint(0, N-1) for _ in range(N)]
        cur_b = [random.randint(0, N-1) for _ in range(N)]
        
        # Try to make it synchronize by biasing b
        # If random doesn't work, we can force synchronization by setting all b[i]=0.
        # But that gives length 1.
        
        # Let's use a "near-cycle" structure for a.
        a = [(i + 1) % N for i in range(N)]
        # b is random
        b = [random.randint(0, N-1) for _ in range(N)]
        
        cur_len = get_reset_length(a, b)
        
        if cur_len == -1:
            continue
            
        if cur_len > best_len:
            best_len = cur_len
            best_a = a[:]
            best_b = b[:]
            
        # Local Search
        steps = 1000
        for _ in range(steps):
            if time.time() > deadline:
                break
                
            # Pick random change
            letter_idx = random.randint(0, 1)
            state_idx = random.randint(0, N-1)
            
            if letter_idx == 0:
                old_val = a[state_idx]
                a[state_idx] = random.randint(0, N-1)
                if a[state_idx] == old_val:
                    continue
                new_len = get_reset_length(a, b)
                if new_len == -1 or new_len <= cur_len:
                    a[state_idx] = old_val
                else:
                    cur_len = new_len
                    if cur_len > best_len:
                        best_len = cur_len
                        best_a = a[:]
                        best_b = b[:]
            else:
                old_val = b[state_idx]
                b[state_idx] = random.randint(0, N-1)
                if b[state_idx] == old_val:
                    continue
                new_len = get_reset_length(a, b)
                if new_len == -1 or new_len <= cur_len:
                    b[state_idx] = old_val
                else:
                    cur_len = new_len
                    if cur_len > best_len:
                        best_len = cur_len
                        best_a = a[:]
                        best_b = b[:]
                        
    if best_a is None:
        # Fallback to simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return [best_a, best_b]
