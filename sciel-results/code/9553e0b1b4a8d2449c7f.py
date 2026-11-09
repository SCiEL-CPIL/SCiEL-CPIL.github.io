import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    def compute_threshold(trans_a, trans_b):
        start_mask = (1 << N) - 1
        
        trans_a_mask = [0] * (1 << N)
        trans_b_mask = [0] * (1 << N)
        
        for mask in range(1 << N):
            ma = 0
            mb = 0
            for i in range(N):
                if mask & (1 << i):
                    ma |= (1 << trans_a[i])
                    mb |= (1 << trans_b[i])
            trans_a_mask[mask] = ma
            trans_b_mask[mask] = mb
            
        singletons = set(1 << i for i in range(N))
        
        if start_mask in singletons:
            return 0
            
        dist = {start_mask: 0}
        queue = deque([start_mask])
        
        while queue:
            cur = queue.popleft()
            d = dist[cur]
            
            next_a = trans_a_mask[cur]
            if next_a not in dist:
                dist[next_a] = d + 1
                if next_a in singletons:
                    return d + 1
                queue.append(next_a)
                
            next_b = trans_b_mask[cur]
            if next_b not in dist:
                dist[next_b] = d + 1
                if next_b in singletons:
                    return d + 1
                queue.append(next_b)
                
        return -1

    # Initial candidate: Černý-like
    trans_a = [(i + 1) % N for i in range(N)]
    trans_b = [(i + 1) % N for i in range(N-1)] + [N-2]
    
    best_len = compute_threshold(trans_a, trans_b)
    best_a = trans_a[:]
    best_b = trans_b[:]
    
    if best_len == (N - 1) ** 2:
        return [best_a, best_b]
        
    # Local Search
    while time.time() < deadline:
        idx = random.randint(0, N - 1)
        old_val = best_b[idx]
        
        # Try a few random values
        for _ in range(20):
            new_val = random.randint(0, N - 1)
            if new_val == old_val:
                continue
                
            best_b[idx] = new_val
            new_len = compute_threshold(best_a, best_b)
            
            if new_len > best_len:
                best_len = new_len
                if best_len == (N - 1) ** 2:
                    return [best_a, best_b]
            else:
                best_b[idx] = old_val
                
    return [best_a, best_b]
# EVOLVE-BLOCK-END
