import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # safety margin
    
    N = q ** n
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Fast neighbor precomputation using digit extraction
    neighbors = [[] for _ in range(N)]
    # Precompute powers of q
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q
    
    for i in range(N):
        # Extract digits
        digits = [0] * n
        temp = i
        for pos in range(n - 1, -1, -1):
            digits[pos] = temp % q
            temp //= q
        
        neighbors[i].append(i)
        for pos in range(n):
            orig = digits[pos]
            pw = q_powers[pos]
            for val in range(q):
                if val != orig:
                    neighbors[i].append(i + (val - orig) * pw)

    best_code = None
    best_size = N + 1
    
    # Multi-restart greedy with random tie-breaking
    while time.time() < deadline:
        # Greedy Construction
        covered = [False] * N
        gain = np.array([len(neighbors[i]) for i in range(N)], dtype=np.int32)
        
        code = []
        
        while True:
            # Find max gain
            max_gain = int(np.max(gain))
            
            if max_gain == 0:
                break
            
            # Find all indices with max gain
            candidates = np.where(gain == max_gain)[0]
            
            # Random choice among candidates
            best_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            
            # Add best_idx to code
            code.append(best_idx)
            
            # Update covered and gains
            for v in neighbors[best_idx]:
                if not covered[v]:
                    covered[v] = True
                    for u in neighbors[v]:
                        gain[u] -= 1
            
            # Time check
            if time.time() > deadline:
                break

        # Peeling Phase
        cover_count = [0] * N
        for c_idx in code:
            for v in neighbors[c_idx]:
                cover_count[v] += 1
        
        code_set = set(code)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for c_idx in code:
                essential = False
                for v in neighbors[c_idx]:
                    if cover_count[v] == 1:
                        essential = True
                        break
                if not essential:
                    to_remove.append(c_idx)
                    
            if to_remove:
                for c_idx in to_remove:
                    code_set.discard(c_idx)
                    for v in neighbors[c_idx]:
                        cover_count[v] -= 1
                code = list(code_set)
                changed = True
        
        # Update best
        if len(code) < best_size:
            best_size = len(code)
            best_code = code

    # Convert indices to lists of ints
    final_code = [idx_to_word(idx) for idx in best_code]
    
    return final_code
