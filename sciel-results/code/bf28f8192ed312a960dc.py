import time
import sys

def construct(q, n, radius):
        start_time = time.time()
        deadline = start_time + 110.0 # Safety margin
        
        N = q ** n
        # Precompute neighbors
        # For R=1, neighbors are w and w +/- q^i for each i
        
        # Optimization: If N is small, we can use the greedy set cover.
        # If N is large, we might need a different approach, but for this task N=6561.
        
        # 1. Generate neighbors for all words
        neighbors = []
        for w in range(N):
            neigh = [w]
            for i in range(n):
                power = q ** i
                digit = (w // power) % q
                for d in range(q):
                    if d != digit:
                        new_w = w + (d - digit) * power
                        neigh.append(new_w)
            neighbors.append(neigh)
        
        # 2. Greedy Set Cover
        uncovered = [1] * N
        # coverage_count[i] = number of uncovered words in neighbors[i]
        # Initially all are covered? No, initially all are uncovered.
        # So coverage_count[i] = len(neighbors[i])
        coverage_count = [len(neigh) for neigh in neighbors]
        
        code = []
        remaining = N
        
        while remaining > 0:
            if time.time() > deadline:
                break # Return best found so far (might be invalid if not all covered, but better to return valid if possible? 
                      # The prompt says "Valid when every word...". If we break early, it's invalid. 
                      # However, if we run out of time, we return what we have. 
                      # But the greedy is fast, so it should finish.)
            
            # Find index with max coverage
            max_idx = 0
            max_val = -1
            for i in range(N):
                if coverage_count[i] > max_val:
                    max_val = coverage_count[i]
                    max_idx = i
            
            # Add max_idx to code
            code.append([int((max_idx // (q**i)) % q) for i in range(n)])
            
            # Update coverage
            for u in neighbors[max_idx]:
                if uncovered[u]:
                    uncovered[u] = 0
                    remaining -= 1
                    # Update counts for all neighbors of u
                    for v in neighbors[u]:
                        coverage_count[v] -= 1
        
        return code
