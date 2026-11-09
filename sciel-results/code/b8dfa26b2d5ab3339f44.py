import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def build_paley_orientation(p):
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    def count_bad_sets(n, orientation):
        # Reconstruct out-neighbors
        out_neighbors = [[] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    out_neighbors[i].append(j)
                else:
                    out_neighbors[j].append(i)
                idx += 1
        
        good_sets = set()
        for v in range(n):
            neigh = out_neighbors[v]
            if len(neigh) < 4:
                continue
            for S in combinations(neigh, 4):
                good_sets.add(S)
                
        total = 1
        for i in range(4):
            total = total * (n - i) // (i + 1)
            
        return total - len(good_sets)

    def local_search(n, time_limit):
        # Returns orientation if valid found, else None
        import random
        random.seed(42) # Fixed seed for reproducibility? Or use time?
        # Let's use time for variation
        random.seed(int(time.time() * 1000))
        
        end_time = time.time() + time_limit
        best_orientation = None
        best_bad = float('inf')
        
        # Try multiple restarts
        while time.time() < end_time:
            # Initialize random
            orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            current_bad = count_bad_sets(n, orientation)
            
            if current_bad == 0:
                return orientation
            
            if current_bad < best_bad:
                best_bad = current_bad
                best_orientation = orientation[:]
                
            stuck_count = 0
            while time.time() < end_time:
                if current_bad == 0:
                    return orientation
                
                # Pick random edge
                idx = random.randint(0, len(orientation) - 1)
                orientation[idx] ^= 1
                
                new_bad = count_bad_sets(n, orientation)
                
                if new_bad < current_bad or (new_bad == current_bad and random.random() < 0.5):
                    current_bad = new_bad
                    stuck_count = 0
                    if new_bad < best_bad:
                        best_bad = new_bad
                        best_orientation = orientation[:]
                else:
                    orientation[idx] ^= 1 # Revert
                    stuck_count += 1
                    
                if stuck_count > 100:
                    break # Restart
                    
        # Return best found if it was valid? 
        # The problem requires a VALID tournament.
        # If we didn't find bad_count == 0, we must return None.
        return None

    # Fallback: Paley 67
    best_n = 67
    best_orientation = build_paley_orientation(67)
    
    # Search for smaller n
    for n in range(31, 67):
        if time.time() > deadline - 5:
            break
            
        # Allocate time
        # Smaller n: more time? Or less?
        # Smaller n is easier to evaluate, so we can afford more iterations.
        # But if solution doesn't exist, we waste time.
        # Let's give 2 seconds to each n.
        alloc = 2.0
        
        result = local_search(n, alloc)
        if result is not None:
            best_n = n
            best_orientation = result
            break # Found smallest valid n
            
    return {'n': best_n, 'orientation': best_orientation}
