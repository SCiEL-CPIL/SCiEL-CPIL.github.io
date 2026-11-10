import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large subset of Z/205Z with no two elements differing by a non-zero square.
    Uses a combination of randomized greedy initialization and local search (swaps).
    """
    m = 205
    start_time = time.time()
    time_limit = 580  # Safety margin for 600s limit
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Precompute neighbors for fast conflict checking
    # neighbors[i] is a set of j such that (i-j) % m is in forbidden
    neighbors = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            if (i - j) % m in forbidden:
                neighbors[i].add(j)
                
    best_set = []
    best_size = 0
    
    def is_compatible(candidate, current_set):
        # Check if candidate can be added to current_set
        # i.e., no neighbor of candidate is in current_set
        for s in current_set:
            if candidate in neighbors[s]:
                return False
        return True

    def greedy_construct(ordering):
        s = []
        for v in ordering:
            if is_compatible(v, s):
                s.append(v)
        return s

    def local_search(s):
        """
        Attempts to improve the set s using swaps.
        Returns the improved set.
        """
        s = list(s)
        stuck_counter = 0
        max_swaps = 5000
        current_size = len(s)
        
        # Convert to set for O(1) lookup if needed, but list is fine for small N
        s_set = set(s)
        
        for _ in range(max_swaps):
            if time.time() - start_time > time_limit:
                break
            
            # 1. Try to add a vertex
            added = False
            # Randomly pick a few candidates to try adding
            candidates = [v for v in range(m) if v not in s_set]
            random.shuffle(candidates)
            for v in candidates[:10]: # Try a few
                if is_compatible(v, s):
                    s.append(v)
                    s_set.add(v)
                    current_size += 1
                    added = True
                    stuck_counter = 0
                    break
            
            if added:
                continue
                
            # 2. If no add, try a swap
            # Pick random u in s, v not in s
            if len(s) == 0:
                break
            u = random.choice(s)
            v = random.choice(candidates) if candidates else None
            if v is None:
                break
            
            # Check if swapping u for v keeps independence
            # New set: (s \ {u}) U {v}
            # v must not conflict with any s' in s \ {u}
            compatible = True
            for s_prime in s:
                if s_prime == u:
                    continue
                if v in neighbors[s_prime]:
                    compatible = False
                    break
            
            if compatible:
                # Perform swap
                s.remove(u)
                s_set.remove(u)
                s.append(v)
                s_set.add(v)
                # Size unchanged, so stuck_counter doesn't reset for "improvement"
                # but we did change state
                stuck_counter = 0
            else:
                stuck_counter += 1
                if stuck_counter > 100:
                    break
        
        return s

    # Main search loop
    while time.time() - start_time < time_limit:
        # Random ordering
        ordering = list(range(m))
        random.shuffle(ordering)
        
        # Greedy
        current_set = greedy_construct(ordering)
        
        # Local Search
        current_set = local_search(current_set)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            # If we found a very large set, we might be near optimal
            if best_size >= 15: # Arbitrary high bar
                break

    return np.array(best_set, dtype=np.int64)
