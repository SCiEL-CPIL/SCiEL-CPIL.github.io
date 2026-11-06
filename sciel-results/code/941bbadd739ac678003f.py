import time
import random
import numpy as np

def construct():
    n = 512
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer before 60s kill

    # Precompute for each x the list of triples (a,b,c) such that {a,b,c,x} forms a 4-AP
    # We'll use numpy arrays for fast membership testing
    conflict_triples = [None] * (n + 1)
    for x in range(1, n + 1):
        triples = []
        # Case 1: x is first term: x, x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            triples.append((x + d, x + 2 * d, x + 3 * d))
        
        # Case 2: x is second term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            triples.append((x - d, x + d, x + 2 * d))
        
        # Case 3: x is third term: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            triples.append((x - 2 * d, x - d, x + d))
        
        # Case 4: x is fourth term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            triples.append((x - 3 * d, x - 2 * d, x - d))
        
        conflict_triples[x] = np.array(triples, dtype=np.int32) if triples else np.empty((0, 3), dtype=np.int32)

    def creates_4ap_bitset(S, x):
        """Check if adding x to set S creates a 4-AP. S is a numpy boolean array."""
        triples = conflict_triples[x]
        if len(triples) == 0:
            return False
        # Check if any triple has all 3 elements in S
        # S[triples] gives a (num_triples, 3) boolean array
        # We need any row where all 3 are True
        in_S = S[triples]
        return np.any(np.all(in_S, axis=1))

    def greedy_fill(S, order):
        """Greedily fill set S given an order. S is a numpy boolean array."""
        S = S.copy()
        for x in order:
            if not S[x]:
                if not creates_4ap_bitset(S, x):
                    S[x] = True
        return S

    random.seed(42)
    
    best_set = np.zeros(n + 1, dtype=bool)
    best_size = 0
    
    # Try multiple random orders to find a good starting point
    num_random_tries = 500
    for i in range(num_random_tries):
        if time.time() - start_time > time_limit * 0.4:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        current = greedy_fill(np.zeros(n + 1, dtype=bool), order)
        current_size = np.sum(current)
        if current_size > best_size:
            best_set = current.copy()
            best_size = current_size
    
    # Also try some structured orders
    for order in [list(range(1, n + 1)), list(range(n, 0, -1))]:
        if time.time() - start_time > time_limit * 0.5:
            break
        current = greedy_fill(np.zeros(n + 1, dtype=bool), order)
        current_size = np.sum(current)
        if current_size > best_size:
            best_set = current.copy()
            best_size = current_size

    # Local search with kicks
    current_set = best_set.copy()
    current_size = best_size
    
    while time.time() - start_time < time_limit:
        added = False
        candidates = [i for i in range(1, n + 1) if not current_set[i]]
        
        if not candidates:
            break
        
        random.shuffle(candidates)
        for x in candidates[:200]:
            if not creates_4ap_bitset(current_set, x):
                current_set[x] = True
                current_size += 1
                added = True
                if current_size > best_size:
                    best_set = current_set.copy()
                    best_size = current_size
                break
        
        if not added:
            # Kick: remove 1-4 random elements
            k = random.randint(1, 4)
            if current_size > k:
                to_remove = random.sample([i for i in range(1, n + 1) if current_set[i]], k)
                for x in to_remove:
                    current_set[x] = False
                    current_size -= 1
                
                # Greedy refill
                candidates = [i for i in range(1, n + 1) if not current_set[i]]
                random.shuffle(candidates)
                for x in candidates:
                    if not creates_4ap_bitset(current_set, x):
                        current_set[x] = True
                        current_size += 1
                
                if current_size > best_size:
                    best_set = current_set.copy()
                    best_size = current_size

    return sorted([i for i in range(1, n + 1) if best_set[i]])
