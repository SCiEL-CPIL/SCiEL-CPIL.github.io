import time
import random
import math
import numpy as np

def _score(U):
    """Compute the score for a set U containing 0."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        U = [0] + [x for x in U if x > 0]
        if not U:
            U = [0]
    n = len(U)
    if n == 1:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute sumset and diffset using numpy
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    
    max_u = U_arr[-1]
    if max_u == 0:
        return 0.0
    
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    
    return 1.0 + math.log(ratio) / denom

def _score_fast(U_tuple):
    """Faster score computation for sorted tuple."""
    U = list(U_tuple)
    n = len(U)
    if n == 1:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    max_u = U_arr[-1]
    if max_u == 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _greedy_construct(deadline, max_size=30, max_val=200):
    """Greedy constructive algorithm: add elements one at a time to maximize score."""
    U = [0]
    current_score = _score(U)
    
    for step in range(max_size - 1):
        if time.time() > deadline:
            break
        
        best_candidate = None
        best_gain = 0.0
        
        # Candidate elements to try
        candidates = set()
        # Small numbers
        for i in range(1, min(max_val, 50)):
            candidates.add(i)
        # Near current max
        max_u = max(U)
        for i in range(max(1, max_u - 3), max_u + 15):
            candidates.add(i)
        # Lacunary-style: powers
        for p in [2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 19, 25, 32, 41, 53, 68, 87, 111, 142]:
            if p <= max_val:
                candidates.add(p)
        # Random
        for _ in range(10):
            candidates.add(random.randint(1, max_val))
        
        for x in candidates:
            if x in U:
                continue
            new_U = U + [x]
            new_U.sort()
            s = _score(new_U)
            gain = s - current_score
            if gain > best_gain:
                best_gain = gain
                best_candidate = x
        
        if best_candidate is not None:
            U.append(best_candidate)
            U.sort()
            current_score += best_gain
        else:
            break
    
    return U, current_score

def _local_search(U, deadline, max_iterations=500):
    """Local search to improve a set."""
    U = sorted(U)
    current_score = _score(U)
    max_val = max(U) * 3 + 50
    max_size = len(U) + 5
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Generate neighbors
        neighbors = []
        
        # Try removing an element (except 0)
        for i in range(1, len(U)):
            if len(U) > 2:
                new_U = U[:i] + U[i+1:]
                neighbors.append(new_U)
        
        # Try adding an element
        candidates_to_add = set()
        for i in range(1, min(max_val, 100)):
            candidates_to_add.add(i)
        for i in range(max(1, max(U) - 5), max(U) + 20):
            candidates_to_add.add(i)
        for _ in range(15):
            candidates_to_add.add(random.randint(1, max_val))
        
        for x in candidates_to_add:
            if x not in U and len(U) < max_size:
                new_U = U + [x]
                new_U.sort()
                neighbors.append(new_U)
        
        # Try replacing an element
        for i in range(1, len(U)):
            for delta in [-3, -2, -1, 1, 2, 3, 5, 8, 13, 21]:
                new_val = U[i] + delta
                if new_val > 0 and new_val not in U:
                    new_U = U[:]
                    new_U[i] = new_val
                    new_U.sort()
                    neighbors.append(new_U)
        
        random.shuffle(neighbors)
        
        best_neighbor = None
        best_neighbor_score = current_score
        
        for n in neighbors:
            if time.time() > deadline:
                break
            s = _score(n)
            if s > best_neighbor_score:
                best_neighbor_score = s
                best_neighbor = n
        
        if best_neighbor is not None:
            U = best_neighbor
            current_score = best_neighbor_score
        else:
            break
    
    return U, current_score

def _simulated_annealing(U, deadline, initial_temp=0.05, cooling=0.95, max_iterations=1000):
    """Simulated annealing to escape local optima."""
    U = sorted(U)
    current_score = _score(U)
    temp = initial_temp
    max_val = max(U) * 3 + 50
    max_size = len(U) + 5
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Generate a random neighbor
        move_type = random.choice(['add', 'remove', 'replace', 'swap'])
        new_U = U[:]
        
        if move_type == 'add':
            if len(new_U) < max_size:
                x = random.randint(1, max_val)
                if x not in new_U:
                    new_U.append(x)
                    new_U.sort()
        
        elif move_type == 'remove':
            if len(new_U) > 2:
                idx = random.randint(1, len(new_U) - 1)
                new_U.pop(idx)
        
        elif move_type == 'replace':
            if len(new_U) > 1:
                idx = random.randint(1, len(new_U) - 1)
                delta = random.randint(-10, 10)
                new_val = new_U[idx] + delta
                if new_val > 0 and new_val not in new_U:
                    new_U[idx] = new_val
                    new_U.sort()
        
        elif move_type == 'swap':
            if len(new_U) > 2:
                idx = random.randint(1, len(new_U) - 1)
                delta = random.randint(-5, 5)
                new_val = new_U[idx] + delta
                if new_val > 0 and new_val not in new_U:
                    new_U[idx] = new_val
                    new_U.sort()
        
        if new_U == U:
            temp *= cooling
            continue
        
        new_score = _score(new_U)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
            U = new_U
            current_score = new_score
        
        temp *= cooling
    
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    # Phase 1: Greedy construction from multiple starting points
    starting_sets = [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, 5],
        [0, 1, 3, 7],
        [0, 1, 4, 9],
    ]
    
    for start in starting_sets:
        if time.time() > deadline:
            break
        U, s = _greedy_construct(deadline, max_size=25, max_val=150)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Phase 2: Local search on best
    if time.time() < deadline:
        U, s = _local_search(best_U, deadline, max_iterations=300)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Phase 3: Simulated annealing from best and other good sets
    if time.time() < deadline:
        U, s = _simulated_annealing(best_U, deadline, initial_temp=0.03, cooling=0.97, max_iterations=800)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Phase 4: More local search passes
    if time.time() < deadline:
        U, s = _local_search(best_U, deadline, max_iterations=200)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Phase 5: Random restarts with greedy + local search
    restarts = 0
    while time.time() < deadline and restarts < 30:
        restarts += 1
        size = random.randint(3, 12)
        max_val = random.randint(20, 100)
        U = {0}
        while len(U) < size:
            U.add(random.randint(1, max_val))
        U = sorted(U)
        
        # Greedy add from here
        for step in range(5):
            if time.time() > deadline:
                break
            best_candidate = None
            best_gain = 0.0
            current_s = _score(U)
            candidates = set(range(1, min(max_val + 20, 120)))
            for x in random.sample(list(candidates), min(30, len(candidates))):
                if x not in U:
                    new_U = U + [x]
                    new_U.sort()
                    s = _score(new_U)
                    gain = s - current_s
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = x
            if best_candidate is not None:
                U.append(best_candidate)
                U.sort()
        
        # Local search
        if time.time() < deadline:
            U, s = _local_search(U, deadline, max_iterations=100)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Final polish: very fine local search
    if time.time() < deadline:
        U, s = _local_search(best_U, deadline, max_iterations=100)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    return best_U
