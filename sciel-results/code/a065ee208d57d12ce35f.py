import time
import random
import math

def compute_score(U):
    U = sorted(set(U))
    if not U or U[0] != 0:
        # Ensure 0 is in U
        U.insert(0, 0)
        U = sorted(U)
    n = len(U)
    if n == 1:
        return 0.0
        
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
            
    max_u = max(U)
    if max_u == 0:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        # Score would be <= 1. We want > 1.
        # But formula is 1 + log(ratio)/log(...). If ratio < 1, log is negative.
        # The prompt says "maximize the sumset/difference-set exponent lower bound".
        # Usually |U-U| >= |U+U| is not guaranteed, but for sets containing 0, 
        # |U-U| contains -U, 0, U. |U+U| contains U, 2U...
        # For {0, 1}, U+U={0,1,2}, U-U={-1,0,1}. Sizes 3 and 3. Ratio 1. Score 1.
        # For {0, 1, 3}, Ratio 7/6 > 1.
        pass
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 9],
        [0, 1, 2, 4],
        [0, 1, 3, 7],
        [0, 1, 4, 7],
        list(range(5))
    ]
    
    for c in candidates:
        s = compute_score(c)
        if s > best_score:
            best_score = s
            best_U = c[:]

    # Local Search
    # We will try to improve best_U
    current_U = best_U[:]
    current_score = best_score
    
    # Define neighborhood moves
    def get_neighbors(U):
        neighbors = []
        # Remove an element (except 0)
        for i in range(1, len(U)):
            new_U = U[:]
            new_U.pop(i)
            neighbors.append(new_U)
            
        # Add an element
        # Try adding small numbers or numbers near max
        max_u = max(U)
        candidates_to_add = set()
        # Small numbers
        for i in range(1, 20):
            if i not in U:
                candidates_to_add.add(i)
        # Near max
        for i in range(max(1, max_u - 5), max_u + 10):
            if i not in U:
                candidates_to_add.add(i)
        # Random larger
        for _ in range(5):
            candidates_to_add.add(random.randint(1, max_u + 50))
            
        for x in candidates_to_add:
            new_U = U[:]
            new_U.append(x)
            new_U.sort()
            neighbors.append(new_U)
            
        # Replace an element (except 0)
        for i in range(1, len(U)):
            for delta in [-2, -1, 1, 2, 3, 5, 10]:
                new_val = U[i] + delta
                if new_val > 0 and new_val not in U:
                    new_U = U[:]
                    new_U[i] = new_val
                    new_U.sort()
                    neighbors.append(new_U)
                    
        return neighbors

    # Hill Climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        neighbors = get_neighbors(current_U)
        # Shuffle to avoid bias
        random.shuffle(neighbors)
        
        best_neighbor = None
        best_neighbor_score = current_score
        
        for n in neighbors:
            if time.time() > deadline:
                break
            s = compute_score(n)
            if s > best_neighbor_score:
                best_neighbor_score = s
                best_neighbor = n
                
        if best_neighbor is not None:
            current_U = best_neighbor
            current_score = best_neighbor_score
            improved = True
            
            if current_score > best_score:
                best_score = current_score
                best_U = current_U[:]
                
    # Random Restarts
    restarts = 0
    while time.time() < deadline and restarts < 50:
        restarts += 1
        # Generate a random seed set
        size = random.randint(3, 10)
        max_val = random.randint(10, 100)
        U = {0}
        while len(U) < size:
            U.add(random.randint(1, max_val))
        U = list(U)
        
        # Hill climb from this seed
        current_U = U
        current_score = compute_score(current_U)
        
        local_improved = True
        while local_improved and time.time() < deadline:
            local_improved = False
            neighbors = get_neighbors(current_U)
            random.shuffle(neighbors)
            
            best_neighbor = None
            best_neighbor_score = current_score
            
            for n in neighbors:
                if time.time() > deadline:
                    break
                s = compute_score(n)
                if s > best_neighbor_score:
                    best_neighbor_score = s
                    best_neighbor = n
                    
            if best_neighbor is not None:
                current_U = best_neighbor
                current_score = best_neighbor_score
                local_improved = True
                
        if current_score > best_score:
            best_score = current_score
            best_U = current_U[:]

    return best_U
