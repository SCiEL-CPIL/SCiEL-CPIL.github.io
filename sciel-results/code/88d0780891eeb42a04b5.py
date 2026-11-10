"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0  # safety margin before 90s kill
    n = 128
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129
    targets = list(range(MIN_SUM, MAX_SUM + 1))
    
    # Precompute for each element x, the sums it contributes to when paired with y in [0,128]
    # For a given x, the sums x+y for y in [0,128] range from x to x+128
    # We only care about sums in [128, 256]
    
    # Helper to calculate multiplicities for a given set A
    # Returns a numpy array of length NUM_SUMS
    def calc_mult(A):
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        A_list = sorted(A)
        n_elem = len(A_list)
        for i in range(n_elem):
            a = A_list[i]
            # a + a
            s = 2 * a
            if MIN_SUM <= s <= MAX_SUM:
                M[s - MIN_SUM] += 1
            # a + b for b > a
            for j in range(i + 1, n_elem):
                b = A_list[j]
                s = a + b
                if MIN_SUM <= s <= MAX_SUM:
                    M[s - MIN_SUM] += 1
        return M
    
    # Helper: remove element x from A and update M incrementally
    def remove_from_A(x, A, M):
        """Remove x from A, update M. Returns (new_M) or None if invalid."""
        new_M = M.copy()
        # For each a in A (including x itself for 2x case):
        # If a != x: sum = a + x, decrement
        # If a == x: sum = 2*x, decrement
        for a in A:
            s = a + x
            if MIN_SUM <= s <= MAX_SUM:
                new_M[s - MIN_SUM] -= 1
        # Check validity: all mults must be >= 1
        if np.min(new_M) < 1:
            return None
        return new_M
    
    # Helper: add element x to A and update M incrementally
    def add_to_A(x, A, M):
        """Add x to A, update M. Returns new_M."""
        new_M = M.copy()
        # For each a in A:
        # sum = a + x, increment
        for a in A:
            s = a + x
            if MIN_SUM <= s <= MAX_SUM:
                new_M[s - MIN_SUM] += 1
        # x + x
        s = 2 * x
        if MIN_SUM <= s <= MAX_SUM:
            new_M[s - MIN_SUM] += 1
        return new_M
    
    # Try multiple initial configurations to find a good starting point
    initial_sets = [
        set(range(64, 129)),   # Upper half
        set(range(0, 65)),     # Lower half
        set(range(32, 97)),    # Middle shifted
        set(range(0, 129, 2)), # Every other
        set(range(16, 113)),   # Another middle
    ]
    
    best_A = None
    best_score = float('inf')
    
    for init_set in initial_sets:
        M_init = calc_mult(init_set)
        if np.min(M_init) < 1:
            continue  # Skip invalid initial sets
        
        A = set(init_set)
        M = M_init
        current_score = int(np.max(M))
        
        if current_score < best_score:
            best_score = current_score
            best_A = set(A)
    
    if best_A is None:
        # Fallback to a valid set
        A = set(range(64, 129))
        M = calc_mult(A)
        best_A = set(A)
        best_score = int(np.max(M))
    
    # Simulated Annealing with improved moves
    current_A = set(best_A)
    current_M = calc_mult(current_A)
    current_score = int(np.max(current_M))
    
    T = 5.0
    cooling = 0.9995
    ALL_ELEMENTS = list(range(0, 129))
    
    while time.time() < deadline:
        # Choose a move type with bias toward removal/swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove an element
            if len(current_A) <= 10:  # Don't make set too small
                T *= cooling
                continue
            
            # Prefer removing elements that contribute to high-multiplicity sums
            A_list = list(current_A)
            x = random.choice(A_list)
            
            new_M = remove_from_A(x, current_A, current_M)
            if new_M is not None:
                score_new = int(np.max(new_M))
                delta = score_new - current_score
                if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                    current_A.remove(x)
                    current_M = new_M
                    current_score = score_new
                    if score_new < best_score:
                        best_score = score_new
                        best_A = set(current_A)
        
        elif move_type < 0.8:
            # Swap: remove x, add y
            if len(current_A) == 0:
                T *= cooling
                continue
            
            x = random.choice(list(current_A))
            # Choose y not in A
            candidates = [y for y in ALL_ELEMENTS if y not in current_A]
            if not candidates:
                T *= cooling
                continue
            y = random.choice(candidates)
            
            # Simulate: remove x then add y
            temp_M = remove_from_A(x, current_A, current_M)
            if temp_M is not None:
                # Create temp_A without x
                temp_A = set(current_A)
                temp_A.remove(x)
                new_M = add_to_A(y, temp_A, temp_M)
                score_new = int(np.max(new_M))
                delta = score_new - current_score
                if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                    current_A.remove(x)
                    current_A.add(y)
                    current_M = new_M
                    current_score = score_new
                    if score_new < best_score:
                        best_score = score_new
                        best_A = set(current_A)
        
        else:
            # Add an element (less common)
            candidates = [y for y in ALL_ELEMENTS if y not in current_A]
            if not candidates:
                T *= cooling
                continue
            y = random.choice(candidates)
            
            new_M = add_to_A(y, current_A, current_M)
            score_new = int(np.max(new_M))
            delta = score_new - current_score
            # Adding usually increases max, so only accept with low probability
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 0.01)) * 0.5:
                current_A.add(y)
                current_M = new_M
                current_score = score_new
        
        # Cool down
        T *= cooling
        if T < 0.05:
            T = 0.05
    
    # Final polish: try to remove more elements from best_A
    if time.time() < deadline - 1.0:
        A = set(best_A)
        M = calc_mult(A)
        removed = True
        while removed and time.time() < deadline - 1.0:
            removed = False
            A_list = list(A)
            random.shuffle(A_list)
            for x in A_list:
                if x not in A:
                    continue
                new_M = remove_from_A(x, A, M)
                if new_M is not None:
                    score_new = int(np.max(new_M))
                    if score_new <= best_score:
                        A.remove(x)
                        M = new_M
                        if score_new < best_score:
                            best_score = score_new
                            best_A = set(A)
                        removed = True
    
    return list(best_A)
# EVOLVE-BLOCK-END
