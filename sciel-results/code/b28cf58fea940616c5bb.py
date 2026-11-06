import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe within 60s
    
    n = 128
    target_start = 128
    target_end = 256
    # Pre-compute indices for target range
    target_indices = np.arange(target_start, target_end + 1)
    
    def evaluate(A_set):
        """
        Returns (max_mult, num_uncovered).
        Uses vectorized numpy operations for speed.
        """
        if not A_set:
            return 1000, 100
        
        # Create indicator vector
        # A_set is a set of ints in [0, 128]
        v = np.zeros(129, dtype=np.int32)
        # Fast assignment
        # Since A_set is small, loop is fine, but let's try to be efficient
        for x in A_set:
            v[x] = 1
        
        # Convolution gives ordered sums
        # v is length 129, conv is length 257
        conv = np.convolve(v, v, mode='full')
        
        # Extract target range [128, 256]
        ordered = conv[target_start:target_end + 1]
        
        # Calculate diagonal contributions (where a == b)
        # For a sum t, if t is even and t/2 is in A, then the pair (t/2, t/2) is an unordered pair.
        # In ordered convolution, (t/2, t/2) is counted once.
        # In unordered pairs, (a, b) with a < b is counted once, (a, a) is counted once.
        # Formula: Unordered = (Ordered + Diagonal) / 2
        # Where Diagonal[t] = 1 if t is even and t/2 in A, else 0.
        
        # We need a vector 'diag' corresponding to target_indices
        # t is in target_indices
        # mid = t / 2
        # Check if mid is integer and in A
        mids = target_indices / 2.0
        is_even = (target_indices % 2 == 0)
        
        # We can just loop over the 129 targets, it's small enough if done in numpy?
        # Or construct the vector:
        diag = np.zeros(len(target_indices), dtype=np.int32)
        # For each t in target_indices, if even, check v[t//2]
        even_mask = (target_indices % 2 == 0)
        if np.any(even_mask):
            even_indices = target_indices[even_mask]
            mid_indices = even_indices // 2
            # Check if mid_indices are within 0..128 (they are, max 256//2=128)
            diag[even_mask] = v[mid_indices]
            
        unordered = (ordered + diag) // 2
        
        max_mult = int(np.max(unordered))
        num_uncovered = int(np.sum(unordered == 0))
        
        return max_mult, num_uncovered

    def cost(A_set):
        m, u = evaluate(A_set)
        # Penalty for uncovered is huge to prioritize coverage first
        # Then minimize max multiplicity
        return u * 10000 + m

    # 1. Smart Initialization
    best_A = set()
    best_cost = float('inf')
    
    # Try different sizes
    for target_size in [20, 25, 30, 35, 40]:
        if time.time() > deadline - 10:
            break
            
        for _ in range(20): # 20 restarts per size
            if time.time() > deadline - 10:
                break
            A = set()
            A.add(128)
            # Add random elements
            while len(A) < target_size:
                A.add(random.randint(0, 128))
            
            c = cost(A)
            if c < best_cost:
                best_cost = c
                best_A = set(A)
                
            # Quick local improvement on this init
            # Hill Climbing: try removing/adding/swapping
            current_A = set(A)
            current_c = c
            
            for _ in range(200): # Increased steps since eval is faster
                if time.time() > deadline - 10:
                    break
                improved = False
                # Try removing
                if len(current_A) > 1:
                    x = random.choice(list(current_A))
                    test_A = current_A - {x}
                    c_test = cost(test_A)
                    if c_test < current_c:
                        current_A = test_A
                        current_c = c_test
                        improved = True
                # Try adding
                if not improved:
                    x = random.randint(0, 128)
                    if x not in current_A:
                        test_A = current_A | {x}
                        c_test = cost(test_A)
                        if c_test < current_c:
                            current_A = test_A
                            current_c = c_test
                            improved = True
                # Try swapping
                if not improved and len(current_A) > 1:
                    x = random.choice(list(current_A))
                    y = random.randint(0, 128)
                    if y != x:
                        test_A = (current_A - {x}) | {y}
                        c_test = cost(test_A)
                        if c_test < current_c:
                            current_A = test_A
                            current_c = c_test
                            improved = True
                            
            if current_c < best_cost:
                best_cost = current_c
                best_A = set(current_A)

    # 2. Simulated Annealing / Global Polish
    current_A = set(best_A)
    current_c = cost(current_A)
    best_global_A = set(current_A)
    best_global_c = current_c
    
    T = 5.0
    T_min = 0.01
    cooling = 0.99
    
    steps = 0
    while time.time() < deadline - 2:
        steps += 1
        move_type = random.random()
        new_A = set(current_A)
        
        if move_type < 0.3 and len(current_A) > 1:
            # Remove
            x = random.choice(list(current_A))
            new_A.remove(x)
        elif move_type < 0.6:
            # Add
            x = random.randint(0, 128)
            if x in current_A:
                continue
            new_A.add(x)
        else:
            # Swap
            if len(current_A) < 2:
                continue
            x = random.choice(list(current_A))
            y = random.randint(0, 128)
            if y == x:
                continue
            new_A.remove(x)
            new_A.add(y)
            
        if not new_A:
            continue
            
        new_c = cost(new_A)
        delta = new_c - current_c
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_A = new_A
            current_c = new_c
            
            if current_c < best_global_c:
                best_global_c = current_c
                best_global_A = set(current_A)
                
        T = max(T * cooling, T_min)
        
    # 3. Final Deterministic Polish
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        m_best, u_best = evaluate(best_global_A)
        
        for x in sorted(best_global_A):
            if time.time() > deadline - 1:
                break
            if x == 128 and len(best_global_A) == 1:
                continue # Can't remove last element, and 128 is critical for 256
            
            test_A = best_global_A - {x}
            if not test_A:
                continue
            m_test, u_test = evaluate(test_A)
            
            # Accept if coverage is maintained and max mult is not increased
            if u_test == 0 and u_best == 0:
                if m_test <= m_best:
                    best_global_A = test_A
                    improved = True
                    # Re-evaluate best
                    m_best, u_best = evaluate(best_global_A)
                    break
            elif u_best > 0:
                # If we currently have uncovered, accept if it improves coverage or keeps same and reduces max
                if u_test < u_best:
                    best_global_A = test_A
                    improved = True
                    m_best, u_best = evaluate(best_global_A)
                    break
                elif u_test == u_best and m_test < m_best:
                    best_global_A = test_A
                    improved = True
                    m_best, u_best = evaluate(best_global_A)
                    break

    # Ensure 128 is in the set (it should be, but just in case)
    if 128 not in best_global_A:
        best_global_A.add(128)

    return sorted(best_global_A)
