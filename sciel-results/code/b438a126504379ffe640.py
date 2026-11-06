import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe within 60s
    
    n = 128
    target_start = 128
    target_end = 256
    
    # Precompute pair information for fast evaluation
    # For each target t in [128, 256], store the list of (a, b) unordered pairs with a<=b
    # This allows fast evaluation of multiplicity for a given set A
    
    def evaluate(A_set):
        """
        Returns (max_mult, num_uncovered).
        """
        if not A_set:
            return 1000, 129
        
        # Create indicator vector
        v = np.zeros(129, dtype=np.int32)
        for x in A_set:
            if 0 <= x <= 128:
                v[x] = 1
        
        # Convolution gives ordered sums
        conv = np.convolve(v, v, mode='full')
        
        # Extract target range [128, 256]
        ordered = conv[target_start:target_end + 1]
        
        # Calculate diagonal contributions
        target_indices = np.arange(target_start, target_end + 1)
        diag = np.zeros(len(target_indices), dtype=np.int32)
        even_mask = (target_indices % 2 == 0)
        if np.any(even_mask):
            even_indices = target_indices[even_mask]
            mid_indices = even_indices // 2
            diag[even_mask] = v[mid_indices]
            
        unordered = (ordered + diag) // 2
        
        max_mult = int(np.max(unordered))
        num_uncovered = int(np.sum(unordered == 0))
        
        return max_mult, num_uncovered

    def cost(A_set):
        m, u = evaluate(A_set)
        return u * 10000 + m

    # 1. Constructive mixed-radix basis initialization
    # This is known to achieve peak=1 in some cases
    # Strategy: build A using a mixed-radix approach
    def constructive_init():
        """
        Construct A using a mixed-radix basis approach.
        For n=128, we can use a binary-like construction.
        """
        A = set()
        
        # Start with 0 and 128 (needed for extremes)
        A.add(0)
        A.add(128)
        
        # Add powers of 2 to enable all sums
        # This gives us a "complete" set where every number in [0, 256] 
        # can be represented, but we need to check [128, 256]
        
        # Better approach: use a greedy construction
        # Start with {0, 128} and add elements to cover all targets
        # while minimizing max multiplicity
        
        current_A = set([0, 128])
        
        # For each target, if not covered, add an element that helps
        for _ in range(50):  # Max 50 additions
            if time.time() > deadline - 40:
                break
                
            m, u = evaluate(current_A)
            if u == 0 and m <= 2:
                break
                
            # Find uncovered targets
            v = np.zeros(129, dtype=np.int32)
            for x in current_A:
                v[x] = 1
            conv = np.convolve(v, v, mode='full')
            ordered = conv[target_start:target_end + 1]
            target_indices = np.arange(target_start, target_end + 1)
            diag = np.zeros(len(target_indices), dtype=np.int32)
            even_mask = (target_indices % 2 == 0)
            if np.any(even_mask):
                even_indices = target_indices[even_mask]
                mid_indices = even_indices // 2
                diag[even_mask] = v[mid_indices]
            unordered = (ordered + diag) // 2
            
            uncovered_targets = target_indices[unordered == 0]
            if len(uncovered_targets) == 0:
                break
            
            # For each uncovered target, find the best element to add
            best_elem = None
            best_score = float('inf')
            
            for t in uncovered_targets[:5]:  # Check first few uncovered
                for a in range(0, 129):
                    b = t - a
                    if b < a or b > 128:
                        continue
                    # If we add a (or b), does it help?
                    if a not in current_A:
                        test_A = current_A | {a}
                        m_t, u_t = evaluate(test_A)
                        score_t = u_t * 100 + m_t
                        if score_t < best_score:
                            best_score = score_t
                            best_elem = a
                    if b not in current_A and b != a:
                        test_A = current_A | {b}
                        m_t, u_t = evaluate(test_A)
                        score_t = u_t * 100 + m_t
                        if score_t < best_score:
                            best_score = score_t
                            best_elem = b
            
            if best_elem is not None:
                current_A.add(best_elem)
            else:
                # Random addition
                current_A.add(random.randint(0, 128))
        
        return current_A

    # 2. Try multiple constructive initializations
    best_A = set()
    best_cost = float('inf')
    
    # Constructive approach
    if time.time() < deadline - 30:
        A_cons = constructive_init()
        c_cons = cost(A_cons)
        if c_cons < best_cost:
            best_cost = c_cons
            best_A = set(A_cons)
        
        # Local improvement on constructive
        current_A = set(A_cons)
        current_c = c_cons
        
        for _ in range(300):
            if time.time() > deadline - 30:
                break
            improved = False
            
            # Try removing
            if len(current_A) > 2:
                x = random.choice(list(current_A))
                test_A = current_A - {x}
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
            
            if not improved:
                break
        
        if current_c < best_cost:
            best_cost = current_c
            best_A = set(current_A)
    
    # 3. Random restarts with local search
    for target_size in [15, 20, 25, 30]:
        if time.time() > deadline - 15:
            break
            
        for _ in range(15):
            if time.time() > deadline - 15:
                break
            A = set()
            A.add(128)
            while len(A) < target_size:
                A.add(random.randint(0, 128))
            
            c = cost(A)
            if c < best_cost:
                best_cost = c
                best_A = set(A)
                
            # Quick local improvement
            current_A = set(A)
            current_c = c
            
            for _ in range(150):
                if time.time() > deadline - 15:
                    break
                improved = False
                
                if len(current_A) > 2:
                    x = random.choice(list(current_A))
                    test_A = current_A - {x}
                    c_test = cost(test_A)
                    if c_test < current_c:
                        current_A = test_A
                        current_c = c_test
                        improved = True
                
                if not improved:
                    x = random.choice(list(current_A))
                    y = random.randint(0, 128)
                    if y != x:
                        test_A = (current_A - {x}) | {y}
                        c_test = cost(test_A)
                        if c_test < current_c:
                            current_A = test_A
                            current_c = c_test
                            improved = True
                
                if not improved:
                    break
                    
            if current_c < best_cost:
                best_cost = current_c
                best_A = set(current_A)

    # 4. Simulated Annealing Polish
    current_A = set(best_A)
    current_c = cost(current_A)
    best_global_A = set(current_A)
    best_global_c = current_c
    
    T = 3.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline - 3:
        move_type = random.random()
        new_A = set(current_A)
        
        if move_type < 0.3 and len(current_A) > 2:
            x = random.choice(list(current_A))
            new_A.remove(x)
        elif move_type < 0.6:
            x = random.randint(0, 128)
            if x in current_A:
                continue
            new_A.add(x)
        else:
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
        
    # 5. Final Deterministic Polish
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        m_best, u_best = evaluate(best_global_A)
        
        for x in sorted(best_global_A):
            if time.time() > deadline - 1:
                break
            if len(best_global_A) == 1:
                continue
            
            test_A = best_global_A - {x}
            if not test_A:
                continue
            m_test, u_test = evaluate(test_A)
            
            if u_test == 0 and u_best == 0:
                if m_test <= m_best:
                    best_global_A = test_A
                    improved = True
                    m_best, u_best = evaluate(best_global_A)
                    break
            elif u_best > 0:
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

    # Ensure 128 is in the set
    if 128 not in best_global_A:
        best_global_A.add(128)

    return sorted(best_global_A)
