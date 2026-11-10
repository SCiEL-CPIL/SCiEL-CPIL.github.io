import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 128
    TARGET_MIN = 128
    TARGET_MAX = 256
    TARGET_SIZE = TARGET_MAX - TARGET_MIN + 1
    
    # Helper to calculate cost
    def get_cost(sums):
        # sums is a numpy array of multiplicities for sums 0..256
        # We only care about 128..256
        target_sums = sums[TARGET_MIN:TARGET_MAX+1]
        
        # Penalty for gaps
        gaps = np.sum(target_sums == 0)
        if gaps > 0:
            return 10000 + gaps * 100 + np.max(target_sums)
            
        # If no gaps, minimize peak, and secondarily sum of squares to flatten
        peak = np.max(target_sums)
        sq = np.sum(target_sums**2)
        # If peak is 1, this is the optimal solution (Value 1)
        # Cost will be 1000 + 129 = 1129
        return peak * 1000 + sq

    # Helper to update sums for adding/removing x
    # A is a set of ints
    # sums is a numpy array
    def apply_add(A, sums, x):
        for a in A:
            s = a + x
            if s <= 256:
                sums[s] += 1
        if 2*x <= 256:
            sums[2*x] += 1
        A.add(x)
        
    def apply_remove(A, sums, x):
        for a in A:
            if a == x:
                continue
            s = a + x
            if s <= 256:
                sums[s] -= 1
        if 2*x <= 256:
            sums[2*x] -= 1
        A.remove(x)

    # Initialize best solution
    best_A = None
    best_cost = float('inf')
    best_sums = None
    
    # Try a few different random initializations to escape local minima
    # Smaller sets are more likely to achieve peak multiplicity 1
    for init_idx in range(20):
        # Random set of size ~15-35
        size = random.randint(15, 35)
        A = set(random.sample(range(N+1), size))
        
        # Recalculate sums properly
        sums = np.zeros(257, dtype=np.int32)
        A_list = list(A)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if s <= 256:
                    sums[s] += 1
        
        current_cost = get_cost(sums)
        
        # Simulated Annealing
        T = 100.0
        T_min = 0.1
        cooling = 0.995
        steps = 30000 # Increased steps due to faster operations on smaller sets
        
        current_A = A
        current_sums = sums
        current_cost = get_cost(current_sums)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = set(current_A)
            best_sums = current_sums.copy()
            
        for step in range(steps):
            if time.time() > deadline:
                return sorted(best_A)
            
            # Early termination if we found a perfect solution (Peak 1, No Gaps)
            if current_cost < 1200: # Min cost for peak 1 is 1129
                return sorted(current_A)

            # Choose move: 10% Add, 10% Remove, 80% Swap
            move_type = random.random()
            
            if move_type < 0.1:
                # Add
                candidates = [x for x in range(N+1) if x not in current_A]
                if not candidates: continue
                x = random.choice(candidates)
                
                # Apply
                apply_add(current_A, current_sums, x)
                new_cost = get_cost(current_sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(current_A)
                        best_sums = current_sums.copy()
                else:
                    # Revert
                    apply_remove(current_A, current_sums, x)
                    
            elif move_type < 0.2:
                # Remove
                if len(current_A) < 10: continue
                x = random.choice(list(current_A))
                
                # Apply
                apply_remove(current_A, current_sums, x)
                new_cost = get_cost(current_sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(current_A)
                        best_sums = current_sums.copy()
                else:
                    # Revert
                    apply_add(current_A, current_sums, x)
                    
            else:
                # Swap
                if len(current_A) == 0: continue
                x = random.choice(list(current_A))
                candidates = [y for y in range(N+1) if y not in current_A]
                if not candidates: continue
                y = random.choice(candidates)
                
                # Apply remove x, add y
                apply_remove(current_A, current_sums, x)
                apply_add(current_A, current_sums, y)
                
                new_cost = get_cost(current_sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(current_A)
                        best_sums = current_sums.copy()
                else:
                    # Revert
                    apply_remove(current_A, current_sums, y)
                    apply_add(current_A, current_sums, x)
            
            T *= cooling
            if T < T_min:
                break

    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
        
    return sorted(best_A)
