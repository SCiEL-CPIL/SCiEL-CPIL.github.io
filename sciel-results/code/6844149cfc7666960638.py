import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for safety
    
    N = 1000
    max_sum = 2 * N
    
    # Data structures
    A = []
    in_A = set()
    sum_counts = [0] * (2 * N + 1)
    
    best_set = []
    best_size = 0
    
    def try_add(A, in_A, sum_counts, x):
        if x in in_A:
            return False
        if sum_counts[2*x] >= 2:
            return False
        for y in A:
            if sum_counts[x+y] >= 2:
                return False
        # Valid
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for y in A[:-1]:
            sum_counts[x+y] += 1
        return True

    def remove_element(A, in_A, sum_counts, x):
        if x not in in_A:
            return False
        in_A.remove(x)
        A.remove(x)
        sum_counts[2*x] -= 1
        for y in A:
            sum_counts[x+y] -= 1
        return True

    def can_add(x, A, in_A, sum_counts):
        """Check if x can be added without modifying state."""
        if x in in_A:
            return False
        if sum_counts[2*x] >= 2:
            return False
        for y in A:
            if sum_counts[x+y] >= 2:
                return False
        return True

    # Initialize with greedy
    for x in range(1, N + 1):
        if try_add(A, in_A, sum_counts, x):
            continue
    
    best_set = A[:]
    best_size = len(A)
    
    # Local Search
    # Strategy: 
    # 1. Try to add any element (greedy improvement).
    # 2. If stuck, try 1-out-2-in moves: remove one element, add two.
    # 3. If stuck, try random 1-out-1-in swaps to explore.
    
    while time.time() < deadline:
        # 1. Try to add any element
        added = False
        for x in range(1, N + 1):
            if x not in in_A:
                if try_add(A, in_A, sum_counts, x):
                    added = True
                    if len(A) > best_size:
                        best_size = len(A)
                        best_set = A[:]
                    break
        
        if added:
            continue
            
        # 2. Try 1-out-2-in moves
        # Remove y, try to add x1, then x2.
        # If we can add 2, net size +1.
        found_improvement = False
        indices = list(range(len(A)))
        random.shuffle(indices)
        
        for i in indices:
            y = A[i]
            remove_element(A, in_A, sum_counts, y)
            
            added_count = 0
            added_elements = []
            
            # Try to add as many as possible
            # We should prioritize elements that were blocked by y
            # But simple iteration works well enough
            
            for x in range(1, N + 1):
                if x not in in_A:
                    if can_add(x, A, in_A, sum_counts):
                        try_add(A, in_A, sum_counts, x)
                        added_count += 1
                        added_elements.append(x)
                        if added_count >= 2:
                            break
            
            if added_count >= 2:
                # Net gain: -1 (removed y) + 2 (added x1, x2) = +1
                if len(A) > best_size:
                    best_size = len(A)
                    best_set = A[:]
                found_improvement = True
                break
            else:
                # Revert: remove added elements, put y back
                for x in added_elements:
                    remove_element(A, in_A, sum_counts, x)
                try_add(A, in_A, sum_counts, y)
                
        if found_improvement:
            continue
            
        # 3. If stuck, try random 1-out-1-in swaps (neutral moves)
        # This helps escape local minima
        if random.random() < 0.2:
            y = random.choice(A)
            remove_element(A, in_A, sum_counts, y)
            
            valid_xs = []
            for x in range(1, N + 1):
                if x not in in_A:
                    if can_add(x, A, in_A, sum_counts):
                        valid_xs.append(x)
            
            if valid_xs:
                x = random.choice(valid_xs)
                try_add(A, in_A, sum_counts, x)
            else:
                # Put y back
                try_add(A, in_A, sum_counts, y)

    # Final Polish: Deterministic pass
    # Try adding any element
    for x in range(1, N + 1):
        if x not in in_A:
            if try_add(A, in_A, sum_counts, x):
                if len(A) > best_size:
                    best_size = len(A)
                    best_set = A[:]
                
    # Try 1-out-2-in one more time
    for y in A[:]: # Iterate over copy
        if y not in in_A: continue
        remove_element(A, in_A, sum_counts, y)
        added_count = 0
        added_elements = []
        for x in range(1, N + 1):
            if x not in in_A:
                if can_add(x, A, in_A, sum_counts):
                    try_add(A, in_A, sum_counts, x)
                    added_count += 1
                    added_elements.append(x)
                    if added_count >= 2:
                        break
        if added_count >= 2:
            if len(A) > best_size:
                best_size = len(A)
                best_set = A[:]
        else:
            for x in added_elements:
                remove_element(A, in_A, sum_counts, x)
            try_add(A, in_A, sum_counts, y)

    return best_set
