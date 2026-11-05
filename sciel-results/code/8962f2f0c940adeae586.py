import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
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

    def is_valid_swap(A, in_A, sum_counts, y, x):
        """Check if swapping y with x is valid without modifying state."""
        if x in in_A or y not in in_A:
            return False
        
        # Check 2x
        s = 2 * x
        contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
        if sum_counts[s] - contrib_y + 1 > 2:
            return False
            
        # Check x+z for z in A, z != y
        for z in A:
            if z == y:
                continue
            s = x + z
            contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
            if sum_counts[s] - contrib_y + 1 > 2:
                return False
        return True

    def perform_swap(A, in_A, sum_counts, y, x):
        remove_element(A, in_A, sum_counts, y)
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for z in A[:-1]:
            sum_counts[x+z] += 1

    def is_valid_1_for_2(A, in_A, sum_counts, y, x1, x2):
        """Check if removing y and adding x1, x2 is valid."""
        if y not in in_A or x1 in in_A or x2 in in_A or x1 == x2:
            return False
        
        # The new set is A' = (A \ {y}) U {x1, x2}
        # We need to check all sums involving x1 or x2 in the context of A'
        # Sums to check:
        # 2*x1, 2*x2
        # x1+x2
        # x1+z, x2+z for all z in A \ {y}
        
        # Helper to get count in A' for a sum s
        # count_A'(s) = count_A(s) - count_y(s) + count_x1(s) + count_x2(s)
        # Where count_y(s) is 1 if y is part of a pair summing to s in A, else 0.
        # Actually, count_y(s) is the number of pairs (a,b) in A such that a+b=s and (a==y or b==y).
        # Since a<=b, if y+y=s, count is 1. If y+z=s (z!=y), count is 1.
        
        def count_y_contribution(s):
            # Number of pairs in A summing to s that include y
            # Case 1: 2y = s
            if 2*y == s:
                return 1
            # Case 2: y + z = s => z = s - y
            z = s - y
            if z in in_A and z != y:
                return 1
            return 0

        def count_x_contribution(s, x):
            # Number of pairs in {x} U (A \ {y}) summing to s that include x
            # Pairs: (x, x) or (x, z) where z in A \ {y}
            if 2*x == s:
                return 1
            z = s - x
            if z in in_A and z != y and z != x: # z != x is redundant if 2x != s
                return 1
            return 0

        # Check 2*x1
        s = 2*x1
        c = sum_counts[s] - count_y_contribution(s) + count_x_contribution(s, x1) + count_x_contribution(s, x2)
        if c > 2:
            return False

        # Check 2*x2
        s = 2*x2
        c = sum_counts[s] - count_y_contribution(s) + count_x_contribution(s, x1) + count_x_contribution(s, x2)
        if c > 2:
            return False

        # Check x1+x2
        s = x1 + x2
        c = sum_counts[s] - count_y_contribution(s) + count_x_contribution(s, x1) + count_x_contribution(s, x2)
        if c > 2:
            return False

        # Check x1+z and x2+z for z in A \ {y}
        for z in A:
            if z == y:
                continue
            
            # Check x1 + z
            s = x1 + z
            # Contributions from x1: pair (x1, z) -> 1
            # Contributions from x2: pair (x2, ?) summing to x1+z? 
            # If x2 + w = x1+z => w = x1+z-x2. 
            # We need to know if (x2, w) is a valid pair in A'.
            # w must be in A \ {y} and w != x2.
            # This is getting complicated to calculate on the fly for every z.
            
            # Alternative: Just simulate the counts for the specific sums.
            # Since we only care about sums involving x1 or x2, 
            # and we are checking x1+z, the sum is s = x1+z.
            # The new count for s is:
            # old_count(s) 
            # - (1 if y is in a pair summing to s in A)
            # + (1 if x1 is in a pair summing to s in A')
            # + (1 if x2 is in a pair summing to s in A')
            
            # Pairs in A summing to s involving y:
            # y + w = s => w = s - y. If w in A and w != y, then 1.
            # Or 2y=s.
            
            # Pairs in A' summing to s involving x1:
            # x1 + v = s => v = s - x1 = z.
            # Since z is in A \ {y}, the pair (x1, z) exists in A'.
            # So contribution is 1.
            
            # Pairs in A' summing to s involving x2:
            # x2 + w = s => w = s - x2 = x1 + z - x2.
            # If w is in A \ {y} and w != x2, then 1.
            
            contrib_y = 0
            if 2*y == s:
                contrib_y = 1
            else:
                w_y = s - y
                if w_y in in_A and w_y != y:
                    contrib_y = 1
            
            contrib_x1 = 1 # Pair (x1, z)
            
            contrib_x2 = 0
            if 2*x2 == s:
                contrib_x2 = 1
            else:
                w_x2 = s - x2
                if w_x2 in in_A and w_x2 != y and w_x2 != x2:
                    contrib_x2 = 1
            
            new_count = sum_counts[s] - contrib_y + contrib_x1 + contrib_x2
            if new_count > 2:
                return False
                
            # Check x2 + z
            s = x2 + z
            contrib_y = 0
            if 2*y == s:
                contrib_y = 1
            else:
                w_y = s - y
                if w_y in in_A and w_y != y:
                    contrib_y = 1
            
            contrib_x2 = 1 # Pair (x2, z)
            
            contrib_x1 = 0
            if 2*x1 == s:
                contrib_x1 = 1
            else:
                w_x1 = s - x1
                if w_x1 in in_A and w_x1 != y and w_x1 != x1:
                    contrib_x1 = 1
            
            new_count = sum_counts[s] - contrib_y + contrib_x1 + contrib_x2
            if new_count > 2:
                return False
                
        return True

    def perform_1_for_2(A, in_A, sum_counts, y, x1, x2):
        remove_element(A, in_A, sum_counts, y)
        # Add x1
        in_A.add(x1)
        A.append(x1)
        sum_counts[2*x1] += 1
        for z in A[:-1]:
            sum_counts[x1+z] += 1
        # Add x2
        in_A.add(x2)
        A.append(x2)
        sum_counts[2*x2] += 1
        for z in A[:-1]:
            sum_counts[x2+z] += 1

    # Initialize with greedy
    for x in range(1, N + 1):
        if try_add(A, in_A, sum_counts, x):
            continue
    
    best_set = A[:]
    best_size = len(A)
    
    # Local Search
    T = 10.0
    cooling_rate = 0.995
    
    while time.time() < deadline:
        # 1. Try to add any element
        added = False
        candidates = [x for x in range(1, N + 1) if x not in in_A]
        random.shuffle(candidates)
        
        for x in candidates:
            if try_add(A, in_A, sum_counts, x):
                added = True
                if len(A) > best_size:
                    best_size = len(A)
                    best_set = A[:]
                break
        
        if added:
            continue
        
        # 2. Try swaps (1-for-1)
        swapped = False
        for _ in range(50):
            if not A or not candidates:
                break
            y = random.choice(A)
            x = random.choice(candidates)
            if is_valid_swap(A, in_A, sum_counts, y, x):
                perform_swap(A, in_A, sum_counts, y, x)
                swapped = True
                if len(A) > best_size:
                    best_size = len(A)
                    best_set = A[:]
                break
        
        if swapped:
            T *= cooling_rate
            continue

        # 3. Try 1-for-2 moves (Structural Change)
        # This allows growing the set by 1
        did_1_for_2 = False
        # Limit attempts to avoid slowing down too much
        for _ in range(10):
            if not A or len(candidates) < 2:
                break
            y = random.choice(A)
            # Pick two distinct candidates
            idx1 = random.randint(0, len(candidates) - 1)
            idx2 = random.randint(0, len(candidates) - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, len(candidates) - 1)
            
            x1 = candidates[idx1]
            x2 = candidates[idx2]
            
            if is_valid_1_for_2(A, in_A, sum_counts, y, x1, x2):
                perform_1_for_2(A, in_A, sum_counts, y, x1, x2)
                did_1_for_2 = True
                if len(A) > best_size:
                    best_size = len(A)
                    best_set = A[:]
                break
        
        if did_1_for_2:
            T *= cooling_rate
            continue

        # 4. If stuck, remove a random element to break deadlock
        if A and random.random() < 0.3:
            y = random.choice(A)
            remove_element(A, in_A, sum_counts, y)
            T = max(T * 1.1, 1.0)
        
        T *= cooling_rate
        T = max(T, 0.1)
    
    return best_set
