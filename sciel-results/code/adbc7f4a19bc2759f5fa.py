import random
import math
import time

def construct():
    n = 256
    
    def is_4ap_free_with_add(A, x):
        """Check if adding x to A creates a 4-AP. Returns True if NO 4-AP is created."""
        # x is 1st term: x, x+d, x+2d, x+3d
        # x+d, x+2d, x+3d must be in A
        for d in range(1, (n - x) // 3 + 1):
            if (x + d) in A and (x + 2 * d) in A and (x + 3 * d) in A:
                return False
        # x is 2nd term: x-d, x, x+d, x+2d
        # x-d, x+d, x+2d must be in A
        for d in range(1, min(x - 1, (n - x) // 2) + 1):
            if (x - d) in A and (x + d) in A and (x + 2 * d) in A:
                return False
        # x is 3rd term: x-2d, x-d, x, x+d
        # x-2d, x-d, x+d must be in A
        for d in range(1, min((x - 2) // 2, n - x) + 1):
            if (x - 2 * d) in A and (x - d) in A and (x + d) in A:
                return False
        # x is 4th term: x-3d, x-2d, x-d, x
        # x-3d, x-2d, x-d must be in A
        for d in range(1, (x - 3) // 3 + 1):
            if (x - 3 * d) in A and (x - 2 * d) in A and (x - d) in A:
                return False
        return True

    def count_3aps_delta_add(A, x):
        """Count new 3-APs created by adding x to A."""
        count = 0
        # x is 1st: x, x+d, x+2d
        for d in range(1, (n - x) // 2 + 1):
            if (x + d) in A and (x + 2 * d) in A:
                count += 1
        # x is 2nd: x-d, x, x+d
        for d in range(1, min(x - 1, n - x) + 1):
            if (x - d) in A and (x + d) in A:
                count += 1
        # x is 3rd: x-2d, x-d, x
        for d in range(1, min((x - 2) // 2, n - x) + 1):
            if (x - 2 * d) in A and (x - d) in A:
                count += 1
        return count

    def count_3aps_delta_remove(A, x):
        """Count 3-APs lost by removing x from A."""
        count = 0
        # x is 1st: x, x+d, x+2d
        for d in range(1, (n - x) // 2 + 1):
            if (x + d) in A and (x + 2 * d) in A:
                count += 1
        # x is 2nd: x-d, x, x+d
        for d in range(1, min(x - 1, n - x) + 1):
            if (x - d) in A and (x + d) in A:
                count += 1
        # x is 3rd: x-2d, x-d, x
        for d in range(1, min((x - 2) // 2, n - x) + 1):
            if (x - 2 * d) in A and (x - d) in A:
                count += 1
        return count

    def total_3aps(S):
        """Calculate total number of 3-APs in S."""
        cnt = 0
        # Iterate through all pairs (a, b) with a < b
        # If 2b - a is in S, it's a 3-AP (a, b, 2b-a)
        # Note: This counts each 3-AP once because a is the first term.
        # We must ensure 2b - a > b, which is true if b > a.
        # And 2b - a must be in [1, n].
        for a in S:
            for b in S:
                if a < b:
                    c = 2 * b - a
                    if c in S:
                        cnt += 1
        return cnt

    # Initial state: Greedy construction
    A = set()
    for x in range(1, n + 1):
        if is_4ap_free_with_add(A, x):
            A.add(x)
            
    # Simulated Annealing
    start_time = time.time()
    T = 10.0
    T_min = 0.1
    alpha = 0.9998
    
    current_score = total_3aps(A)
    best_score = current_score
    best_A = set(A)
    
    # Run until time limit
    while time.time() - start_time < 55:
        x = random.randint(1, n)
        
        if x in A:
            # Try to remove x
            delta = -count_3aps_delta_remove(A, x)
            # Removing is always feasible
            if delta > 0 or random.random() < math.exp(delta / T):
                A.remove(x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(A)
        else:
            # Try to add x
            if is_4ap_free_with_add(A, x):
                delta = count_3aps_delta_add(A, x)
                if delta > 0 or random.random() < math.exp(delta / T):
                    A.add(x)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_A = set(A)
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    return sorted(best_A)
