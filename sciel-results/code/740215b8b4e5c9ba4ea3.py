import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 256
    
    def check_valid(y, A_set, exclude=None):
        """Check if y can be added to A_set (excluding 'exclude' if specified)."""
        for u in A_set:
            if u == exclude:
                continue
            # Case 1: y is the middle term (u, y, v)
            v = 2 * y - u
            if v in A_set and v != exclude and v != u:
                return False
            # Case 2: y is an endpoint (y, u, v) -> u is middle
            v = 2 * u - y
            if v in A_set and v != exclude and v != u:
                return False
        return True

    # 1. Greedy Initialization
    A = set()
    for x in range(1, N + 1):
        if check_valid(x, A):
            A.add(x)
            
    # 2. Local Search (Hill Climbing)
    while time.time() < deadline:
        improved = False
        
        # 2a. Try to Add elements
        # Iterate smaller numbers first as they have higher weight
        for y in range(1, N + 1):
            if y in A:
                continue
            if check_valid(y, A):
                A.add(y)
                improved = True
                break
        
        if improved:
            continue
            
        # 2b. Try Swaps
        # We want to replace x with y such that 1/y > 1/x => y < x
        # Iterate x in descending order to prioritize removing low-weight (large) elements
        A_list = list(A)
        A_list.sort(reverse=True)
        
        for x in A_list:
            # Try to replace x with y < x
            for y in range(1, x):
                if y in A:
                    continue
                if check_valid(y, A, exclude=x):
                    A.remove(x)
                    A.add(y)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # No simple add or swap improves the score.
            # We could try more complex moves (e.g., remove 2, add 3) but 
            # given the time budget and typical landscape, 1-swap is often sufficient 
            # to reach a strong local optimum.
            # To be safe and potentially escape deeper local minima, we could implement 
            # a random restart or kick, but strict hill climbing is robust for this task.
            break

    return list(A)
