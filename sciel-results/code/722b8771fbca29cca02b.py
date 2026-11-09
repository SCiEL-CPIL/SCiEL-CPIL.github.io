"""Combined approach: greedy construction with simulated annealing refinement."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 256
    elements = list(range(1, N + 1))
    
    def count_3ap_delta_add(x, A):
        """Returns number of new 3-APs created by adding x to A."""
        delta = 0
        # x is middle: (x-d, x, x+d)
        max_d_mid = min(x - 1, N - x)
        for d in range(1, max_d_mid + 1):
            if (x - d in A) and (x + d in A):
                delta += 1
        
        # x is first: (x, x+d, x+2d)
        max_d_first = (N - x) // 2
        for d in range(1, max_d_first + 1):
            if (x + d in A) and (x + 2 * d in A):
                delta += 1
                
        # x is last: (x-2d, x-d, x)
        max_d_last = x // 2
        for d in range(1, max_d_last + 1):
            if (x - 2 * d in A) and (x - d in A):
                delta += 1
        return delta

    def is_valid_add(x, A):
        """Checks if adding x to A creates a non-constant 4-AP."""
        # 1. x is 1st term: x, x+d, x+2d, x+3d
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d in A) and (x + 2 * d in A) and (x + 3 * d in A):
                return False
        
        # 2. x is 2nd term: x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d in A) and (x + d in A) and (x + 2 * d in A):
                return False
                
        # 3. x is 3rd term: x-2d, x-d, x, x+d
        max_d3 = min(x // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (x - 2 * d in A) and (x - d in A) and (x + d in A):
                return False
                
        # 4. x is 4th term: x-3d, x-2d, x-d, x
        max_d4 = x // 3
        for d in range(1, max_d4 + 1):
            if (x - 3 * d in A) and (x - 2 * d in A) and (x - d in A):
                return False
                
        return True

    def count_3ap_total(A):
        """Counts total 3-APs in A."""
        count = 0
        s = set(A)
        for a in s:
            # a is middle
            max_d_mid = min(a - 1, N - a)
            for d in range(1, max_d_mid + 1):
                if (a - d in s) and (a + d in s):
                    count += 1
            # a is first
            max_d_first = (N - a) // 2
            for d in range(1, max_d_first + 1):
                if (a + d in s) and (a + 2 * d in s):
                    count += 1
            # a is last
            max_d_last = a // 2
            for d in range(1, max_d_last + 1):
                if (a - 2 * d in s) and (a - d in s):
                    count += 1
        return count

    best_set = set()
    best_score = -1
    
    def greedy_construct(order):
        """Build a 4-AP-free set greedily in the given order."""
        A = set()
        for x in order:
            if is_valid_add(x, A):
                A.add(x)
        return A

    def simulated_annealing(A, current_score, T_start, cooling, steps):
        """Refine A using simulated annealing."""
        T = T_start
        for _ in range(steps):
            if time.time() > deadline:
                break
            
            x = random.choice(elements)
            
            if x in A:
                # Try to remove x
                A.remove(x)
                delta = -count_3ap_delta_add(x, A)
                # delta is negative (removing destroys APs)
                if delta > 0 or random.random() < math.exp(delta / T):
                    # Accepted: keep x removed
                    current_score += delta
                else:
                    # Rejected: put x back
                    A.add(x)
            else:
                # Try to add x
                if is_valid_add(x, A):
                    delta = count_3ap_delta_add(x, A)
                    # delta is non-negative
                    if delta > 0 or random.random() < math.exp(delta / T):
                        A.add(x)
                        current_score += delta
                    # else: rejected, do nothing
            
            T *= cooling
        
        return A, current_score

    # Multiple restarts with different strategies
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Choose a random order for greedy construction
        random_order = elements[:]
        random.shuffle(random_order)
        
        # Greedy construction
        A = greedy_construct(random_order)
        current_score = count_3ap_total(A)
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_set = set(A)
        
        # Simulated annealing refinement
        T_start = 3.0
        cooling = 0.9995
        steps = 15000
        
        A, current_score = simulated_annealing(A, current_score, T_start, cooling, steps)
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_set = set(A)
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Final verification: ensure no 4-APs
    def has_4ap(A):
        s = set(A)
        for a in s:
            for b in s:
                if a >= b:
                    continue
                d = b - a
                if d == 0:
                    continue
                # Check if a, b, b+d, b+2d are all in s
                if (b + d in s) and (b + 2 * d in s):
                    return True
        return False

    if has_4ap(best_set):
        # Fallback: use the original greedy approach
        selected = set()
        for x in range(1, 257):
            if all(not all(x - j * d in selected for j in (1, 2, 3)) for d in range(1, (x - 1) // 3 + 1)):
                selected.add(x)
        return sorted(selected)
    
    return sorted(best_set)
