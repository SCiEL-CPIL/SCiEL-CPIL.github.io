import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    all_numbers = set(range(1, N + 1))
    
    def get_score(A):
        return sum(1.0 / x for x in A)

    def check_valid(A_set, x):
        """
        Check if x can be added to A_set without forming a 3-AP.
        A_set is assumed to be 3-AP free.
        """
        # x forms a 3-AP with a, b in A_set if:
        # 1. x is middle: a + b = 2x => b = 2x - a
        # 2. x is end: a + x = 2b => b = (a + x) / 2
        # 3. x is end: b + x = 2a => a = (b + x) / 2 (same as 2)
        
        for a in A_set:
            # Case 1: x is middle. Check if 2x - a is in A_set.
            # If 2x - a == a, then 2x = 2a => x=a, but x not in A_set.
            # So we just check if (2x - a) is in A_set.
            b1 = 2 * x - a
            if b1 in A_set:
                return False
            
            # Case 2: x is end. Check if (a + x) / 2 is in A_set.
            # Let b2 be the middle term. b2 = (a + x) / 2.
            # If a + x is even, check if b2 is in A_set.
            if (a + x) % 2 == 0:
                b2 = (a + x) // 2
                if b2 in A_set:
                    return False
                    
        return True

    def greedy_init():
        A = set()
        # Greedy by increasing x (highest weight first)
        for x in range(1, N + 1):
            if check_valid(A, x):
                A.add(x)
        return A

    def local_search(A):
        A = set(A)
        current_score = get_score(A)
        improved = True
        # We will alternate between single swaps and double swaps
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # --- Single Swap Phase ---
            # Try to replace x with y where y < x
            # Iterate x in descending order (remove low weight items first)
            A_list = sorted(A, reverse=True)
            for x in A_list:
                if time.time() > deadline - 2.0:
                    break
                # Find best y to replace x
                # y must be < x and not in A
                # We want the smallest y (highest weight)
                A_minus_x = A - {x}
                # Iterate y from 1 to x-1
                for y in range(1, x):
                    if y in A:
                        continue
                    # Check if y can be added to A_minus_x
                    if check_valid(A_minus_x, y):
                        # Perform swap
                        A.remove(x)
                        A.add(y)
                        current_score = get_score(A)
                        improved = True
                        break # Restart scan as set changed
                if improved:
                    break # Break outer loop to restart with new A
            
            if not improved:
                # --- Double Swap Phase (1 out, 2 in) ---
                # Try to replace x with y1, y2 where 1/y1 + 1/y2 > 1/x
                A_list = sorted(A, reverse=True)
                for x in A_list:
                    if time.time() > deadline - 2.0:
                        break
                    
                    A_minus_x = A - {x}
                    # Get available numbers not in A_minus_x
                    # We only care about small numbers for high weight
                    # Let's consider the smallest K available numbers
                    avail = [y for y in range(1, N + 1) if y not in A_minus_x]
                    avail.sort()
                    
                    # Limit search to first K=50 smallest available numbers
                    # This balances speed and quality
                    K = 50 
                    candidates = avail[:K]
                    
                    # Iterate pairs y1, y2 from candidates
                    # We want 1/y1 + 1/y2 > 1/x
                    # Since y1, y2 are small, this is likely for small x?
                    # Actually, if x is large, 1/x is small, so easy to beat.
                    # If x is small, 1/x is large, hard to beat.
                    # We iterate x in descending order, so we start with large x (easy to beat).
                    
                    found_swap = False
                    # To ensure we find a good swap, we iterate y1, y2 in increasing order
                    # This prioritizes smallest y1, y2 (highest weight)
                    for i in range(len(candidates)):
                        y1 = candidates[i]
                        # Optimization: if 1/y1 + 1/y1 (max possible for y2 > y1) <= 1/x, skip?
                        # Actually, just check validity first, score second?
                        # No, we must improve score.
                        # Check score gain:
                        # gain = 1/y1 + 1/y2 - 1/x
                        # If gain <= 0, no need to check validity.
                        # Since y2 > y1, max gain for this y1 is 1/y1 + 1/(y1+1) - 1/x (approx)
                        # Let's just check validity for the first few promising pairs.
                        
                        for j in range(i + 1, len(candidates)):
                            y2 = candidates[j]
                            
                            # Score check
                            if (1.0 / y1 + 1.0 / y2) <= (1.0 / x):
                                # Since y2 is increasing, 1/y2 is decreasing.
                                # So for this y1, no larger y2 will improve score.
                                break
                            
                            # Validity check
                            # Can we add y1 to A_minus_x?
                            if not check_valid(A_minus_x, y1):
                                continue
                            
                            # Can we add y2 to A_minus_x U {y1}?
                            # Note: A_minus_x U {y1} must be 3-AP free.
                            # Since A_minus_x is 3-AP free and y1 is valid, it is.
                            if check_valid(A_minus_x | {y1}, y2):
                                # Perform swap
                                A.remove(x)
                                A.add(y1)
                                A.add(y2)
                                current_score = get_score(A)
                                improved = True
                                found_swap = True
                                break
                        if found_swap:
                            break
                    if improved:
                        break # Restart scan
        
        return A, current_score

    # Main Logic
    best_A = greedy_init()
    best_score = get_score(best_A)
    
    # Run local search
    best_A, best_score = local_search(best_A)
    
    # Random Restarts with Perturbation
    while time.time() < deadline - 5.0:
        # Perturb best_A
        current_A = set(best_A)
        # Remove 1-3 random elements to allow different local optima
        num_remove = random.randint(1, 3)
        for _ in range(num_remove):
            if current_A:
                to_remove = random.choice(list(current_A))
                current_A.remove(to_remove)
        
        # Run local search
        current_A, current_score = local_search(current_A)
        
        if current_score > best_score:
            best_score = current_score
            best_A = current_A

    return list(best_A)
