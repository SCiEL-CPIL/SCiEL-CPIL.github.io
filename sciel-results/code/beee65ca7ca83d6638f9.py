import time
from fractions import Fraction
import random

def construct():
    """
    Constructs a subset of [1, 256] with no non-constant 3-term arithmetic progression
    that maximizes the sum of reciprocals.
    
    Strategy:
    1. Greedy initialization.
    2. Local search with removal and swap moves, followed by greedy re-filling.
    3. Randomized restarts/perturbations to escape local optima.
    """
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute reciprocals for speed
    recip = [0] + [1.0 / i for i in range(1, n + 1)]
    
    def can_add(x, in_set):
        """
        Check if x can be added to the set represented by in_set (list of bools).
        Returns True if valid.
        """
        # Check if x forms a 3-AP with any two existing elements
        # Case 1: x is an endpoint. AP is (a, x, b) or (b, x, a).
        # Then 2*x = a + b. So b = 2*x - a.
        # We need to check if for any a in S, b = 2*x - a is in S and b != x.
        # Case 2: x is the middle. AP is (a, x, b).
        # Then 2*x = a + b. So b = 2*x - a.
        # Wait, if x is middle, a and b are endpoints.
        # The condition "no 3-AP" means there are no distinct a,b,c in S U {x} such that a+b=2c.
        
        # Efficient check:
        # For each a in S:
        #   1. Check if 2*x - a is in S (and != x). This covers x being an endpoint (a, x, 2x-a) or (2x-a, x, a).
        #      Note: if 2x-a == x, then a=x, which is not in S.
        #   2. Check if (a + x) / 2 is in S. This covers x being an endpoint (a, (a+x)/2, x) or (x, (a+x)/2, a).
        #      Wait, if x is an endpoint, the other endpoint is a, middle is m. a+x=2m -> m=(a+x)/2.
        #      So if m is in S, we have a, m, x in S U {x}. Since a, m in S, adding x creates AP.
        
        # So we just need to check:
        # For each a in S:
        #   b1 = 2*x - a. If b1 != x and b1 in S: Invalid. (x is endpoint, a is other endpoint, b1 is middle? No.)
        #   Let's re-verify.
        #   If we add x, and a is in S.
        #   If 2*x - a is in S, let b = 2*x - a. Then a + b = 2*x. So (a, x, b) is an AP.
        #   If a, b are in S, and we add x, then {a, x, b} is a 3-AP.
        #   So if 2*x - a is in S and 2*x - a != x, it's invalid.
        
        #   Also, if (a + x) is even, let m = (a + x) // 2.
        #   If m is in S and m != x and m != a, then (a, m, x) is an AP.
        #   If a, m are in S, and we add x, then {a, m, x} is a 3-AP.
        
        # So:
        for a in range(1, n + 1):
            if in_set[a]:
                # Check if x is an endpoint with a as the other endpoint
                b1 = 2 * x - a
                if b1 != x and 1 <= b1 <= n and in_set[b1]:
                    return False
                
                # Check if x is an endpoint with a as the middle? No, a is in S, x is new.
                # If a is middle, then endpoints are e1, e2. e1 + e2 = 2a.
                # If x is one endpoint, say e1=x, then e2 = 2a - x.
                # This is covered by b1 = 2*x - a? No.
                # If a is middle, AP is (x, a, b). Then x + b = 2a => b = 2a - x.
                # We need to check if b = 2a - x is in S.
                b2 = 2 * a - x
                if b2 != x and 1 <= b2 <= n and in_set[b2]:
                    return False
                
                # Check if x is the middle. AP is (a, x, b).
                # Then a + b = 2x => b = 2x - a.
                # This is the same as b1.
                
                # What if a is an endpoint and x is the other endpoint?
                # AP (a, m, x). Then a + x = 2m => m = (a+x)/2.
                if (a + x) % 2 == 0:
                    m = (a + x) // 2
                    if m != x and m != a and 1 <= m <= n and in_set[m]:
                        return False
        return True

    def greedy_fill(in_set):
        """Add all valid numbers in increasing order."""
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if not in_set[x]:
                    if can_add(x, in_set):
                        in_set[x] = True
                        changed = True
        return in_set

    def get_set(in_set):
        return [i for i in range(1, n + 1) if in_set[i]]

    def get_sum(in_set):
        return sum(recip[i] for i in range(1, n + 1) if in_set[i])

    # Initialize with standard greedy
    in_set = [False] * (n + 1)
    for x in range(1, n + 1):
        if can_add(x, in_set):
            in_set[x] = True
    
    # Greedy fill to ensure maximality (though greedy is usually maximal)
    greedy_fill(in_set)
    
    best_in_set = in_set[:]
    best_sum = get_sum(best_in_set)
    
    # Local Search
    def local_search(current_in_set):
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_sum = get_sum(current_in_set)
            
            # Try removing one element and re-filling
            for rem in range(1, n + 1):
                if not current_in_set[rem]:
                    continue
                if time.time() > deadline - 5.0:
                    break
                
                trial = current_in_set[:]
                trial[rem] = False
                greedy_fill(trial)
                new_sum = get_sum(trial)
                
                if new_sum > current_sum + 1e-10:
                    current_in_set = trial
                    current_sum = new_sum
                    improved = True
                    break
            
            # Try swapping one element with another
            if not improved:
                for a in range(1, n + 1):
                    if not current_in_set[a]:
                        continue
                    if time.time() > deadline - 5.0:
                        break
                    for b in range(a + 1, n + 1):
                        if current_in_set[b]:
                            continue
                        # Try removing a and adding b
                        trial = current_in_set[:]
                        trial[a] = False
                        # Check if b can be added
                        if can_add(b, trial):
                            trial[b] = True
                            greedy_fill(trial)
                            new_sum = get_sum(trial)
                            if new_sum > current_sum + 1e-10:
                                current_in_set = trial
                                current_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
        return current_in_set

    best_in_set = local_search(best_in_set)
    best_sum = get_sum(best_in_set)
    
    # Randomized Restart / Perturbation
    # We will try to escape local optima by perturbing the current best set
    random.seed(42)
    
    while time.time() < deadline - 10.0:
        # Create a perturbed version of the best set
        perturbed = best_in_set[:]
        
        # Remove 1 to 3 random elements
        num_removals = random.randint(1, 3)
        candidates_to_remove = [i for i in range(1, n + 1) if perturbed[i]]
        if not candidates_to_remove:
            break
            
        for _ in range(num_removals):
            if not candidates_to_remove:
                break
            idx = random.randint(0, len(candidates_to_remove) - 1)
            rem_val = candidates_to_remove[idx]
            perturbed[rem_val] = False
            candidates_to_remove.pop(idx)
        
        # Re-greedy fill
        greedy_fill(perturbed)
        
        # Local search on the perturbed set
        improved_set = local_search(perturbed)
        new_sum = get_sum(improved_set)
        
        if new_sum > best_sum + 1e-10:
            best_sum = new_sum
            best_in_set = improved_set

    result = get_set(best_in_set)
    
    # Final validation
    assert len(result) > 0
    # Check 3-AP free
    s_set = set(result)
    for i, a in enumerate(result):
        for b in result[i+1:]:
            if (a + b) % 2 == 0:
                mid = (a + b) // 2
                if mid != a and mid != b and mid in s_set:
                    raise AssertionError("3-AP found")
    
    return result

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a subset of [1, 256] with no non-constant 3-term arithmetic progression
    that maximizes the sum of reciprocals.
    """
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0
    
    recip = [0] + [1.0 / i for i in range(1, n + 1)]
    
    def can_add(x, in_set):
        for a in range(1, n + 1):
            if in_set[a]:
                b1 = 2 * x - a
                if b1 != x and 1 <= b1 <= n and in_set[b1]:
                    return False
                
                b2 = 2 * a - x
                if b2 != x and 1 <= b2 <= n and in_set[b2]:
                    return False
                
                if (a + x) % 2 == 0:
                    m = (a + x) // 2
                    if m != x and m != a and 1 <= m <= n and in_set[m]:
                        return False
        return True

    def greedy_fill(in_set):
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if not in_set[x]:
                    if can_add(x, in_set):
                        in_set[x] = True
                        changed = True
        return in_set

    def get_set(in_set):
        return [i for i in range(1, n + 1) if in_set[i]]

    def get_sum(in_set):
        return sum(recip[i] for i in range(1, n + 1) if in_set[i])

    in_set = [False] * (n + 1)
    for x in range(1, n + 1):
        if can_add(x, in_set):
            in_set[x] = True
    
    greedy_fill(in_set)
    
    best_in_set = in_set[:]
    best_sum = get_sum(best_in_set)
    
    def local_search(current_in_set):
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_sum = get_sum(current_in_set)
            
            for rem in range(1, n + 1):
                if not current_in_set[rem]:
                    continue
                if time.time() > deadline - 5.0:
                    break
                
                trial = current_in_set[:]
                trial[rem] = False
                greedy_fill(trial)
                new_sum = get_sum(trial)
                
                if new_sum > current_sum + 1e-10:
                    current_in_set = trial
                    current_sum = new_sum
                    improved = True
                    break
            
            if not improved:
                for a in range(1, n + 1):
                    if not current_in_set[a]:
                        continue
                    if time.time() > deadline - 5.0:
                        break
                    for b in range(a + 1, n + 1):
                        if current_in_set[b]:
                            continue
                        trial = current_in_set[:]
                        trial[a] = False
                        if can_add(b, trial):
                            trial[b] = True
                            greedy_fill(trial)
                            new_sum = get_sum(trial)
                            if new_sum > current_sum + 1e-10:
                                current_in_set = trial
                                current_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
        return current_in_set

    best_in_set = local_search(best_in_set)
    best_sum = get_sum(best_in_set)
    
    random.seed(42)
    
    while time.time() < deadline - 10.0:
        perturbed = best_in_set[:]
        
        num_removals = random.randint(1, 3)
        candidates_to_remove = [i for i in range(1, n + 1) if perturbed[i]]
        if not candidates_to_remove:
            break
            
        for _ in range(num_removals):
            if not candidates_to_remove:
                break
            idx = random.randint(0, len(candidates_to_remove) - 1)
            rem_val = candidates_to_remove[idx]
            perturbed[rem_val] = False
            candidates_to_remove.pop(idx)
        
        greedy_fill(perturbed)
        
        improved_set = local_search(perturbed)
        new_sum = get_sum(improved_set)
        
        if new_sum > best_sum + 1e-10:
            best_sum = new_sum
            best_in_set = improved_set

    result = get_set(best_in_set)
    
    assert len(result) > 0
    s_set = set(result)
    for i, a in enumerate(result):
        for b in result[i+1:]:
            if (a + b) % 2 == 0:
                mid = (a + b) // 2
                if mid != a and mid != b and mid in s_set:
                    raise AssertionError("3-AP found")
    
    return result
# EVOLVE-BLOCK-END
