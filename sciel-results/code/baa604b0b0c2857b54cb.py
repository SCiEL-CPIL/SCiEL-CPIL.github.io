import time
import random

def construct():
    deadline = time.time() + 85
    
    def is_mean_of_subset(element, others):
        """Check if element is the arithmetic mean of any subset of others with size >= 2.
        Uses bitset DP."""
        n = len(others)
        if n < 2:
            return False
        
        max_sum = sum(others)
        # For k elements, target sum is k*element
        # Max possible sum with k elements is at most max_sum
        
        # dp[j] is a bitmask where bit s is set if sum s is achievable with exactly j elements
        # We only need to track up to max_sum
        max_target = (n) * element  # max k is n, but we exclude element so max k is n-1... wait
        
        # Actually others has n elements, we pick k from 2 to n
        # target = k * element
        # max sum with k elements <= sum of largest k in others
        
        sorted_others = sorted(others)
        
        # dp[j] = bitmask of achievable sums using exactly j elements
        # Max sum we need to track: max over k of k*element, but also bounded by sum(others)
        max_s = min(max_sum, n * element)
        
        dp = [0] * (n + 1)
        dp[0] = 1  # bit 0 set: sum 0 with 0 elements
        
        for val in others:
            for j in range(n, 0, -1):
                if dp[j-1]:
                    # Shift dp[j-1] by val and OR into dp[j]
                    # But we need to mask to max_s bits
                    dp[j] |= (dp[j-1] << val)
                    # Mask to keep only bits up to max_s
                    dp[j] &= (1 << (max_s + 1)) - 1
        
        for k in range(2, n + 1):
            target = k * element
            if target <= max_s and (dp[k] >> target) & 1:
                return True
        
        return False
    
    def can_add(element, current):
        """Check if element can be added to current set."""
        # Check 1: is element the mean of any subset of current?
        if is_mean_of_subset(element, current):
            return False
        
        # Check 2: for each a in current, does adding element create a violation?
        # a is mean of subset S of (current \ {a}) ∪ {element} with |S| >= 2
        # If element not in S: already checked (a not mean of subset of current \ {a})
        # If element in S: S = T ∪ {element} where T ⊆ current \ {a}, |T| >= 1
        #   sum(T) + element = (|T|+1) * a
        #   sum(T) = (|T|+1)*a - element
        for a in current:
            others_without_a = [x for x in current if x != a]
            n = len(others_without_a)
            if n < 1:
                continue
            
            # For j from 1 to n: check if subset of size j from others_without_a sums to (j+1)*a - element
            # Use bitset DP
            max_s = sum(others_without_a)
            dp = [0] * (n + 1)
            dp[0] = 1
            
            for val in others_without_a:
                for j in range(n, 0, -1):
                    if dp[j-1]:
                        dp[j] |= (dp[j-1] << val)
                        dp[j] &= (1 << (max_s + 1)) - 1
            
            for j in range(1, n + 1):
                target = (j + 1) * a - element
                if 0 <= target <= max_s and (dp[j] >> target) & 1:
                    return False
        
        return True
    
    # Greedy with multiple orderings
    best_set = []
    
    # Deterministic orderings
    orderings = [
        list(range(1, 513)),
        list(range(512, 0, -1)),
        [x for x in range(1, 513) if x % 2 == 1] + [x for x in range(1, 513) if x % 2 == 0],
        [x for x in range(1, 513) if x % 3 != 0] + [x for x in range(1, 513) if x % 3 == 0],
        [x for x in range(1, 513) if x % 5 != 0] + [x for x in range(1, 513) if x % 5 == 0],
    ]
    
    # Add some random orderings
    rng = random.Random(42)
    for _ in range(5):
        order = list(range(1, 513))
        rng.shuffle(order)
        orderings.append(order)
    
    for ordering in orderings:
        if time.time() > deadline - 5:
            break
        
        selected = []
        for x in ordering:
            if time.time() > deadline - 5:
                break
            if len(selected) >= 48:
                break
            if can_add(x, selected):
                selected.append(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
        
        if len(best_set) >= 48:
            break
    
    # Try to add more elements
    if best_set and len(best_set) < 48 and time.time() < deadline - 10:
        improved = best_set[:]
        remaining = [x for x in range(1, 513) if x not in improved]
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if len(improved) >= 48:
                break
            if can_add(x, improved):
                improved.append(x)
        
        if len(improved) > len(best_set):
            best_set = improved
    
    return best_set
