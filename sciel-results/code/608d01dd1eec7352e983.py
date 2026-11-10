import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    def find_solution_for_k(k):
        target = Fraction(1, 1)
        solution = []
        
        def dfs(start, remaining, count):
            if count == 0:
                return remaining == 0
            if remaining <= 0:
                return False
            
            # Prune 1: 1/d must be <= remaining
            # d >= 1/remaining
            min_d = max(start, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            
            # Prune 2: Max possible sum of 'count' terms starting from 'd'
            # is sum(1/i for i in range(d, d+count))
            # If this sum < remaining, then we can't reach 'remaining'
            # with this 'd' or any larger 'd' (since sum decreases as d increases)
            
            # Calculate sum of 1/d + 1/(d+1) + ... + 1/(d+count-1)
            # We can approximate or compute exactly.
            # For small count (9-15), exact is fast.
            
            max_possible_sum = Fraction(0, 1)
            for i in range(count):
                max_possible_sum += Fraction(1, min_d + i)
                
            if max_possible_sum < remaining:
                return False
                
            for d in range(min_d, 10000000):
                if time.time() > deadline:
                    return False # Timeout signal
                if Fraction(1, d) > remaining:
                    break
                
                # Check if remaining terms can sum to remaining - 1/d
                # The max sum of remaining 'count-1' terms starting from d+1 is:
                # sum(1/i for i in range(d+1, d+1+count-1))
                # If this is < remaining - 1/d, we can break?
                # No, if this max sum is too small, then this d is too small?
                # Wait. If d increases, 1/d decreases, so remaining increases.
                # But the starting point for next terms (d+1) also increases, so max sum decreases.
                # This interaction is tricky.
                
                # Let's just recurse.
                if dfs(d + 1, remaining - Fraction(1, d), count - 1):
                    solution.append(d)
                    return True
            return False
            
        if dfs(5, target, k):
            return solution
        return None

    # Try k from 9 upwards
    for k in range(9, 25):
        if time.time() > deadline - 5: # Reserve 5s for fallback
            break
        sol = find_solution_for_k(k)
        if sol:
            return sol
            
    # Fallback: Use the seed program's logic to get a valid solution
    # This ensures we don't return None or timeout.
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no solution found")
