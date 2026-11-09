"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    def check_feasible_fast(selected_list, candidate):
        """Check if adding candidate to selected_list maintains the property."""
        sel = sorted(selected_list + [candidate])
        k = len(sel)
        for i in range(k):
            a = sel[i]
            for j in range(i+1, k):
                b = sel[j]
                for k2 in range(j, k):
                    c = sel[k2]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Strategy 1: Greedy from small to large
    selected1 = []
    for num in range(1, n+1):
        if check_feasible_fast(selected1, num):
            selected1.append(num)
    
    # Strategy 2: Start with large tail, try to add small
    selected2 = list(range(192, n+1))
    for num in range(1, 192):
        if check_feasible_fast(selected2, num):
            selected2.append(num)
            selected2.sort()
    
    # Strategy 3: Greedy prioritizing numbers with fewer divisors
    def divisor_count(x):
        count = 0
        for d in range(1, int(x**0.5)+1):
            if x % d == 0:
                count += 1
                if d != x//d:
                    count += 1
        return count
    
    nums_sorted = sorted(range(1, n+1), key=lambda x: (divisor_count(x), x))
    selected3 = []
    for num in nums_sorted:
        if check_feasible_fast(selected3, num):
            selected3.append(num)
    
    # Strategy 4: Try starting from different thresholds
    best_result = max([selected1, selected2, selected3], key=len)
    
    # Try to improve best_result by attempting swaps
    best_set = set(best_result)
    improved = list(best_result)
    improved.sort()
    
    for i in range(len(improved)):
        for j in range(1, n+1):
            if j not in best_set:
                # Try swapping improved[i] with j
                new_set = improved.copy()
                new_set[i] = j
                new_set.sort()
                if check_feasible_fast(new_set[:i] + new_set[i+1:], j):
                    # This is getting complex, skip for now
                    pass
    
    return improved
# EVOLVE-BLOCK-END
