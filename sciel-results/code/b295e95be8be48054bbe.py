"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    def is_feasible(selected_set, candidate):
        """Check if adding candidate to selected_set maintains the property."""
        # For every a in selected_set with a < candidate, and for every b,c in selected_set
        # with a < b <= c, we need a not to divide (b+c).
        # Also need to check triples where candidate is b or c.
        
        # Case 1: candidate is the largest (c), so a < b <= candidate
        # Need: for all a in S, b in S with a < b <= candidate, a not divide (b + candidate)
        for a in selected_set:
            if a >= candidate:
                continue
            for b in selected_set:
                if a < b <= candidate:
                    if (b + candidate) % a == 0:
                        return False
        
        # Case 2: candidate is in the middle (b), so a < candidate <= c
        # Need: for all a in S with a < candidate, c in S with candidate <= c, a not divide (candidate + c)
        for a in selected_set:
            if a >= candidate:
                continue
            for c in selected_set:
                if candidate <= c:
                    if (candidate + c) % a == 0:
                        return False
        
        # Case 3: candidate is the smallest (a), so candidate < b <= c
        # Need: for all b, c in S with candidate < b <= c, candidate not divide (b + c)
        for b in selected_set:
            if b <= candidate:
                continue
            for c in selected_set:
                if b <= c:
                    if (b + c) % candidate == 0:
                        return False
        
        return True
    
    def greedy_with_order(order):
        """Greedy algorithm using a specific order of candidates."""
        selected = set()
        for num in order:
            if is_feasible(selected, num):
                selected.add(num)
        return selected
    
    # Strategy 1: Ascending order
    order1 = list(range(1, n+1))
    set1 = greedy_with_order(order1)
    
    # Strategy 2: Descending order  
    order2 = list(range(n, 0, -1))
    set2 = greedy_with_order(order2)
    
    # Strategy 3: Weighted order - prioritize numbers that are "less conflicting"
    # A number x is more conflicting if it has many divisors or if many numbers in [1,n] 
    # would create conflicts with it. We approximate by counting how many pairs (b,c) with b<=c
    # have b+c divisible by x, or how many a divide x.
    
    def conflict_score(x):
        """Score how much x conflicts with others. Lower is better (less conflict)."""
        score = 0
        # Count how many a in [1,x-1] divide x (these a would conflict if x is b or c)
        for a in range(1, x):
            if x % a == 0:
                score += 1
        # Count how many pairs (b,c) with 1<=b<=c<=n, b<=c, such that x divides (b+c)
        # This is expensive, so approximate: for each b in [x+1, n], count c in [b, n] where (b+c)%x==0
        # But this is O(n^2/x) which could be large. Let's use a simpler heuristic.
        # Numbers with many divisors are more likely to cause conflicts.
        div_count = 0
        for d in range(1, int(x**0.5)+1):
            if x % d == 0:
                div_count += 1
                if d != x//d:
                    div_count += 1
        score += div_count * 2
        return score
    
    order3 = sorted(range(1, n+1), key=lambda x: (conflict_score(x), x))
    set3 = greedy_with_order(order3)
    
    # Strategy 4: Start with large numbers, then add small ones
    # Large numbers are less likely to divide sums of other large numbers
    order4 = list(range(n, 128, -1)) + list(range(1, 129))
    set4 = greedy_with_order(order4)
    
    # Strategy 5: Try multiple threshold-based approaches
    best_set = max([set1, set2, set3, set4], key=len)
    best_size = len(best_set)
    
    # Try to improve by local search: remove one element, try to add a different one
    improved = set(best_set)
    improved_list = sorted(improved)
    
    for _ in range(50):  # Limit iterations for time
        progress = False
        for i in range(len(improved_list)):
            removed = improved_list[i]
            remaining = improved - {removed}
            
            # Try to add any number not in remaining
            for candidate in range(1, n+1):
                if candidate in remaining:
                    continue
                if is_feasible(remaining, candidate):
                    new_set = remaining | {candidate}
                    if len(new_set) > len(improved):
                        improved = new_set
                        improved_list = sorted(improved)
                        progress = True
                        break
            if progress:
                break
        
        if not progress:
            break
    
    return list(improved)
# EVOLVE-BLOCK-END
