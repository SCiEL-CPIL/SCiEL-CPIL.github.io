from fractions import Fraction

def construct():
    target = Fraction(1, 1)
    
    # Try k from 8 upwards
    for k in range(8, 13):
        solution = find_solution(k)
        if solution:
            return solution
            
    # Fallback
    # If no solution found up to 12, return a known valid one or raise
    # But a solution must exist.
    # The seed program implies one exists.
    # Let's try k up to 15 just in case.
    for k in range(13, 16):
        solution = find_solution(k)
        if solution:
            return solution
            
    raise RuntimeError("No solution found")

def find_solution(k):
    target = Fraction(1, 1)
    
    def dfs(start_d, remaining_sum, remaining_terms, current_list):
        if remaining_terms == 0:
            if remaining_sum == 0:
                return current_list
            return None
        
        if remaining_sum < 0:
            return None
            
        # Max possible sum with remaining_terms terms starting from start_d
        # If this max sum < remaining_sum, we can't reach the target
        # We compute this sum.
        max_sum = 0
        for i in range(remaining_terms):
            max_sum += Fraction(1, start_d + i)
            # Early exit if max_sum already >= remaining_sum? 
            # No, we need to know if it's < remaining_sum to break the outer loop.
            # But inside the loop, we just need the total.
            
        if max_sum < remaining_sum:
            return None
            
        # Determine min_d
        # 1/d <= remaining_sum => d >= 1/remaining_sum
        min_d = max(start_d, int(1 / remaining_sum) + 1)
        
        d = min_d
        while True:
            # Check if max possible sum with d is sufficient
            # We can reuse the logic: sum(1/(d+i) for i in range(remaining_terms)) >= remaining_sum
            # If not, d is too large, break.
            
            # Optimization: 
            # If 1/d + 1/(d+1) + ... + 1/(d+k-1) < remaining_sum, break.
            # We can approximate or compute.
            # Computing is safe.
            curr_max_sum = 0
            for i in range(remaining_terms):
                curr_max_sum += Fraction(1, d + i)
                if curr_max_sum > remaining_sum:
                    break
            if curr_max_sum < remaining_sum:
                break
            
            term = Fraction(1, d)
            if term > remaining_sum:
                # This implies d < 1/remaining_sum, which is handled by min_d
                # But if floating point errors or logic gaps, skip
                d += 1
                continue
                
            new_remaining = remaining_sum - term
            res = dfs(d + 1, new_remaining, remaining_terms - 1, current_list + [d])
            if res:
                return res
            
            d += 1
            
        return None

    return dfs(5, target, k, [])
