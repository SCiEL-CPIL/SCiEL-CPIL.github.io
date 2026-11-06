"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s/90s kill
    
    target = Fraction(5, 31)
    best_solution = None
    best_max_den = float('inf')
    
    def greedy_complete(r, used, max_terms=50):
        """Complete the expansion using greedy method with distinct denominators."""
        if r == 0:
            return []
        if r < 0:
            return None
        values = []
        while r > 0:
            if len(values) >= max_terms:
                return None
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < 1:
                return None
            # Find smallest d' >= d not in used
            while d in used:
                d += 1
            if d > 10000000:
                return None
            new_r = r - Fraction(1, d)
            if new_r < 0:
                return None
            values.append(d)
            used.add(d)
            r = new_r
        return values
    
    # Strategy 1: Try different first denominators with greedy completion
    # The first denominator d1 must satisfy 1/d1 < 5/31 => d1 > 6.2, so d1 >= 7
    for d1 in range(7, 200):
        if time.time() > deadline:
            break
        
        r = target - Fraction(1, d1)
        if r <= 0:
            continue
        
        used = {d1}
        completion = greedy_complete(r, used)
        if completion is not None:
            solution = [d1] + completion
            max_den = max(solution)
            if max_den < best_max_den:
                best_max_den = max_den
                best_solution = solution[:]
    
    # Strategy 2: Try different first TWO denominators with greedy completion
    # This allows us to explore a larger search space
    for d1 in range(7, 50):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        # d2 must satisfy 1/d2 < r1 => d2 > 1/r1
        # Also d2 != d1
        min_d2 = (r1.denominator + r1.numerator - 1) // r1.numerator
        if min_d2 < 1:
            min_d2 = 1
        
        # Try a limited range for d2
        for d2 in range(min_d2, min(min_d2 + 30, 1000)):
            if d2 == d1:
                continue
            if d2 > 10000000:
                break
            
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                if r2 == 0:
                    solution = [d1, d2]
                    max_den = max(solution)
                    if max_den < best_max_den:
                        best_max_den = max_den
                        best_solution = solution[:]
                continue
            
            used = {d1, d2}
            completion = greedy_complete(r2, used)
            if completion is not None:
                solution = [d1, d2] + completion
                max_den = max(solution)
                if max_den < best_max_den:
                    best_max_den = max_den
                    best_solution = solution[:]
    
    # Strategy 3: Local search / refinement
    # Try to improve the best solution found so far by swapping denominators
    if best_solution is not None:
        for _ in range(100):  # Limited number of refinement iterations
            if time.time() > deadline:
                break
            
            current = best_solution[:]
            current_max = max(current)
            
            # Try to reduce the largest denominator
            largest_idx = current.index(current_max)
            largest_val = current[largest_idx]
            
            # Try replacing the largest denominator with something smaller
            # while maintaining the sum
            sum_without_largest = target - Fraction(1, largest_val)
            
            # Try to find a smaller d such that the remaining sum can be completed
            # with denominators not in current (excluding largest)
            other_denoms = set(current) - {largest_val}
            
            # This is a heuristic: try a few candidates
            found_improvement = False
            for candidate in range(largest_val - 1, max(largest_val - 50, 1), -1):
                if candidate in other_denoms:
                    continue
                if candidate < 1:
                    break
                
                new_r = sum_without_largest - Fraction(1, candidate)
                if new_r < 0:
                    continue
                
                new_used = other_denoms | {candidate}
                completion = greedy_complete(new_r, new_used)
                if completion is not None:
                    new_solution = [candidate] + list(other_denoms - {candidate}) + completion
                    # Need to ensure all denominators are distinct
                    if len(set(new_solution)) == len(new_solution):
                        new_max = max(new_solution)
                        if new_max < current_max:
                            best_solution = new_solution
                            best_max_den = new_max
                            found_improvement = True
                            break
            
            if not found_improvement:
                break
    
    # Fallback if no solution found
    if best_solution is None:
        r, values = target - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_solution
# EVOLVE-BLOCK-END
