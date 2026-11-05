"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort in descending order to prefer larger divisors (minimizes count)
    divisors.sort(reverse=True)
    
    # Remove divisors > target since we can't use them
    divisors = [d for d in divisors if d <= target]
    
    # Precompute suffix sums for pruning: suffix_sum[i] = sum of divisors[i:]
    # But we need sum of top k from index i, so let's use a different approach
    # For pruning, we'll compute the maximum possible sum with k divisors from index i
    
    # Iterative deepening: try to find solution with k terms
    # Start from a lower bound: at least ceil(target / max_divisor)
    max_d = divisors[0] if divisors else 1
    min_k = max(1, (target + max_d - 1) // max_d)
    
    best_solution = None
    
    # Precompute: for each starting index and count k, what's the max sum?
    # This is expensive, so let's just do inline pruning
    
    def dfs(k, start_idx, remaining, current):
        """Try to find k divisors starting from index start_idx that sum to remaining."""
        if time.time() - start_time > DEADLINE:
            return None
        
        if k == 0:
            if remaining == 0:
                return list(current)
            return None
        
        if remaining < 0:
            return None
        
        # Pruning: not enough divisors left
        if start_idx + k > len(divisors):
            return None
        
        # Pruning: even taking the largest k available can't reach remaining
        # Sum of the next k largest divisors from start_idx
        max_possible = 0
        for i in range(start_idx, min(start_idx + k, len(divisors))):
            max_possible += divisors[i]
        if max_possible < remaining:
            return None
        
        # Pruning: even taking the smallest possible (all 1s, but we have distinct divisors)
        # The minimum sum with k distinct divisors from start_idx is sum of the k smallest
        # from start_idx onward, but since sorted descending, the smallest k from start_idx
        # are at the end. Actually, let's skip this for now.
        
        # Try each divisor from start_idx
        for i in range(start_idx, len(divisors)):
            d = divisors[i]
            if d > remaining:
                continue
            
            current.append(d)
            result = dfs(k - 1, i + 1, remaining - d, current)
            if result is not None:
                return result
            current.pop()
        
        return None
    
    # Try increasing k from min_k
    for k in range(min_k, len(divisors) + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        result = dfs(k, 0, target, [])
        if result is not None:
            best_solution = result
            break
    
    # Fallback: if no solution found within deadline, use greedy
    if best_solution is None:
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        if remainder == 0:
            best_solution = chosen
        else:
            # Should not happen for practical numbers, but safety
            raise ValueError("No solution found")
    
    # Verify
    assert len(set(best_solution)) == len(best_solution), "Divisors must be distinct"
    assert all(n % d == 0 for d in best_solution), "All must be divisors of 5040"
    assert sum(best_solution) == target, f"Sum must be {target}, got {sum(best_solution)}"
    assert len(best_solution) <= 256, "At most 256 divisors"
    
    return best_solution
# EVOLVE-BLOCK-END
