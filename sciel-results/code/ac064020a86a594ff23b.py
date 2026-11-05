"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Find 1..256 distinct denominators in [1,10000000] summing reciprocally to 5/31,
    minimizing the largest denominator.
    
    Strategy: Exhaustive search over k-term Egyptian fractions (k=3,4,5,...)
    using tight mathematical bounds to prune the search space. For each k,
    we search for solutions with a < b < ... < z, computing the last term
    from the remaining fraction. We check k=3 first (fast), then k=4, etc.
    We also try to find solutions where the maximum denominator is smaller
    than what k=3 gives, by checking if smaller M values are feasible.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    best_solution = None
    best_max = float('inf')
    
    def check_solution(denoms):
        """Verify a solution and update best if valid."""
        nonlocal best_solution, best_max
        if len(denoms) == 0 or len(denoms) > 256:
            return
        if len(set(denoms)) != len(denoms):
            return
        if any(d < 1 or d > 10000000 for d in denoms):
            return
        s = sum(Fraction(1, d) for d in denoms)
        if s != target:
            return
        m = max(denoms)
        if m < best_max:
            best_max = m
            best_solution = sorted(denoms)
    
    # Search for k-term solutions
    # For k terms: 1/a1 + 1/a2 + ... + 1/ak = 5/31 with a1 < a2 < ... < ak
    # We fix the first k-1 terms and compute ak from the remaining fraction.
    
    def search_k_terms(k, a_min=1, a_max=10000000):
        """
        Search for k-term Egyptian fraction solutions.
        Uses recursive search with tight bounds.
        """
        nonlocal best_solution, best_max, deadline
        
        def dfs(pos, start_val, remaining, chosen):
            nonlocal best_solution, best_max, deadline
            
            if time.time() > deadline:
                return
            
            if pos == k - 1:
                # Last term: 1/ak = remaining, so ak = 1/remaining
                # remaining must be positive and 1/remaining must be integer > start_val
                if remaining <= 0:
                    return
                if remaining.numerator != 1:
                    return
                ak = remaining.denominator
                if ak <= start_val:
                    return
                if ak > 10000000:
                    return
                if ak >= best_max:
                    return
                sol = chosen + [ak]
                check_solution(sol)
                return
            
            # Determine range for a[pos]
            # 1/a[pos] < remaining (since other terms are positive)
            # So a[pos] > 1/remaining
            # Also, the sum of the remaining (k - pos) terms is 'remaining'
            # The maximum possible sum of (k-pos) terms starting from a[pos]+1 is:
            #   sum(1/j for j in range(a[pos]+1, a[pos]+(k-pos)+1))
            # This is an upper bound on what we can achieve.
            # The minimum is when we take the largest possible denominators, but that's not useful.
            
            # Lower bound for a[pos]:
            # 1/a[pos] < remaining => a[pos] > 1/remaining
            # Also a[pos] > start_val
            lb = max(start_val + 1, remaining.denominator // remaining.numerator + 1)
            
            # Upper bound for a[pos]:
            # We need (k - pos) terms starting from a[pos] to sum to 'remaining'
            # The maximum sum of (k-pos) reciprocals starting from a[pos] is:
            #   1/a[pos] + 1/(a[pos]+1) + ... + 1/(a[pos] + k - pos - 1)
            # If this max sum < remaining, then a[pos] is too small... wait, no.
            # Actually, if a[pos] is too large, the sum will be too small.
            # We need: sum of (k-pos) terms >= remaining
            # The maximum possible sum with a[pos] as the first term is:
            #   S_max = sum(1/j for j in range(a[pos], a[pos] + (k-pos)))
            # We need S_max >= remaining
            # Also, 1/a[pos] < remaining (strictly, since other terms are positive)
            # So a[pos] > 1/remaining
            
            # Upper bound: a[pos] must be small enough that we can still reach 'remaining'
            # with (k-pos) terms. The minimum sum of (k-pos) terms starting from a[pos]
            # is achieved when denominators are as large as possible, but we have no upper
            # bound on denominators other than 10000000. However, since we're minimizing
            # the max denominator, and we already have best_max, we can use that.
            
            # A simpler upper bound: since 1/a[pos] + 1/(a[pos]+1) + ... + 1/(a[pos]+k-pos-1) >= remaining
            # The LHS decreases as a[pos] increases. So we can find the upper bound by
            # checking when the max possible sum drops below remaining.
            
            # Also, since a[pos] < a[pos+1] < ... and we want to minimize the max,
            # a[pos] should not be too large.
            
            # Practical upper bound: a[pos] < remaining.denominator / remaining.numerator * 2
            # (rough estimate)
            ub = max(lb, (remaining.denominator * 2) // remaining.numerator)
            
            # Cap the range to avoid too many iterations
            # If the range is too large, we might need to be smarter
            # But for small k, the ranges should be manageable
            
            for a_val in range(lb, ub + 1):
                if time.time() > deadline:
                    return
                
                # Check if a_val is valid
                term = Fraction(1, a_val)
                if term >= remaining:
                    continue
                
                new_remaining = remaining - term
                if new_remaining <= 0:
                    continue
                
                # Check if it's possible to achieve new_remaining with (k - pos - 1) more terms
                # starting from a_val + 1
                # Max possible sum: sum(1/j for j in range(a_val+1, a_val+1+(k-pos-1)))
                if k - pos - 1 > 0:
                    max_sum = sum(Fraction(1, j) for j in range(a_val + 1, a_val + 1 + (k - pos - 1)))
                    if max_sum < new_remaining:
                        continue
                
                chosen.append(a_val)
                dfs(pos + 1, a_val, new_remaining, chosen)
                chosen.pop()
        
        dfs(0, a_min - 1, target, [])
    
    # Try k=3 first (fastest)
    search_k_terms(3)
    
    # Try k=4
    if time.time() < deadline:
        search_k_terms(4)
    
    # Try k=5
    if time.time() < deadline - 5:
        search_k_terms(5)
    
    # Try k=6
    if time.time() < deadline - 10:
        search_k_terms(6)
    
    if best_solution is not None:
        return best_solution
    else:
        # Fallback
        return [8, 31, 248]

# EVOLVE-BLOCK-END
