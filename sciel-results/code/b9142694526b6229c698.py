import time
import itertools
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Max possible sum: 30 elements * 256 = 7680
    MAX_SUM = 30 * 256
    
    # Precompute divisibility relationships for all possible sums up to MAX_SUM
    # divisors[s] will be a set of all d < s such that s % d == 0
    # We actually need to check if a divides b.
    # Let's create a list of sets: `multiples[d]` contains all m such that m % d == 0.
    # Or simpler: `divides[a][b]` is too big.
    # Let's use a boolean matrix or sets.
    # Since MAX_SUM is ~7680, a set of divisors for each number is fine.
    # divisors_of[n] = set of d in 1..n-1 such that n % d == 0
    
    divisors_of = [set() for _ in range(MAX_SUM + 1)]
    for d in range(1, MAX_SUM + 1):
        for m in range(2*d, MAX_SUM + 1, d):
            divisors_of[m].add(d)
    
    def is_valid_incremental(current_sums_set, new_element):
        """
        Check if adding new_element to a set represented by current_sums_set is valid.
        current_sums_set is a set of all valid subset sums of the current array.
        Returns True if valid, False otherwise.
        """
        # Generate new sums
        # The new sums are: new_element, and s + new_element for all s in current_sums_set
        # Note: The empty set sum is 0, but we only consider non-empty subsets.
        # However, when we add a new element x, the new subsets are {x} and {x} U S for S in subsets of old array.
        # The sums of subsets of old array are in current_sums_set.
        # So new sums = {x} U {s + x for s in current_sums_set}
        
        new_sums = set()
        new_sums.add(new_element)
        for s in current_sums_set:
            new_sums.add(s + new_element)
        
        # 1. Check for duplicates with existing sums
        # If any new sum is in current_sums_set, invalid (two subsets same sum)
        if new_sums & current_sums_set:
            return False
        
        # 2. Check divisibility between new sums and existing sums
        # For each new sum ns:
        #   - Does ns divide any existing sum es? (es % ns == 0)
        #   - Does any existing sum es divide ns? (ns % es == 0)
        
        # To make this fast, we can iterate over new_sums and check against divisors_of[ns]
        # and check if any es is in divisors_of[ns].
        # Also check if ns is a divisor of any es. This is harder.
        # We can precompute `multiples_of[d]` = set of m such that m % d == 0.
        # But we already have divisors_of.
        # If es divides ns, then es is in divisors_of[ns].
        # If ns divides es, then ns is in divisors_of[es].
        
        # Let's check:
        # A) Existing divides New:
        #    For each ns in new_sums:
        #       If any es in current_sums_set is in divisors_of[ns], then invalid.
        #       Intersection: current_sums_set & divisors_of[ns]
        
        # B) New divides Existing:
        #    For each es in current_sums_set:
        #       If any ns in new_sums is in divisors_of[es], then invalid.
        #       This is equivalent to: For each ns in new_sums, check if ns is in divisors_of[es] for any es.
        #       Or: Union of divisors_of[es] for all es in current_sums_set. Let's call it AllDivisorsOfExisting.
        #       If new_sums & AllDivisorsOfExisting, invalid.
        
        # Let's precompute AllDivisorsOfExisting for the current state?
        # No, current_sums_set changes.
        # But we can just iterate.
        
        # Optimization: 
        # Check A: 
        for ns in new_sums:
            if current_sums_set & divisors_of[ns]:
                return False
                
        # Check B:
        # We need to know if any ns divides any es.
        # This is equivalent to: Is there any pair (ns, es) such that es % ns == 0?
        # We can iterate over new_sums and check if ns is a divisor of any es.
        # Since new_sums size is ~2^(n-1), and current_sums_set size is ~2^(n-1), 
        # direct cross product is too slow for large n.
        
        # Alternative for B:
        # For each es in current_sums_set, check if any ns in new_sums is in divisors_of[es].
        # We can precompute a set of all divisors of all existing sums?
        # Let's maintain `all_existing_divisors` as a set.
        # When we add a sum s, we add all divisors of s to `all_existing_divisors`.
        # Then check B is: if new_sums & all_existing_divisors: return False.
        
        # So the function signature should accept `all_existing_divisors` as well.
        
        # Wait, the function `is_valid_incremental` currently only takes `current_sums_set`.
        # I will modify the outer loop to maintain `all_existing_divisors`.
        
        # For now, let's implement the check inside the main loop where we have access to state.
        # I will restructure the code to keep state variables.
        
        # Actually, I can just return a tuple (valid, new_sums_set, new_all_divisors) or similar.
        # Let's just do the check here assuming we pass the necessary sets.
        # I'll rewrite the helper to take (current_sums, all_divs_of_current, new_elem)
        
        pass # Placeholder, logic moved to main loop for efficiency

    def check_addition(current_sums, all_divs_of_current, new_elem):
        """
        Returns (is_valid, new_sums_set, new_all_divs_set)
        """
        new_sums = set()
        new_sums.add(new_elem)
        for s in current_sums:
            new_sums.add(s + new_elem)
            
        # Check duplicates
        if new_sums & current_sums:
            return False, None, None
            
        # Check: Existing divides New
        # For each ns in new_sums, if any es in current_sums is a divisor of ns, invalid.
        # This is: current_sums & divisors_of[ns]
        for ns in new_sums:
            if current_sums & divisors_of[ns]:
                return False, None, None
                
        # Check: New divides Existing
        # For each ns in new_sums, if ns is a divisor of any es in current_sums, invalid.
        # This is: ns is in all_divs_of_current
        if new_sums & all_divs_of_current:
            return False, None, None
            
        # Check: New sums divide each other
        # We need to ensure no two distinct sums in new_sums divide each other.
        # Also, new_sums are distinct from current_sums (checked above).
        # So we just check within new_sums.
        # new_sums size is 2^(n-1) + 1.
        # For small n this is fine.
        # Sort new_sums to make it easier?
        # Just check pairs? Or use divisors_of.
        # For each ns in new_sums, check if any other ns2 in new_sums is in divisors_of[ns].
        # To avoid O(N^2), we can iterate.
        # Since new_sums is a set, we can do:
        # for ns in new_sums:
        #    if (new_sums - {ns}) & divisors_of[ns]:
        #        return False
        # This is O(|new_sums| * |divisors|).
        
        for ns in new_sums:
            # Check if any other new sum divides ns
            if (new_sums - {ns}) & divisors_of[ns]:
                return False, None, None
            # Note: We don't need to check if ns divides another new sum explicitly 
            # if we iterate all ns and check divisors_of[ns] against the set.
            # If ns1 divides ns2, then ns1 is in divisors_of[ns2]. 
            # When we check ns2, we find ns1.
            # When we check ns1, we check if anything in new_sums divides ns1. 
            # ns2 does not divide ns1 (since ns2 > ns1).
            # So checking divisors_of[ns] against the set covers both directions?
            # Yes. If a|b, a is in divisors_of[b].
            
        # If all checks pass, compute new state
        new_all_divs = set(all_divs_of_current)
        for ns in new_sums:
            new_all_divs.update(divisors_of[ns])
            
        return True, new_sums, new_all_divs

    best_size = 0
    best_set = []
    
    # Random search with incremental validation
    # We try to build sets of size up to 10-15?
    # Max elements is 30, but finding such a set is hard.
    # Let's try random restarts.
    
    num_restarts = 5000 # High number, but each step is fast
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random seed for reproducibility within run, but varied across restarts
        random.seed(r)
        
        # Start with a random small set
        # Pick 3 random distinct numbers
        # Ensure they are valid initially
        arr = []
        current_sums = set()
        all_divs = set()
        
        # Greedily add 3 elements
        for _ in range(3):
            # Pick a random candidate
            # Bias towards larger numbers? Or random?
            # Larger numbers have fewer divisors, so easier to satisfy "no divisibility".
            # But subset sums grow.
            # Let's pick from 1..256.
            cand = random.randint(1, 256)
            if cand in arr:
                continue
            valid, ns, nd = check_addition(current_sums, all_divs, cand)
            if valid:
                arr.append(cand)
                current_sums = ns
                all_divs = nd
        
        if len(arr) < 3:
            continue
            
        # Try to extend
        while time.time() < deadline - 0.5:
            current_size = len(arr)
            if current_size > best_size:
                best_size = current_size
                best_set = list(arr)
                
            # Try to add a new element
            # Candidate selection strategy:
            # Try a few random candidates
            found = False
            # Limit candidates per step to avoid hanging on one bad spot
            for _ in range(100):
                cand = random.randint(1, 256)
                if cand in arr:
                    continue
                valid, ns, nd = check_addition(current_sums, all_divs, cand)
                if valid:
                    arr.append(cand)
                    current_sums = ns
                    all_divs = nd
                    found = True
                    break
                    
            if not found:
                break
                
    # Final verification of the best set found (just in case)
    # We should return the best_set
    # Convert to list of ints
    return [int(x) for x in best_set]

if __name__ == "__main__":
    construct()
