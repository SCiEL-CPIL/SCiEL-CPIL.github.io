"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def construct():
    """
    Find a strictly increasing sequence of integers in [1,256] such that
    all nonempty consecutive block sums are distinct. Maximize length.
    
    Uses a greedy approach with lookahead/backtracking-lite:
    We build the sequence left to right. At each step, we try candidates
    from high to low (or low to high) and check if adding the candidate
    creates any duplicate consecutive block sums.
    
    The key insight: consecutive block sums of a sequence a[0..n-1] are
    all sums a[i] + ... + a[j] for 0 <= i <= j < n.
    When we append a new element x to the end, the new block sums are:
    x, a[n-1]+x, a[n-2]+a[n-1]+x, ..., sum(a)+x
    i.e., x + suffix_sums where suffix_sums are the suffix sums of the current sequence.
    
    We maintain the set of all existing block sums. For each new candidate x,
    we compute the new sums and check for conflicts.
    
    To maximize length, we want to use as many numbers as possible from [1,256].
    A greedy strategy: try to add the smallest available number first to keep
    sums small, allowing more room. But sometimes skipping small numbers allows
    fitting more total numbers.
    
    Strategy: Use a priority-based greedy where we try candidates in increasing
    order, but if a candidate leads to a dead end (no further additions possible
    and we haven't used all 256), we might need backtracking. However, full
    backtracking is expensive.
    
    Let's try: greedy with increasing order, but if stuck, try a different approach.
    
    Actually, let's think about the maximum possible length. The number of consecutive
    blocks in a sequence of length n is n(n+1)/2. These must all be distinct sums.
    The minimum possible sum is 1 (single element 1) and maximum is at most 256*n.
    But more precisely, the sums range from min_element to max possible block sum.
    
    For n elements, we need n(n+1)/2 distinct sums. The sums are positive integers.
    The smallest possible maximum sum for a given n with elements in [1,256]...
    
    Let's just do a smart greedy: try adding the smallest unused number that doesn't
    conflict. If that fails to extend further, try the next smallest, etc. This is
    essentially what the original does but in increasing order.
    
    Wait, the original does increasing order (for x in range(1, 257)). Let me check
    what length it produces.
    
    Let me implement a more sophisticated approach:
    1. Start with empty sequence.
    2. At each step, consider all unused numbers from 1 to 256.
    3. For each candidate, check if it can be appended without conflict.
    4. Among valid candidates, pick the one that leaves the most "room" for future
       additions. A good heuristic: pick the smallest valid candidate, OR pick the
       one that minimizes the maximum new block sum, OR use a look-ahead.
    
    For simplicity and speed, let's try multiple greedy strategies and keep the best:
    - Strategy 1: smallest valid first
    - Strategy 2: largest valid first  
    - Strategy 3: try all, pick based on heuristic
    
    Given 60s time limit, let's do a bounded backtracking or multiple greedy passes.
    """
    
    def try_greedy(order):
        """
        order: 'asc' or 'desc' - in which order to try candidates
        """
        values = []
        sums = set()
        used = set()
        
        for x in range(1, 257):
            if order == 'asc':
                candidates = [x] if x not in used else []
            else:
                candidates = [x] if x not in used else []
            
            # Actually, let's restructure: at each step, try all unused candidates
            pass
        
        # Let's do it properly with a loop that tries to extend
        values = []
        sums = set()
        used = set()
        
        while True:
            best_candidate = None
            best_score = None
            
            # Determine candidate order
            if order == 'asc':
                cand_range = range(1, 257)
            else:
                cand_range = range(256, 0, -1)
            
            for x in cand_range:
                if x in used:
                    continue
                # Check if x can be appended
                # New sums: x + suffix sums of current values
                if not values:
                    new_sums = {x}
                else:
                    new_sums = set()
                    s = x
                    new_sums.add(s)
                    for a in reversed(values):
                        s += a
                        new_sums.add(s)
                
                if not new_sums.intersection(sums):
                    # Valid candidate
                    if best_candidate is None:
                        best_candidate = x
                        best_score = 0
                    else:
                        # Heuristic: prefer smaller max new sum to leave room
                        max_new = max(new_sums)
                        if best_score is None or max_new < best_score:
                            best_candidate = x
                            best_score = max_new
            
            if best_candidate is None:
                break
            
            # Append best_candidate
            values.append(best_candidate)
            used.add(best_candidate)
            if not values[:-1]:
                sums.update({best_candidate})
            else:
                s = best_candidate
                sums.add(s)
                for a in reversed(values[:-1]):
                    s += a
                    sums.add(s)
        
        return values
    
    # Try both orders and keep the best
    result_asc = try_greedy('asc')
    result_desc = try_greedy('desc')
    
    # Also try a mixed approach: start with asc, but if stuck early, restart with different strategy
    # For now, return the longer one
    if len(result_asc) >= len(result_desc):
        return result_asc
    else:
        return result_desc

# EVOLVE-BLOCK-END
