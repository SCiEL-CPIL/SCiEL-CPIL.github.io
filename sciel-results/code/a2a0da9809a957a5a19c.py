import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    
    def is_valid_set(S):
        """Check if S satisfies the constraint: for every a<b<=c in S, a does not divide b+c."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                for k in range(j, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def greedy_build(order, S=None, sums=None):
        """Greedily build set following given order."""
        if S is None:
            S = set()
        if sums is None:
            sums = set()
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Check if x divides any existing sum
            valid = True
            for s in sums:
                if s % x == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Also need to check: for all existing a in S where a < x,
            # a must not divide x + y for any y in S with y >= x... 
            # Actually the constraint is: for a < b <= c, a not divide b+c
            # If we add x, we need to check:
            # 1. For all existing a < x: a must not divide x + y for any y in S with y >= x (but x is new, so y could be x itself or existing elements >= x)
            #    Wait, if a < x and we have b=x, c=y where y>=x, then a must not divide x+y
            #    Also b=y, c=x where y>=x: same thing
            #    And b=x, c=x: a must not divide 2x
            # 2. For all existing a where a < some existing b <= c, those are already checked
            # 3. New triples involving x as b or c:
            #    - a < x = b <= c: a in S, a < x, c in S with c >= x. Check a | (x+c)
            #    - a < b <= x = c: a,b in S, a < b <= x. Check a | (b+x)
            #    - a < x = b = c: a in S, a < x. Check a | (2x)
            
            # Let's do a full check for safety since S is small
            new_S = S | {x}
            if is_valid_set(new_S):
                S = new_S
                # Update sums
                sums.add(2 * x)
                for y in S:
                    if y != x:
                        sums.add(x + y)
        
        return S, sums
    
    def local_improve(S, max_time=5.0):
        """Try to improve S by swapping elements."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_sorted = sorted(S, reverse=True)
            n_S = len(S)
            
            for i, large in enumerate(S_sorted):
                if time.time() > deadline:
                    break
                
                # Try replacing 'large' with some smaller x not in S
                S_without_large = S - {large}
                
                # Precompute sums for S_without_large
                S_list_wo = sorted(S_without_large)
                sums_wo_large = set()
                for j in range(len(S_list_wo)):
                    for k in range(j, len(S_list_wo)):
                        sums_wo_large.add(S_list_wo[j] + S_list_wo[k])
                
                # Try candidate replacements
                for x in range(1, large):
                    if x in S:
                        continue
                    
                    # Quick check: x must not divide any sum in sums_wo_large
                    valid = True
                    for s in sums_wo_large:
                        if s % x == 0:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Full validation of new set
                    new_set = S_without_large | {x}
                    if is_valid_set(new_set):
                        S = new_set
                        improved = True
                        break
        
        return S
    
    best_S = set()
    
    # Strategy 1: Greedy from largest to smallest
    S1, _ = greedy_build(range(n, 0, -1))
    S1 = local_improve(S1)
    if len(S1) > len(best_S):
        best_S = S1
    
    # Strategy 2: Greedy from smallest to largest
    if time.time() < deadline:
        S2, _ = greedy_build(range(1, n + 1))
        S2 = local_improve(S2)
        if len(S2) > len(best_S):
            best_S = S2
    
    # Strategy 3: Multiple random restarts
    rng = random.Random(42)
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Create a random ordering with bias toward larger elements
        order = list(range(1, n + 1))
        rng.shuffle(order)
        
        S_r, _ = greedy_build(order)
        S_r = local_improve(S_r, max_time=3.0)
        
        if len(S_r) > len(best_S):
            best_S = S_r
    
    # Final: try to add more elements to best_S
    if time.time() < deadline - 5:
        for x in range(1, n + 1):
            if x in best_S:
                continue
            new_S = best_S | {x}
            if is_valid_set(new_S):
                best_S = new_S
    
    return list(best_S)
# EVOLVE-BLOCK-END
