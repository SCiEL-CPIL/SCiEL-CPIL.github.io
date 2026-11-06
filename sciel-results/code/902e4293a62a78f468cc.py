"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s safety margin
    
    n = 256
    
    # Step 1: Generate the Cantor set (base-3 digits in {0,1}) in [1, n]
    cantor = set()
    for x in range(1, n + 1):
        # Check if x in base 3 has only digits 0 and 1
        temp = x
        valid = True
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            cantor.add(x)
    
    # Step 2: Try to add elements that don't create 4-APs
    # A 4-AP is a, a+d, a+2d, a+3d for d > 0
    def has_4ap(s, candidate):
        """Check if adding candidate to set s creates a 4-AP."""
        s_with = s | {candidate}
        # For each possible 4-AP that includes candidate
        # candidate can be at position 0, 1, 2, or 3 in the 4-AP
        for d in range(1, (n + 1) // 3 + 1):
            # candidate as position 0: a, a+d, a+2d, a+3d where a=candidate
            a = candidate
            if a + 3 * d <= n:
                if (a + d in s_with) and (a + 2 * d in s_with) and (a + 3 * d in s_with):
                    return True
            # candidate as position 1: a, a+d, a+2d, a+3d where a+d=candidate
            a = candidate - d
            if a >= 1:
                if (a in s_with) and (a + 2 * d in s_with) and (a + 3 * d in s_with):
                    return True
            # candidate as position 2: a, a+d, a+2d, a+3d where a+2d=candidate
            a = candidate - 2 * d
            if a >= 1:
                if (a in s_with) and (a + d in s_with) and (a + 3 * d in s_with):
                    return True
            # candidate as position 3: a, a+d, a+2d, a+3d where a+3d=candidate
            a = candidate - 3 * d
            if a >= 1:
                if (a in s_with) and (a + d in s_with) and (a + 2 * d in s_with):
                    return True
        return False
    
    # Greedily add elements that maximize 3-AP count
    def count_3aps(s):
        """Count distinct 3-term APs (positive difference, counted once)."""
        count = 0
        s_list = sorted(s)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                a = s_list[i]
                b = s_list[j]
                d = b - a
                c = b + d
                if c in s and c != b:
                    count += 1
        return count
    
    # Try adding each candidate in order of potential 3-AP gain
    current_set = set(cantor)
    current_3ap_count = count_3aps(current_set)
    
    candidates = [x for x in range(1, n + 1) if x not in cantor]
    
    # Sort candidates by how many 3-APs they could potentially form
    def potential_3aps(x, s):
        """Count how many 3-APs x would form if added to s."""
        count = 0
        # x as middle: a, x, c where a+c=2x, a,c in s
        for a in s:
            c = 2 * x - a
            if c in s and c != x and c > x:
                count += 1
        # x as first: x, x+d, x+2d
        for d in range(1, n):
            if x + d in s and x + 2 * d in s:
                count += 1
        # x as last: x-2d, x-d, x
        for d in range(1, n):
            if x - d in s and x - 2 * d in s:
                count += 1
        return count
    
    # Iteratively add the best candidate
    max_iter = 1000
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        best_candidate = None
        best_gain = 0
        
        for x in candidates:
            if x in current_set:
                continue
            if has_4ap(current_set, x):
                continue
            gain = potential_3aps(x, current_set)
            if gain > best_gain:
                best_gain = gain
                best_candidate = x
        
        if best_candidate is None or best_gain == 0:
            break
        
        current_set.add(best_candidate)
        candidates.remove(best_candidate)
        current_3ap_count += best_gain
    
    # Local search: try swapping elements to improve 3-AP count
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Try removing one element and adding another
        for remove_x in list(current_set):
            if time.time() > deadline - 5:
                break
            
            s_without = current_set - {remove_x}
            current_count = count_3aps(s_without)
            
            for add_x in range(1, n + 1):
                if add_x in s_without:
                    continue
                if has_4ap(s_without, add_x):
                    continue
                
                new_set = s_without | {add_x}
                new_count = count_3aps(new_set)
                
                if new_count > current_3ap_count:
                    current_set = new_set
                    current_3ap_count = new_count
                    improved = True
                    break
        
        if not improved:
            break
    
    return sorted(current_set)

# EVOLVE-BLOCK-END
