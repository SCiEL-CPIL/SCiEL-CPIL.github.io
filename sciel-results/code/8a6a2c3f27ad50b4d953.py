"""Program for constructing large 4-term AP-free subsets of [1, 512]."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 512
    
    # Precompute conflicts: for each element x (0-indexed), list of bitmasks
    # representing the other 3 elements needed to form a 4-AP with x.
    # Bit i in a mask corresponds to element i+1.
    conflicts = [[] for _ in range(N)]
    
    # Iterate over all 4-APs (a, a+d, a+2d, a+3d)
    for a in range(1, N - 2):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # For a: need b, c, e
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[a - 1].append(mask_a)
            
            # For b: need a, c, e
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[b - 1].append(mask_b)
            
            # For c: need a, b, e
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
            conflicts[c - 1].append(mask_c)
            
            # For e: need a, b, c
            mask_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflicts[e - 1].append(mask_e)

    def is_valid_addition(x, S):
        """Check if element x (1-indexed) can be added to bitmask S."""
        idx = x - 1
        for m in conflicts[idx]:
            if (S & m) == m:
                return False
        return True

    def base3_greedy():
        """
        Construct a 4-AP-free set by taking numbers in [1, N] whose base-3
        representation contains only digits 0 and 1.
        This is a known construction (Salem-Spencer type) that guarantees no 3-AP,
        and thus no 4-AP.
        """
        S = 0
        for x in range(1, N + 1):
            num = x
            valid = True
            while num > 0:
                if num % 3 == 2:
                    valid = False
                    break
                num //= 3
            if valid:
                S |= (1 << (x - 1))
        return S

    def local_search_improve(S, time_limit):
        """
        Improve the set S by trying to add elements greedily.
        Uses multiple passes with different random orders to find better additions.
        """
        current_S = S
        current_size = bin(current_S).count('1')
        best_S = current_S
        best_size = current_size
        
        # Try random order additions
        # We maintain a list of candidates not in S
        # But for simplicity and speed, we just shuffle [1..N] and try to add if not present.
        
        # Since N=512, checking "if not in S" is fast with bitmask.
        
        while time.time() < time_limit:
            order = list(range(1, N + 1))
            random.shuffle(order)
            
            # We can try to add elements in this order.
            # But to be more efficient, we could skip elements already in S.
            # However, the loop is fast enough.
            
            # One pass of greedy addition
            new_S = current_S
            # Optimization: Only try to add elements not in current_S?
            # Actually, if an element is in current_S, is_valid_addition is irrelevant.
            # But we don't want to "add" it again.
            
            # Let's just iterate and try to add if valid and not present.
            for x in order:
                if not (new_S & (1 << (x - 1))):
                    if is_valid_addition(x, new_S):
                        new_S |= (1 << (x - 1))
            
            size = bin(new_S).count('1')
            if size > best_size:
                best_size = size
                best_S = new_S
                current_S = new_S # Continue from this improved state? 
                # Or restart from base? 
                # Usually, continuing from the best found so far is good for hill climbing.
                # But random shuffles might get stuck. 
                # Let's update current_S to the best so far to build on it.
            
        # Final check: try adding any element that might have been missed
        # This is a "saturation" step
        for x in range(1, N + 1):
            if not (best_S & (1 << (x - 1))):
                if is_valid_addition(x, best_S):
                    best_S |= (1 << (x - 1))
        
        return best_S

    # 1. Start with Base-3 construction
    base_S = base3_greedy()
    
    # 2. Try a few random initial sets too?
    # Base-3 gives ~ N / sqrt(N) ? 
    # Actually, the density of numbers with only 0/1 in base 3 up to 3^k is 2^k.
    # 3^k ~ N => k ~ log_3 N. Count ~ 2^(log_3 N) = N^(log_3 2) ~ N^0.63.
    # For N=512, 512^0.63 ~ 86.
    # Let's see the actual count.
    base_size = bin(base_S).count('1')
    
    # 3. Local Search
    # Use most of the time budget for local search
    # We can run local search starting from base_S.
    # We can also try random restarts of local search from different seeds?
    # But base_S is a very specific structure. 
    # Maybe combining base_S with random additions is the key.
    
    best_S = base_S
    best_size = base_size
    
    # Run local search
    # We need to be careful with the time limit inside local_search_improve.
    # Let's just run it until deadline - 2s.
    
    # To diversify, we might try starting from slightly perturbed base sets?
    # Or just rely on the random order shuffling in local_search_improve.
    # The local_search_improve function above updates current_S to the best found,
    # which might lead to getting stuck if the first few shuffles are bad.
    # Let's modify local_search_improve to keep the global best and maybe restart?
    
    def local_search_global(S_init, time_limit):
        current_S = S_init
        current_size = bin(current_S).count('1')
        best_S = current_S
        best_size = current_size
        
        while time.time() < time_limit:
            order = list(range(1, N + 1))
            random.shuffle(order)
            
            new_S = current_S
            for x in order:
                if not (new_S & (1 << (x - 1))):
                    if is_valid_addition(x, new_S):
                        new_S |= (1 << (x - 1))
            
            size = bin(new_S).count('1')
            if size > best_size:
                best_size = size
                best_S = new_S
                current_S = new_S
            
        return best_S

    # Use 80% of remaining time for local search
    # But we want to try multiple "strategies" if possible?
    # For now, just run the local search from base_S.
    
    # To ensure we don't spend too long in one pass if it's slow,
    # the loop in local_search_global is bounded by time.
    
    final_deadline = deadline - 2.0
    best_S = local_search_global(base_S, final_deadline)
    
    # Final saturation pass
    for x in range(1, N + 1):
        if not (best_S & (1 << (x - 1))):
            if is_valid_addition(x, best_S):
                best_S |= (1 << (x - 1))
                
    # Convert to list
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    # Verification
    def has_4ap(s):
        s_set = set(s)
        for a in s_set:
            for b in s_set:
                if a >= b:
                    continue
                d = b - a
                if d == 0:
                    continue
                c = b + d
                e = c + d
                if c in s_set and e in s_set:
                    return True
        return False
        
    if has_4ap(result):
        return [1]
        
    return sorted(result)

# EVOLVE-BLOCK-END
