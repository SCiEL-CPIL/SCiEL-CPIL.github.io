import random
import time

def construct():
    n = 256
    deadline = time.time() + 50  # 50s limit to be safe
    
    def is_valid(mask):
        # Extract elements
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        
        # Check condition
        # For each a in elems:
        #   Let S_gt_a = [x for x in elems if x > a]
        #   Check if any b, c in S_gt_a (b <= c) such that a | (b+c)
        
        # Optimize:
        # Sort elems
        elems.sort()
        
        for idx_a, a in enumerate(elems):
            # Elements greater than a are elems[idx_a+1:]
            S_gt_a = elems[idx_a+1:]
            if not S_gt_a:
                continue
            
            # Count residues
            # We need to check if there exist b, c in S_gt_a such that (b+c) % a == 0
            # b, c can be same or different.
            # Let R be the set of residues of elements in S_gt_a mod a.
            # We need to know if there are b, c in S_gt_a with b % a + c % a == 0 (mod a).
            
            # Use a dictionary or array for counts
            # Since a <= 256, array of size a is fine.
            counts = [0] * a
            
            for x in S_gt_a:
                counts[x % a] += 1
            
            # Check for conflicts
            # 1. b != c: need r1 + r2 == 0 (mod a) with r1 != r2 (or r1==r2 but count>=2)
            #    Actually, if r1 != r2, we just need count[r1]>=1 and count[r2]>=1.
            #    If r1 == r2, we need count[r1]>=2.
            # 2. b == c: need 2*r == 0 (mod a). This is covered by r1==r2 case if we allow picking same element?
            #    The problem says a < b <= c.
            #    If b=c, it's the same element. So we need count[r] >= 1 for r such that 2r == 0 (mod a)?
            #    No, if b=c=x, then x is picked once. So we need count[r] >= 1.
            #    Wait, if b=c, we pick x. Sum is 2x.
            #    So if there exists x in S_gt_a such that 2x % a == 0, it's invalid.
            #    This corresponds to residue r = x % a such that 2r % a == 0.
            #    And count[r] >= 1.
            
            # Let's iterate through residues
            for r in range(a):
                target = (-r) % a
                if r < target:
                    if counts[r] > 0 and counts[target] > 0:
                        return False
                elif r == target:
                    # 2r == 0 (mod a)
                    # Case b=c: need at least one element with residue r.
                    # Case b!=c: need at least two elements with residue r.
                    # Since b=c is allowed, having 1 element is enough to form b=c=x.
                    # So if counts[r] >= 1, it's invalid.
                    if counts[r] >= 1:
                        return False
            # Note: if r > target, it's covered by r < target in the other iteration.
            
        return True

    def greedy_construct(order):
        mask = 0
        for x in order:
            new_mask = mask | (1 << (x - 1))
            if is_valid(new_mask):
                mask = new_mask
        return mask

    def local_search(mask):
        # Try to add elements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each missing element
            for x in range(1, n + 1):
                if mask & (1 << (x - 1)):
                    continue
                new_mask = mask | (1 << (x - 1))
                if is_valid(new_mask):
                    mask = new_mask
                    improved = True
                    # Restart loop as mask changed
                    break
            
            # If no single addition worked, try swaps?
            # Simple swap: remove i, add j.
            # This is O(n^2 * cost(is_valid)). Might be too slow.
            # Let's stick to single additions for now, and rely on restarts.
            
        return mask

    best_mask = 0
    best_size = 0
    
    # Phase 1: Randomized Greedy
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        mask = greedy_construct(order)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Phase 2: Local Search on best
    if time.time() < deadline - 5:
        best_mask = local_search(best_mask)
        best_size = bin(best_mask).count('1')
        
    # Convert back to list
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    # Fallback
    if not result:
        return list(range(192, 257))
        
    return result
