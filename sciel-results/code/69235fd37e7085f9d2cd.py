import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 55
    
    # Bitmask representation
    # Bit i represents integer i+1? Or i?
    # Let's use bit i for integer i. 1..128.
    # So bit 1 is 1, bit 128 is 128.
    
    def is_valid_addition_mask(S_mask, x):
        """
        Check if adding x to S_mask keeps it valid.
        S_mask is an int.
        x is an int (1..128).
        """
        # K = {a in S | a+x in S}
        # a is in S: (S_mask >> a) & 1
        # a+x is in S: (S_mask >> (a+x)) & 1
        
        # We need to check if there exist distinct a, b in K such that a+b in S or a+b == x.
        # Also Case 2: x = a+b.
        
        # Precompute K as a list? Or iterate?
        # Iterate a from 1 to n-x.
        K = []
        for a in range(1, n - x + 1):
            if (S_mask >> a) & 1 and (S_mask >> (a + x)) & 1:
                K.append(a)
                
        if not K:
            # No a such that a+x in S.
            # But wait, we also need to check Case 2: x = a+b.
            # If K is empty, it means no a in S has a+x in S.
            # This doesn't prevent x = a+b.
            pass
        else:
            # Check Case 1: a, b in K, a!=b, a+b in S or a+b == x.
            # K is small.
            for i in range(len(K)):
                a = K[i]
                for j in range(len(K)):
                    if i == j: continue
                    b = K[j]
                    ab = a + b
                    if ab == x:
                        return False
                    if ab <= n and ((S_mask >> ab) & 1):
                        return False

        # Case 2: x = a+b.
        # Need a, b, c in S such that a+b=x, a+c in S, b+c in S.
        # Iterate a in S.
        # b = x - a.
        # if b in S and b != a:
        #   check c in S such that a+c in S and b+c in S.
        
        for a in range(1, x):
            if (S_mask >> a) & 1:
                b = x - a
                if b > 0 and b <= n and (S_mask >> b) & 1 and b != a:
                    # Check c
                    # c in S, c != a, c != b
                    # a+c in S, b+c in S
                    # Iterate c?
                    # We can check if there exists c.
                    # c must be in S.
                    # a+c in S.
                    # b+c in S.
                    
                    # Let's iterate c in S?
                    # Or iterate c from 1 to n?
                    # n=128.
                    for c in range(1, n + 1):
                        if c == a or c == b: continue
                        if (S_mask >> c) & 1:
                            ac = a + c
                            bc = b + c
                            if ac <= n and bc <= n:
                                if ((S_mask >> ac) & 1) and ((S_mask >> bc) & 1):
                                    return False
                            # If ac > n, it's not in S (since S subset 1..128).
                            # So condition fails.
                            
        return True

    def try_add_mask(S_mask, x):
        if (S_mask >> x) & 1:
            return S_mask, False
        if is_valid_addition_mask(S_mask, x):
            return S_mask | (1 << x), True
        return S_mask, False

    # Base set: 65..128
    base_mask = 0
    for i in range(65, 129):
        base_mask |= (1 << i)
        
    candidates = list(range(1, 65))
    
    best_mask = base_mask
    best_size = bin(base_mask).count('1')
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        c = candidates.copy()
        random.shuffle(c)
        
        current_mask = base_mask
        
        for x in c:
            if (current_mask >> x) & 1:
                continue
            current_mask, _ = try_add_mask(current_mask, x)
            
        # Local Search
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            s_list = [i for i in range(1, 129) if (current_mask >> i) & 1]
            random.shuffle(s_list)
            
            for y in s_list:
                if time.time() > deadline - 2:
                    break
                
                # Remove y
                temp_mask = current_mask & ~(1 << y)
                
                # Try to add
                added_any = False
                c_local = [z for z in candidates if not (temp_mask >> z) & 1]
                random.shuffle(c_local)
                
                for z in c_local:
                    new_mask, added = try_add_mask(temp_mask, z)
                    if added:
                        temp_mask = new_mask
                        added_any = True
                
                if added_any:
                    current_mask = temp_mask
                    improved = True
                else:
                    # Revert
                    pass # temp_mask was just temp, current_mask unchanged
                
                if improved:
                    break
        
        size = bin(current_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = current_mask

    # Convert mask to list
    result = [i for i in range(1, 129) if (best_mask >> i) & 1]
    return result
