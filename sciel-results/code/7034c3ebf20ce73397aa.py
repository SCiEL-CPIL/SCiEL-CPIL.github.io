import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    
    def bit(i):
        return 1 << i

    def has(mask, i):
        return (mask >> i) & 1

    def popcount(mask):
        return bin(mask).count('1')

    # State: mask, elems (list of ints in mask)
    # Initial: {65..128}
    mask = 0
    elems = list(range(65, 129))
    for i in elems:
        mask |= bit(i)
        
    best_mask = mask
    best_elems = elems[:]
    best_size = len(elems)
    
    # Helper to check if x can be added
    def can_add(x, mask, elems):
        # x not in mask
        # Check pairs a, b in elems
        # a+b, a+x, b+x must not all be in mask|{x}
        m_new = mask | bit(x)
        # We can check if (a+b) in m_new AND (a+x) in m_new AND (b+x) in m_new
        # Note: a+x and b+x are definitely in m_new only if they are <= 128 and in mask or equal x (impossible)
        # Actually, if a+x > 128, it's not in m_new.
        
        L = len(elems)
        for i in range(L):
            a = elems[i]
            for j in range(i+1, L):
                b = elems[j]
                s1 = a + b
                s2 = a + x
                s3 = b + x
                if s1 <= n and s2 <= n and s3 <= n:
                    if has(m_new, s1) and has(m_new, s2) and has(m_new, s3):
                        return False
        return True

    while time.time() < deadline:
        # Get missing elements
        # Efficient way: check 1..128
        missing = [i for i in range(1, 129) if not has(mask, i)]
        
        if not missing:
            break
            
        random.shuffle(missing)
        added = False
        for x in missing:
            if can_add(x, mask, elems):
                mask |= bit(x)
                elems.append(x)
                added = True
                # Update best if improved
                if len(elems) > best_size:
                    best_size = len(elems)
                    best_mask = mask
                    best_elems = elems[:]
                    
        if not added:
            # Kick: remove a random element
            if len(elems) > 0:
                idx = random.randint(0, len(elems)-1)
                rem = elems[idx]
                mask &= ~bit(rem)
                elems.pop(idx)
                # Don't update best here, as size decreased
                
    # Return best
    result = sorted(best_elems)
    return [int(x) for x in result]
