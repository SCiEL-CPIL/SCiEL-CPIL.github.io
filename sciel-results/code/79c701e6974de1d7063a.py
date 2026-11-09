import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 55
    
    # Precompute forbidden triples: for each element x, which pairs (a,b) with a<b, a!=x, b!=x
    # would form a forbidden triple {a,b,x} where all pair sums are in 1..n.
    # A triple {a,b,c} is forbidden if a+b, a+c, b+c are all in 1..n (they don't need to be in S,
    # but if they ARE in S along with a,b,c, that's a violation).
    # Actually, the violation is: a,b,c in S AND a+b in S AND a+c in S AND b+c in S.
    # So when adding x, we check: does there exist a,b in S (a!=x, b!=x, a!=b) such that
    # {a,b,x} has all three pair sums in S?
    
    # For each x, precompute the list of pairs (a, b) with a < b, a != x, b != x,
    # such that a+b <= n, a+x <= n, b+x <= n.
    # Then when adding x to S, check if any such pair (a,b) is in S AND a+b in S AND a+x in S AND b+x in S.
    
    # But even better: for each x, precompute conflict_pairs[x] = list of (a, b, a+b, a+x, b+x)
    # where a < b, a != x, b != x, a+b <= n, a+x <= n, b+x <= n.
    
    conflict_pairs = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for a in range(1, n + 1):
            if a == x:
                continue
            for b in range(a + 1, n + 1):
                if b == x:
                    continue
                s_ab = a + b
                s_ax = a + x
                s_bx = b + x
                if s_ab <= n and s_ax <= n and s_bx <= n:
                    conflict_pairs[x].append((a, b, s_ab, s_ax, s_bx))
    
    def is_valid_add(S_mask, x):
        """Check if adding x to S_mask keeps it valid."""
        if (S_mask >> x) & 1:
            return False
        for (a, b, s_ab, s_ax, s_bx) in conflict_pairs[x]:
            # Check if a, b are in S
            if not ((S_mask >> a) & 1):
                continue
            if not ((S_mask >> b) & 1):
                continue
            
            # Check sums
            # s_ax and s_bx cannot be x, so they must be in S
            if not ((S_mask >> s_ax) & 1):
                continue
            if not ((S_mask >> s_bx) & 1):
                continue
            
            # s_ab can be x (in which case it will be in S after adding x)
            # or it must be in S already
            if not ((S_mask >> s_ab) & 1) and s_ab != x:
                continue
            
            # If we reach here, all conditions for violation are met
            return False
        return True
    
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
            if is_valid_add(current_mask, x):
                current_mask |= (1 << x)
        
        # Local Search: try swaps
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
                
                # Try to add candidates
                added_any = False
                c_local = [z for z in candidates if not (temp_mask >> z) & 1]
                random.shuffle(c_local)
                
                for z in c_local:
                    if is_valid_add(temp_mask, z):
                        temp_mask |= (1 << z)
                        added_any = True
                        break  # Only add one per swap attempt for simplicity
                
                if added_any:
                    current_mask = temp_mask
                    improved = True
                    break
        
        size = bin(current_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = current_mask

    # Convert mask to list
    result = [i for i in range(1, 129) if (best_mask >> i) & 1]
    
    # Final validation check
    def validate(S_mask):
        s_list = [i for i in range(1, n + 1) if (S_mask >> i) & 1]
        s_set = set(s_list)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                for k in range(j + 1, len(s_list)):
                    a, b, c = s_list[i], s_list[j], s_list[k]
                    if (a + b in s_set) and (a + c in s_set) and (b + c in s_set):
                        return False
        return True

    if not validate(best_mask):
        # Fallback to valid base set
        result = list(range(65, 129))
        
    return result
