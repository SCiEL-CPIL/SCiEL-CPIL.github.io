import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    
    def is_valid_set(S):
        """Check if S satisfies: for all a<b<=c in S, a does not divide b+c."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                if (2 * b) % a == 0:
                    return False
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add_greedy(S, x):
        """Check if x can be added to S when all elements in S are > x (greedy case)."""
        # x will be the smallest, so we only need to check if x divides any b+c for b,c in S
        s_list = list(S)
        m = len(s_list)
        for i in range(m):
            b = s_list[i]
            if (2 * b) % x == 0:
                return False
            for j in range(i + 1, m):
                c = s_list[j]
                if (b + c) % x == 0:
                    return False
        return True
    
    def can_add_general(S, x):
        """General check if x can be added to S."""
        if x in S:
            return False
        S_list = sorted(S)
        m = len(S_list)
        
        # For each a in S where a < x, check if a divides (x + b) or (x + c) for b,c in S
        for i in range(m):
            a = S_list[i]
            if a >= x:
                break
            # x is b: need a < x <= c, check (x + c) % a != 0 for c >= x in S
            for j in range(m):
                c = S_list[j]
                if c >= x and (x + c) % a == 0:
                    return False
            # x is c: need a < b <= x, check (b + x) % a != 0 for b <= x in S
            for j in range(m):
                b = S_list[j]
                if a < b <= x and (b + x) % a == 0:
                    return False
        
        # x is a: need x < b <= c, check (b + c) % x != 0
        for i in range(m):
            b = S_list[i]
            if b > x:
                if (2 * b) % x == 0:
                    return False
                for j in range(i + 1, m):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        return False
        return True
    
    def greedy_from_top():
        """Greedy addition from largest to smallest."""
        S = set()
        for x in range(n, 0, -1):
            if time.time() > deadline - 5.0:
                break
            if can_add_greedy(S, x):
                S.add(x)
        return S
    
    def local_improve(S, max_time=20.0):
        """Try to swap larger elements with smaller ones to increase size."""
        S = set(S)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            S_sorted = sorted(S, reverse=True)
            
            for i, large in enumerate(S_sorted):
                if time.time() > max_time:
                    break
                
                S_without = S - {large}
                
                # Try replacing with smaller x
                for x in range(1, large):
                    if x in S:
                        continue
                    if can_add_general(S_without, x):
                        new_S = S_without | {x}
                        if len(new_S) >= len(S):
                            S = new_S
                            improved = True
                            break
        return S
    
    best_S = greedy_from_top()
    best_size = len(best_S)
    
    # Try local improvement on greedy result
    if time.time() < deadline - 10.0:
        improved_S = local_improve(best_S, max_time=deadline - time.time() - 10.0)
        if len(improved_S) > best_size:
            best_S = improved_S
            best_size = len(best_S)
    
    # Multiple restarts with different strategies
    for restart in range(30):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart * 7919 + 42)
        
        # Strategy: start with random subset of upper half, then greedily add
        base_start = n - rng.randint(30, 70)
        S = set()
        
        # Add upper elements greedily
        for x in range(n, base_start - 1, -1):
            if time.time() > deadline - 3.0:
                break
            if can_add_greedy(S, x):
                S.add(x)
        
        # Try to add lower elements in random order
        candidates = [x for x in range(1, base_start) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add_general(S, x):
                S.add(x)
        
        # Second pass
        candidates2 = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 3.0:
                break
            if can_add_general(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
        
        # Quick local improvement if promising
        if len(S) >= best_size - 2 and time.time() < deadline - 8.0:
            imp_S = local_improve(S, max_time=deadline - time.time() - 8.0)
            if len(imp_S) > best_size:
                best_S = imp_S
                best_size = len(imp_S)
    
    return list(best_S)
