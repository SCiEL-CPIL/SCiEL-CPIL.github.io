import time
import random

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    # Precompute conflicts
    # conflicts[a] = set of frozenset({b, c}) such that 1/a = 1/b + 1/c and a,b,c distinct in [1, 180]
    conflicts = {a: set() for a in range(1, 181)}
    
    for a in range(1, 181):
        a2 = a * a
        # Find divisors of a^2
        # d = b - a
        # b = a + d
        # c = a + a^2/d
        # We need b, c in [1, 180]
        # d must be a divisor of a^2
        # Also b > a => d > 0
        # c > a => a^2/d > 0 => d > 0
        # We iterate d from 1 to a^2
        # Optimization: only check d such that b, c <= 180
        # b = a + d <= 180 => d <= 180 - a
        # c = a + a^2/d <= 180 => a^2/d <= 180 - a => d >= a^2 / (180 - a)
        
        # Let's just iterate divisors
        divs = []
        i = 1
        while i * i <= a2:
            if a2 % i == 0:
                divs.append(i)
                if i != a2 // i:
                    divs.append(a2 // i)
            i += 1
        
        for d in divs:
            b = a + d
            c_val = a2 // d
            c = a + c_val
            
            # Check bounds and distinctness
            if 1 <= b <= 180 and 1 <= c <= 180:
                if b != a and c != a and b != c:
                    # Store the pair
                    pair = frozenset((b, c))
                    conflicts[a].add(pair)
    
    def can_add(a, S):
        """
        Check if adding 'a' to valid set 'S' keeps it valid.
        """
        for pair in conflicts[a]:
            b, c = tuple(pair)
            if b in S and c in S:
                return False
        return True

    # Start with a strong base: All odds + Powers of 2
    S = set()
    for i in range(1, 181):
        if i % 2 != 0:
            S.add(i)
        else:
            # Check if power of 2
            p = i
            is_pow2 = True
            while p % 2 == 0:
                p //= 2
            if p == 1:
                S.add(i)
    
    # Greedily add remaining numbers
    # Sort candidates by number of conflicts (fewest first) to allow more additions?
    # Or just ascending.
    candidates = [a for a in range(1, 181) if a not in S]
    # Heuristic: sort by number of conflicts with current S?
    # Or just try all and add if possible.
    # Let's try a few random orders or sorted by conflict count.
    # For now, ascending is fine, but let's try to be smarter.
    # Actually, just iterating 1..180 is fast enough with the new can_add.
    for a in range(1, 181):
        if time.time() > deadline:
            break
        if a not in S:
            if can_add(a, S):
                S.add(a)
    
    # Local search
    random.seed(42)
    best_S = set(S)
    best_size = len(S)
    
    all_nums = set(range(1, 181))
    
    # Hill climbing with swaps
    # Since can_add is fast, we can do more iterations.
    # We will try:
    # 1. Add a random excluded element if possible.
    # 2. Swap a random included element with a random excluded element if it improves or maintains size (but we want max size, so only add or swap that allows adding more later? No, just size).
    # Actually, if we swap, size stays same. We want to increase size.
    # So:
    # - Try to add any excluded element.
    # - If no add is possible, try to remove one element to enable adding two others? (2-opt)
    # - Or just do random restarts with different greedy orders.
    
    # Let's do a more aggressive local search:
    # 1. Try to add any element.
    # 2. If stuck, try to remove one element x, then try to add two elements y, z.
    
    for _ in range(50000): # Increased iterations due to speedup
        if time.time() > deadline:
            break
        
        excluded = all_nums - S
        if not excluded:
            break
        
        # Try to add
        added = False
        for a in random.sample(list(excluded), min(10, len(excluded))):
            if can_add(a, S):
                S.add(a)
                added = True
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                break
        
        if not added:
            # Try swap: remove b, add a. This doesn't increase size, but might unblock future adds.
            # Or try 2-opt: remove b, add a, add c.
            # Let's try simple swap first to escape local minima.
            if S:
                b = random.choice(list(S))
                a = random.choice(list(excluded))
                S.remove(b)
                if can_add(a, S):
                    S.add(a)
                    # Check if we can add another now
                    remaining_excluded = all_nums - S
                    if remaining_excluded:
                        for c in random.sample(list(remaining_excluded), min(5, len(remaining_excluded))):
                            if can_add(c, S):
                                S.add(c)
                                if len(S) > best_size:
                                    best_S = set(S)
                                    best_size = len(S)
                                break
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                else:
                    S.add(b)

    return sorted(best_S)
