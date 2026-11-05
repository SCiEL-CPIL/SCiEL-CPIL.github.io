"""4-term-progression-free subset of [1,512] via greedy + local search."""
import time
import random

DEADLINE = time.time() + 85  # safety margin before 90s kill

def has_4ap_with(s, x):
    """Check if adding x to set s creates a 4-term AP."""
    # For each pair (a, b) in s with a < b, check if x completes a 4-AP
    # 4-AP: a, a+d, a+2d, a+3d
    # x could be at position 1, 2, or 3 (position 0 means x is the start)
    # If x is at position 0: x, x+d, x+2d, x+3d -> need x+d, x+2d, x+3d in s
    # If x is at position 1: a, x, x+d, x+2d where a = x-d -> need x-d, x+d, x+2d in s
    # If x is at position 2: a, a+d, x, x+d where a = x-2d, a+d = x-d -> need x-2d, x-d, x+d in s
    # If x is at position 3: a, a+d, a+2d, x where a+2d = x-2d... let's be more careful
    
    # A 4-AP is {a, a+d, a+2d, a+3d} for d > 0
    # x can replace any of the 4 positions
    # Position 0: x = a, so need x+d, x+2d, x+3d in s for some d>0
    # Position 1: x = a+d, so a = x-d, need x-d, x+d, x+2d in s for some d>0
    # Position 2: x = a+2d, so a = x-2d, a+d = x-d, a+3d = x+d, need x-2d, x-d, x+d in s
    # Position 3: x = a+3d, so a = x-3d, a+d = x-2d, a+2d = x-d, need x-3d, x-2d, x-d in s
    
    for d in range(1, 200):
        # Position 0
        if (x + d) in s and (x + 2*d) in s and (x + 3*d) in s:
            return True
        # Position 1
        if (x - d) in s and (x + d) in s and (x + 2*d) in s:
            return True
        # Position 2
        if (x - 2*d) in s and (x - d) in s and (x + d) in s:
            return True
        # Position 3
        if (x - 3*d) in s and (x - 2*d) in s and (x - d) in s:
            return True
    return False

def has_4ap_set(s):
    """Check if set s contains any nonconstant 4-term AP."""
    lst = sorted(s)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            d = b - a
            if d <= 0:
                continue
            if (a + 2*d) in s and (a + 3*d) in s:
                return True
    return False

def base3_01_set():
    """Generate set of numbers with only 0/1 digits in base 3, up to 512."""
    s = set()
    for mask in range(1, 1 << 6):  # 6 base-3 digits: positions 0-5
        x = 0
        for i in range(6):
            if mask & (1 << i):
                x += 3 ** i
        if 1 <= x <= 512:
            s.add(x)
    return s

def greedy_forward(s):
    """Try to add elements from 1 to 512 in order."""
    s = set(s)
    for x in range(1, 513):
        if x in s:
            continue
        if not has_4ap_with(s, x):
            s.add(x)
    return s

def local_search(s, max_iters=5000):
    """Local search: try to add elements, or swap to improve."""
    s = set(s)
    best_size = len(s)
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        improved = False
        # Try to add a random element
        candidates = [x for x in range(1, 513) if x not in s]
        random.shuffle(candidates)
        for x in candidates[:50]:
            if not has_4ap_with(s, x):
                s.add(x)
                improved = True
                break
        if not improved:
            # Try removing one element and adding two
            if len(s) >= 3:
                lst = list(s)
                random.shuffle(lst)
                for rem in lst[:20]:
                    s2 = s - {rem}
                    added = 0
                    for x in range(1, 513):
                        if x in s2 or x == rem:
                            continue
                        if not has_4ap_with(s2, x):
                            s2.add(x)
                            added += 1
                            if added >= 2:
                                break
                    if len(s2) > len(s):
                        s = s2
                        improved = True
                        break
        if not improved:
            # Try single swap: remove one, add one different
            lst = list(s)
            random.shuffle(lst)
            for rem in lst[:30]:
                s2 = s - {rem}
                best_add = None
                for x in range(1, 513):
                    if x in s2:
                        continue
                    if not has_4ap_with(s2, x):
                        best_add = x
                        break
                if best_add is not None:
                    s = s2 | {best_add}
                    improved = True
                    break
    return s

def construct():
    random.seed(42)
    # Start with base-3 0/1 set
    s = base3_01_set()
    s = greedy_forward(s)
    best = s
    best_size = len(s)
    
    # Local search with multiple restarts
    for restart in range(20):
        if time.time() > DEADLINE:
            break
        s_copy = set(best)
        # Perturb
        lst = list(s_copy)
        random.shuffle(lst)
        for _ in range(min(5, len(lst))):
            s_copy.discard(lst.pop())
        s_result = local_search(s_copy, max_iters=3000)
        if len(s_result) > best_size:
            best = s_result
            best_size = len(s_result)
    
    return sorted(best)
