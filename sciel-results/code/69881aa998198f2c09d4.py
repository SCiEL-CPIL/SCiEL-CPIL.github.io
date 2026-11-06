import time

def _precompute_triples(N=180):
    """Precompute all forbidden triples and conflicts."""
    # conflicts[x] will store sets of pairs (b, c) with b < c, b, c > x
    # such that 1/x = 1/b + 1/c
    conflicts = [set() for _ in range(N+1)]
    
    for x in range(1, N+1):
        x2 = x * x
        # (b-x)(c-x) = x^2
        # Let d = b-x, e = c-x. d*e = x^2.
        # b = x+d, c = x+e.
        # We assume wlog b <= c, so d <= e.
        # d <= sqrt(x^2) = x.
        # Also b, c <= N => x+d <= N, x+e <= N.
        for d in range(1, int(x2**0.5) + 1):
            if x2 % d == 0:
                e = x2 // d
                if d == e:
                    # b = c = 2x, not distinct
                    continue
                b = x + d
                c = x + e
                if b <= N and c <= N:
                    # Ensure b != c is handled by d != e
                    # Store pair (min, max)
                    if b < c:
                        conflicts[x].add((b, c))
                    else:
                        conflicts[x].add((c, b))
    return conflicts

def _is_valid_addition(x, selected_set, conflicts, N):
    """
    Check if x can be added to selected_set without creating a forbidden triple.
    1. x as 'a': 1/x = 1/b + 1/c with b,c in selected_set.
    2. x as 'b' or 'c': 1/a = 1/x + 1/c with a,c in selected_set, a < x.
    """
    # Check 1: x is the smallest element 'a'
    # If 1/x = 1/b + 1/c, then b,c > x.
    for (b, c) in conflicts[x]:
        if b in selected_set and c in selected_set:
            return False
    
    # Check 2: x is a larger element 'b' (or 'c')
    # 1/a = 1/x + 1/c  =>  1/c = 1/a - 1/x = (x-a)/(ax)
    # c = ax / (x-a)
    # We need a in selected_set, a < x.
    # If c is integer, c in selected_set, c != x, c != a.
    for a in selected_set:
        if a >= x:
            continue
        # Calculate c
        num = a * x
        den = x - a
        if num % den == 0:
            c = num // den
            # c must be distinct from a and x
            # Since a < x, c = ax/(x-a).
            # If c == a, ax = a(x-a) => x = x-a => a=0 (impossible)
            # If c == x, ax = x(x-a) => a = x-a => 2a=x => c=x.
            # So we just need c != x.
            if c != x and c in selected_set:
                return False
            # Note: If c == x, it's not distinct, so it's allowed by "three distinct" rule?
            # The problem says "no THREE DISTINCT elements".
            # If c == x, the elements are a, x, x. Not distinct. So it's okay.
            # But wait, if c is in selected_set and c == x, that means x is already in selected_set?
            # No, we are checking if we *can* add x. x is NOT in selected_set yet.
            # So c cannot be x if c is in selected_set.
            # Unless selected_set already contains x? No, we check `if x in selected_set: continue` before calling this.
            
    return True

def _greedy_add(selected, conflicts, order, deadline, N):
    """Greedily add elements in given order."""
    selected_set = set(selected)
    result = list(selected)
    for x in order:
        if time.time() > deadline:
            break
        if x in selected_set:
            continue
        if _is_valid_addition(x, selected_set, conflicts, N):
            result.append(x)
            selected_set.add(x)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 180
    conflicts = _precompute_triples(N)
    
    if time.time() > deadline:
        return list(range(91, 181))
    
    best = list(range(91, 181))
    best_size = len(best)
    
    # Strategy 1: Greedy from large to small
    order1 = list(range(N, 0, -1))
    result1 = _greedy_add([], conflicts, order1, deadline, N)
    if len(result1) > best_size:
        best = result1
        best_size = len(result1)
    
    if time.time() > deadline:
        return best
    
    # Strategy 2: Greedy from small to large
    order2 = list(range(1, N+1))
    result2 = _greedy_add([], conflicts, order2, deadline, N)
    if len(result2) > best_size:
        best = result2
        best_size = len(result2)
    
    if time.time() > deadline:
        return best
    
    # Strategy 3: Start with large numbers, then add small ones that fit
    base = set(range(91, 181))
    result3 = list(range(91, 181))
    for x in range(1, 91):
        if time.time() > deadline:
            break
        if _is_valid_addition(x, base, conflicts, N):
            result3.append(x)
            base.add(x)
    
    if len(result3) > best_size:
        best = result3
        best_size = len(result3)
    
    if time.time() > deadline:
        return best
    
    # Strategy 4: Local improvement / Randomized Greedy
    import random
    random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        order5 = list(range(1, N+1))
        random.shuffle(order5)
        result5 = _greedy_add([], conflicts, order5, deadline, N)
        if len(result5) > best_size:
            best = result5
            best_size = len(result5)
            
    return best
