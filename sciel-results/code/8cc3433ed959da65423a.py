import time
import random

# EVOLVE-BLOCK-START

def _validate_bitset(S):
    """Validate using a boolean array for O(1) membership. Returns True if valid."""
    n = len(S)
    if n < 3:
        return True
    in_set = [False] * 257
    for x in S:
        in_set[x] = True
    
    S_list = sorted(S)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            ab = a + b
            if ab > 128 or not in_set[ab]:
                continue
            for k in range(j + 1, n):
                c = S_list[k]
                ac = a + c
                if ac > 128 or not in_set[ac]:
                    continue
                bc = b + c
                if bc <= 128 and in_set[bc]:
                    return False
    return True

def _can_add_bitset(S, x, in_set):
    """Check if adding x to S creates a violation. in_set is a boolean array for S."""
    # New triples must include x.
    # For pair (x, a) where a in S: need x+a in S (since x+a > x, it must be in S not just S∪{x})
    # For triple (x, a, b): need x+a in S, x+b in S, and a+b in S∪{x}
    S_list = list(S)
    for a in S_list:
        xa = x + a
        if xa > 128 or not in_set[xa]:
            continue
        for b in S_list:
            if a == b:
                continue
            xb = x + b
            if xb > 128 or not in_set[xb]:
                continue
            ab = a + b
            if ab <= 128 and (in_set[ab] or ab == x):
                return False
    return True

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Start with known valid set {63..128}
    S = set(range(63, 129))
    best = set(S)
    best_size = len(S)
    
    all_nums = set(range(1, 129))
    
    def build_in_set(S):
        in_set = [False] * 257
        for x in S:
            in_set[x] = True
        return in_set
    
    # Phase 1: Greedy addition of elements from 1 to 62
    in_set = build_in_set(S)
    for x in range(1, 63):
        if time.time() > deadline - 5:
            break
        if x in S:
            continue
        if _can_add_bitset(S, x, in_set):
            S.add(x)
            in_set[x] = True
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
    
    # Phase 2: Local search - try remove 1, add 2 (net +1)
    random.seed(42)
    max_iterations = 5000
    improved = True
    iter_count = 0
    while improved and time.time() < deadline - 3 and iter_count < max_iterations:
        improved = False
        iter_count += 1
        S_list = list(S)
        missing = list(all_nums - S)
        if len(missing) < 2:
            break
        
        in_set = build_in_set(S)
        
        # Try random swaps: remove one element, add two
        for attempt in range(20):
            if time.time() > deadline - 3:
                break
            if not missing:
                break
            
            r = random.choice(S_list)
            a1 = random.choice(missing)
            missing2 = [m for m in missing if m != a1]
            if not missing2:
                continue
            a2 = random.choice(missing2)
            
            new_S = set(S)
            new_S.discard(r)
            new_S.add(a1)
            new_S.add(a2)
            
            if len(new_S) > len(S) and _validate_bitset(new_S):
                S = new_S
                if len(S) > best_size:
                    best = set(S)
                    best_size = len(S)
                improved = True
                S_list = list(S)
                missing = list(all_nums - S)
                break
        
        # Also try single additions after swaps
        in_set = build_in_set(S)
        for x in sorted(missing):
            if time.time() > deadline - 3:
                break
            if _can_add_bitset(S, x, in_set):
                S.add(x)
                in_set[x] = True
                if len(S) > best_size:
                    best = set(S)
                    best_size = len(S)
                improved = True
                missing = list(all_nums - S)
    
    # Phase 3: More aggressive local search with different strategy
    # Try removing 2, adding 3 (net +1)
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        S_list = list(S)
        missing = list(all_nums - S)
        if len(missing) < 3:
            break
        
        in_set = build_in_set(S)
        
        for attempt in range(10):
            if time.time() > deadline - 1:
                break
            if len(missing) < 3 or len(S_list) < 2:
                break
            
            r1 = random.choice(S_list)
            remaining = [s for s in S_list if s != r1]
            if not remaining:
                continue
            r2 = random.choice(remaining)
            
            a1 = random.choice(missing)
            missing2 = [m for m in missing if m != a1]
            if len(missing2) < 2:
                continue
            a2 = random.choice(missing2)
            missing3 = [m for m in missing2 if m != a2]
            if not missing3:
                continue
            a3 = random.choice(missing3)
            
            new_S = set(S)
            new_S.discard(r1)
            new_S.discard(r2)
            new_S.add(a1)
            new_S.add(a2)
            new_S.add(a3)
            
            if len(new_S) > len(S) and _validate_bitset(new_S):
                S = new_S
                if len(S) > best_size:
                    best = set(S)
                    best_size = len(S)
                improved = True
                break
        
        # Try single additions
        in_set = build_in_set(S)
        missing = list(all_nums - S)
        for x in sorted(missing):
            if time.time() > deadline - 1:
                break
            if _can_add_bitset(S, x, in_set):
                S.add(x)
                in_set[x] = True
                if len(S) > best_size:
                    best = set(S)
                    best_size = len(S)
                improved = True
                missing = list(all_nums - S)
    
    # Final: try to add any remaining elements
    in_set = build_in_set(best)
    for x in sorted(all_nums - best):
        if time.time() > deadline:
            break
        if _can_add_bitset(best, x, in_set):
            best.add(x)
            in_set[x] = True
    
    result = sorted(best)
    return list(result)

# EVOLVE-BLOCK-END
