import time
import random

def _can_add(x, S_set):
    """
    Check if x can be added to S_set without violating the constraint.
    Constraint: No distinct a,b,c in S such that a+b, a+c, b+c are in S.
    If we add x, the only new triples are those containing x.
    Let the triple be {x, a, b} with a, b in S_set, a != b.
    Sums: x+a, x+b, a+b.
    For the triple to be invalid in S_new = S_set U {x}:
    (x+a) must be in S_new
    (x+b) must be in S_new
    (a+b) must be in S_new
    
    Since x is the new element:
    - x+a > x (since a>=1), so x+a cannot be x. It must be in S_set.
    - x+b > x (since b>=1), so x+b cannot be x. It must be in S_set.
    - a+b can be equal to x. If a+b == x, it is in S_new. Otherwise, it must be in S_set.
    
    So the condition for INVALID is:
    (x+a in S_set) AND (x+b in S_set) AND (a+b in S_set OR a+b == x)
    """
    elems = list(S_set)
    n = len(elems)
    for i in range(n):
        a = elems[i]
        # If x+a is not in S_set, no pair with this 'a' can form a forbidden triple
        # because one of the required sums (x+a) would be missing from S_new.
        if (x + a) not in S_set:
            continue
            
        for j in range(i + 1, n):
            b = elems[j]
            # Check x+b
            if (x + b) not in S_set:
                continue
            # Check a+b
            if (a + b) in S_set or (a + b) == x:
                return False
    return True

def construct():
    start = time.time()
    deadline = 85.0  # safety margin
    
    best_mask = 0
    best_size = 0
    best_S = set()
    
    def greedy_random():
        """Randomized greedy: pick random order, add if valid."""
        order = list(range(1, 129))
        random.shuffle(order)
        mask = 0
        S = set()
        for x in order:
            if _can_add(x, S):
                mask |= (1 << (x-1))
                S.add(x)
        return mask, len(S), S
    
    # Phase 1: Multiple randomized greedy restarts
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        mask, size, S = greedy_random()
        if size > best_size:
            best_mask = mask
            best_size = size
            best_S = S.copy()
    
    # Phase 2: Local search (Additions)
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        candidates = [x for x in range(1, 129) if not (best_mask >> (x-1)) & 1]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if _can_add(x, best_S):
                best_mask |= (1 << (x-1))
                best_S.add(x)
                best_size += 1
                improved = True
                
    # Final validation
    def _is_valid(S):
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                for k in range(j+1, n):
                    c = S_list[k]
                    if (a+b) in S and (a+c) in S and (b+c) in S:
                        return False
        return True

    if not _is_valid(best_S):
        # Fallback
        best_S = set(range(65, 129))
        
    result = sorted(list(best_S))
    return [int(x) for x in result]
