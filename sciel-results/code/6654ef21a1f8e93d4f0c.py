import time

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    def is_valid(S, target=200):
        """Check if no subset of S sums to target. S is a set of ints."""
        if not S:
            return True
        # DP: can we form target from elements of S?
        # Use a boolean array
        max_sum = sum(S)
        if max_sum < target:
            return True
        # Bitset approach for efficiency
        bits = 1  # bit i set means sum i is achievable
        for x in S:
            if x > target:
                continue
            bits |= (bits << x)
        return (bits >> target) & 1 == 0
    
    # Start with known valid set
    S = set(range(66, 101))
    best = S.copy()
    best_size = len(S)
    
    # Try to add elements from [1,65]
    candidates = list(range(1, 66))
    
    # Greedy: try adding each candidate if it doesn't break validity
    added = set()
    for x in candidates:
        if time.time() > deadline:
            break
        new_S = S | {x}
        if is_valid(new_S):
            S = new_S
            added.add(x)
            if len(S) > best_size:
                best = S.copy()
                best_size = len(S)
    
    # Now try local search: swap elements to potentially add more
    # Try removing one element from S and adding two from outside
    outside = set(range(1, 101)) - S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for remove in list(S):
            if time.time() > deadline:
                break
            S_without = S - {remove}
            for add1 in outside:
                if time.time() > deadline:
                    break
                for add2 in outside:
                    if add2 <= add1:
                        continue
                    if add1 == remove or add2 == remove:
                        continue
                    new_S = S_without | {add1, add2}
                    if len(new_S) > len(S):
                        if is_valid(new_S):
                            S = new_S
                            outside = set(range(1, 101)) - S
                            if len(S) > best_size:
                                best = S.copy()
                                best_size = len(S)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    # Also try single additions after swaps
    outside = set(range(1, 101)) - S
    for x in outside:
        if time.time() > deadline:
            break
        new_S = S | {x}
        if is_valid(new_S):
            S = new_S
            outside = set(range(1, 101)) - S
            if len(S) > best_size:
                best = S.copy()
                best_size = len(S)
    
    result = list(best)
    # Final validation
    assert len(result) > 0
    assert is_valid(set(result))
    return result
