import time
import random

def is_valid(S):
    n = len(S)
    if n > 20:
        # Fast rejection: check all pairs of (sum, cardinality) via DP
        # Use a set of (sum, k) pairs
        pairs = set()
        pairs.add((0, 0))
        for val in S:
            new_pairs = set()
            for (s, k) in pairs:
                new_pairs.add((s + val, k + 1))
            # Check for collision: if (s, k) in new_pairs and (s, k) in pairs, 
            # that's fine (same cardinality). But if (s, k) in new_pairs 
            # and (s, k') in pairs with k' != k, that's a conflict.
            # Actually we need: for each (s, k) in new_pairs, if s already exists 
            # in pairs with a different k, conflict.
            existing_sums = {}
            for (s, k) in pairs:
                if s not in existing_sums:
                    existing_sums[s] = set()
                existing_sums[s].add(k)
            conflict = False
            for (s, k) in new_pairs:
                if s in existing_sums and k not in existing_sums[s]:
                    conflict = True
                    break
            if conflict:
                return False
            pairs = pairs | new_pairs
        return True
    
    # Exact check for small n
    sums = {}
    for mask in range(1 << n):
        s = 0
        k = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
                k += 1
        if s in sums:
            if k not in sums[s]:
                return False
        else:
            sums[s] = {k}
    return True

def try_add(S, x, used):
    """Try to add x to S, return True if valid."""
    n = len(S)
    S_new = S + [x]
    if not is_valid(S_new):
        return False
    return True

def build_greedy(order, deadline, S):
    S = list(S)
    used = set(S)
    for x in order:
        if time.time() > deadline:
            break
        if x in used:
            continue
        if try_add(S, x, used):
            S.append(x)
            used.add(x)
    return S

def construct():
    deadline = time.time() + 55  # Safety margin
    
    # Start with a known good set
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_size = len(best_set)
    
    # Try to build larger sets with different strategies
    strategies = [
        # Strategy 1: Small to large
        list(range(1, 257)),
        # Strategy 2: Large to small
        list(range(256, 0, -1)),
        # Strategy 3: Random
    ]
    
    # Generate a few random orders
    for _ in range(5):
        order = list(range(1, 257))
        random.shuffle(order)
        strategies.append(order)
    
    # Also try starting from empty
    strategies.append(list(range(1, 257)))  # will be handled separately
    
    for strategy_idx, order in enumerate(strategies):
        if time.time() > deadline - 10:
            break
        
        # Determine starting set
        if strategy_idx == len(strategies) - 1:
            # Start from empty
            S = []
        else:
            S = best_set[:]
        
        # Allocate time for this attempt
        remaining = deadline - time.time()
        attempt_budget = max(2.0, remaining / (len(strategies) - strategy_idx))
        attempt_deadline = min(deadline - 10, time.time() + attempt_budget)
        
        result = build_greedy(order, attempt_deadline, S)
        
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    # Final attempt with best set and random order
    if time.time() < deadline - 2:
        remaining = deadline - time.time()
        order = list(range(1, 257))
        random.shuffle(order)
        result = build_greedy(order, deadline - 2, best_set[:])
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    # Return as list of Python ints
    return [int(x) for x in best_set]

if __name__ == "__main__":
    s = construct()
    print(f"Size: {len(s)}")
    print(f"Valid: {is_valid(s)}")
    print(s)
