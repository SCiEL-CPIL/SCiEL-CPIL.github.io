"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin
    
    N = 128
    HIGH_START = 64
    HIGH_END = 128
    
    # The high set is fixed to {64, ..., 128}
    # Size 65
    # We need to find the largest X subset of {1, ..., 63} such that:
    # 1. For all distinct x, y in X: x + y < 64
    # 2. For all distinct x, y, z in X: NOT (x+y in X AND x+z in X AND y+z in X)
    #    Note: Since x+y < 64, x+y is not in High. So x+y is in the total set iff x+y in X.
    
    def is_valid_X(X_set):
        """Check if a set of low numbers is valid when combined with High."""
        # Constraint 1: Pair sums < 64
        # If any pair sum >= 64, it's in High, causing violation with any c in High.
        # So we must have max(x,y) + min(x,y) < 64 for all distinct x,y?
        # Actually, just x+y < 64 for all distinct x,y in X.
        items = list(X_set)
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                if items[i] + items[j] >= 64:
                    return False
        
        # Constraint 2: No triple x,y,z in X such that x+y, x+z, y+z are all in X.
        # Since sums are < 64, they are in the total set iff they are in X.
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                for k in range(j+1, len(items)):
                    x, y, z = items[i], items[j], items[k]
                    s1 = x + y
                    s2 = x + z
                    s3 = y + z
                    if s1 in X_set and s2 in X_set and s3 in X_set:
                        return False
        return True

    def can_add_to_X(X_set, x):
        """Check if x can be added to X_set while maintaining validity with High."""
        if x in X_set:
            return True
        
        new_X = X_set | {x}
        return is_valid_X(new_X)

    # Initial Greedy Construction of X
    # Sort candidates 1..63
    # We want to maximize size. Small numbers are more likely to satisfy x+y<64.
    # But small numbers form sums that might be in X.
    # Let's try greedy with random restarts.
    
    best_X = set()
    best_size = 0
    
    def greedy_build_X():
        X = set()
        candidates = list(range(1, 64))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add_to_X(X, x):
                X.add(x)
        return X

    # Run multiple greedy restarts
    for _ in range(1000):
        if time.time() > deadline - 10:
            break
        X = greedy_build_X()
        if len(X) > best_size:
            best_size = len(X)
            best_X = X

    # Local Search on X
    # Try swapping elements or removing/adding.
    # Since the space is small, we can be more exhaustive.
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        current_X = best_X.copy()
        
        # Strategy 1: Try to add any missing element
        missing = [i for i in range(1, 64) if i not in current_X]
        random.shuffle(missing)
        for x in missing:
            if time.time() > deadline - 2:
                break
            if can_add_to_X(current_X, x):
                current_X.add(x)
                improved = True
                best_X = current_X
                best_size = len(current_X)
        
        # Strategy 2: Try to remove an element and see if we can add more
        if not improved:
            items = list(current_X)
            random.shuffle(items)
            for r in items:
                if time.time() > deadline - 2:
                    break
                new_X = current_X - {r}
                # Try to fill new_X
                missing = [i for i in range(1, 64) if i not in new_X]
                random.shuffle(missing)
                for x in missing:
                    if can_add_to_X(new_X, x):
                        new_X.add(x)
                
                if len(new_X) > len(best_X):
                    best_X = new_X
                    best_size = len(best_X)
                    improved = True
                    break
                elif len(new_X) == len(best_X):
                    # Optional: accept equal size to escape local optima?
                    # Let's stick to strict improvement for now to ensure progress.
                    pass

    # Construct final set
    result = list(range(64, 129))
    result.extend(best_X)
    
    # Final validation just in case
    # The task requires distinct integers in 1..128.
    # The logic above ensures validity.
    
    return result

# EVOLVE-BLOCK-END
