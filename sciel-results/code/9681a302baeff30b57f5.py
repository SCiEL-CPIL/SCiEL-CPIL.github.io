import time

def construct():
    start_time = time.time()
    deadline = start_time + 50

    N = 100
    T = 200

    def is_valid(S):
        reachable = 1
        for x in S:
            reachable |= (reachable << x)
            reachable &= (1 << 201) - 1
            if (reachable >> 200) & 1:
                return False
        return True

    def find_bad_subset(S, target=T):
        # Find a subset of S summing to target
        # Use DFS with pruning
        S_sorted = sorted(S, reverse=True)
        
        def dfs(start, curr_sum, curr_set):
            if curr_sum == target:
                return curr_set
            if curr_sum > target:
                return None
            for i in range(start, len(S_sorted)):
                if S_sorted[i] > target - curr_sum:
                    continue
                res = dfs(i + 1, curr_sum + S_sorted[i], curr_set + [S_sorted[i]])
                if res:
                    return res
            return None
        
        return dfs(0, 0, [])

    best_set = None
    best_size = 0

    # Try different initial thresholds
    for k in range(50, 80):
        if time.time() > deadline:
            break
            
        # Initial set: {k, ..., 100}
        current = set(range(k, N + 1))
        removed = set(range(1, k))
        
        # Ensure validity (it should be valid for k>=66, but check)
        if not is_valid(list(current)):
            # Find bad subset and fix by removing
            while not is_valid(list(current)):
                bad = find_bad_subset(list(current))
                if not bad:
                    break
                # Remove the largest element in bad to preserve small ones? 
                # Or remove the one that appears in most other sums?
                # For now, just remove one.
                # Remove the largest to keep the "dense" small part? 
                # Actually, we want to keep as many as possible.
                # Removing any one breaks this sum.
                # Let's remove the largest element in the bad subset to keep the set "high"
                # or remove the smallest?
                # Let's try removing the largest.
                to_remove = max(bad)
                current.remove(to_remove)
                removed.add(to_remove)
                if time.time() > deadline:
                    break

        # Now try to add back elements from 'removed'
        # Sort removed by value (descending) to try adding large ones first?
        # Or ascending?
        # Adding large numbers is "safer" as they form fewer sums?
        # Actually, large numbers need small partners.
        # Let's try adding in descending order.
        for x in sorted(removed, reverse=True):
            if time.time() > deadline:
                break
            # Try to add x
            new_current = set(current)
            new_current.add(x)
            
            if is_valid(list(new_current)):
                current = new_current
                removed.remove(x)
            else:
                # Try to swap
                bad = find_bad_subset(list(new_current))
                if bad:
                    # bad contains x
                    # Try to remove one other element y from bad
                    for y in bad:
                        if y == x:
                            continue
                        if y in removed:
                            continue
                        # Try swap: remove y, add x
                        new_current2 = set(current)
                        new_current2.add(x)
                        new_current2.discard(y)
                        
                        if is_valid(list(new_current2)):
                            current = new_current2
                            removed.remove(x)
                            removed.add(y)
                            break

        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)
            if best_size == N: # Perfect
                break

    if best_set is None:
        return list(range(66, N + 1))
        
    return best_set

# EVOLVE-BLOCK-END
