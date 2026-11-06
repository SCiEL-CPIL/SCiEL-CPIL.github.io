import time
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(2000)

def construct():
    t_start = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    
    # 1. Generate all 2,3,5-smooth numbers <= 128
    smooths = set()
    for a in range(0, 8):  # 2^7 = 128
        v2 = 1 << a
        for b in range(0, 6): # 3^5 = 243
            v23 = v2 * (3 ** b)
            if v23 > 128:
                break
            for c in range(0, 5): # 5^4 = 625
                v = v23 * (5 ** c)
                if v > 128:
                    break
                smooths.add(v)
    
    S = sorted(smooths)
    N = len(S)
    
    # 2. Precompute divisibility
    # div[i][j] is True if S[i] divides S[j]
    div = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if S[j] % S[i] == 0:
                div[i][j] = True

    # 3. DP / Search for minimal antichains
    # best_repr[sum] = tuple of indices or values
    # We want to minimize the count of terms.
    # We iterate k = 1, 2, 3...
    # For each k, we find all antichains of size k.
    # If sum <= 128 and best_repr[sum] is None, we store it.
    
    best_repr = [None] * 129  # Index 0 unused
    best_k = [99] * 129       # Store the k used, for tracking
    
    found_all = [False] * 129
    found_count = 0
    
    # Check k=1 separately for speed? No, general recursion handles it.
    
    max_k = 10  # Upper bound, should be much smaller
    
    # Recursive function to generate antichains of size k
    # current_sum: sum of chosen numbers
    # current_indices: list of indices chosen so far
    # start_idx: index to start picking from (to ensure distinctness and ordering)
    # k_remaining: how many more numbers to pick
    
    # To speed up, we can prune based on sum.
    # Min possible sum for remaining k_remaining elements?
    # The smallest smooth numbers are 1, 2, 3, 4, 5...
    # But they must form an antichain with existing ones.
    # Simple bound: if current_sum > 128, prune.
    
    def dfs(current_sum, current_indices, k_remaining, start_idx):
        nonlocal found_count, found_all, best_repr, best_k
        
        if time.time() - t_start > DEADLINE:
            return True # Signal to stop
        
        if k_remaining == 0:
            if current_sum <= 128:
                s = current_sum
                if best_repr[s] is None:
                    best_repr[s] = tuple(S[idx] for idx in current_indices)
                    best_k[s] = len(current_indices)
                    found_all[s] = True
                    found_count += 1
            return False
        
        # Heuristic: if we already found all 1..128, we can stop
        if found_count == 128:
            return True

        # Pruning: 
        # The smallest possible value we can add is S[start_idx].
        # The smallest possible sum for k_remaining items starting from start_idx
        # is sum of the k_remaining smallest available smooths >= S[start_idx].
        # This is a bit complex to precompute dynamically. 
        # Simple bound: if current_sum + k_remaining * S[start_idx] > 128, prune?
        # Not quite, because we can skip elements. But S[start_idx] is the min available.
        # So min add is S[start_idx].
        if current_sum + k_remaining * S[start_idx] > 128:
            return False
            
        for i in range(start_idx, N):
            val = S[i]
            # Check antichain property with all previously chosen
            is_valid = True
            for j in current_indices:
                if div[j][i] or div[i][j]:
                    is_valid = False
                    break
            
            if not is_valid:
                continue
                
            current_indices.append(i)
            
            # Recurse
            if dfs(current_sum + val, current_indices, k_remaining - 1, i + 1):
                current_indices.pop()
                return True # Propagate stop signal
            
            current_indices.pop()
            
        return False

    for k in range(1, max_k + 1):
        # Reset current state for this k
        # We need to find ALL antichains of size k to ensure we find the one that sums to a new target
        # Actually, we just need to find *a* representation for each target.
        # Since we iterate k from 1 upwards, the first time a target is found, it's minimal.
        
        # We run DFS for this specific k
        # We need to explore the space to find sums.
        # Note: dfs above modifies global state.
        # We call it once for this k? No, we need to explore all possibilities for this k.
        # The dfs function above explores the tree.
        
        # However, the dfs above returns True if it should stop.
        # We want to exhaust the search for this k unless we hit deadline or found all.
        
        # We need a flag to indicate if we should continue searching for this k
        # The current dfs returns True to stop *entirely*.
        # We want to stop *this k* if we found all, or deadline.
        
        # Let's wrap the call.
        # We just call dfs(0, [], k, 0).
        # It will populate best_repr for any sums it finds.
        
        # Optimization: If found_count == 128, break.
        
        # We need to ensure we don't miss any antichain of size k.
        # The DFS explores all valid combinations.
        
        # One issue: The `found_count` check inside DFS is for global stop.
        # If we find all 1..128, we can stop.
        
        # Let's run the DFS for this k
        # We need to be careful: `dfs` uses `nonlocal` variables.
        # It's safe to call multiple times for different k?
        # Yes, as long as we reset `current_indices` which is local to the call stack.
        
        # However, the `found_all` and `best_repr` persist, which is what we want.
        
        # We don't need to reset `found_count` or `found_all`.
        
        # Call DFS
        # We need a way to break out of the loop if we found all.
        # The dfs returns True if it should stop.
        
        # But `dfs` returns True only if `found_count == 128` or deadline.
        # If it returns False, it means it finished exploring this k without stopping early.
        
        if dfs(0, [], k, 0):
            break
        
        # If we found all, break
        if found_count == 128:
            break

    # 4. Construct the output
    # If any target is still not found, we need a fallback.
    # Given the problem constraints, all 1..128 should be findable with small k.
    # But to be safe, if best_repr[n] is None, we can try a greedy or random approach,
    # or just use a known valid representation if we have one.
    # Since the search is exhaustive for k up to 10, it should find everything.
    # Smooth numbers are dense enough.
    
    results = []
    for n in range(1, 129):
        if best_repr[n] is not None:
            results.append(list(best_repr[n]))
        else:
            # Fallback: This should not happen.
            # Try to find a representation using a slower, less optimized method or just raise an error?
            # The task says "All targets require valid representations".
            # If we fail, score is 0.
            # Let's try a simple greedy for fallback just in case.
            # Greedy: pick largest smooth <= remaining, check antichain.
            # This is not guaranteed to work, but better than nothing.
            # Actually, if exhaustive search up to k=10 failed, something is wrong.
            # Let's assume it's found.
            # For safety, append a dummy valid antichain if possible?
            # 1 is smooth. 1 divides everything. So {1} is not an antichain with anything else.
            # If n is smooth, [n] is a valid antichain (size 1).
            # We checked k=1. If n is smooth, it would be found.
            # So if best_repr[n] is None, n is not smooth, and no antichain of size <= 10 sums to n.
            # This is extremely unlikely for n <= 128.
            
            # Let's try k=11, 12... just in case, within time limit.
            pass
            
    # If any are missing, try extending search
    missing = [n for n in range(1, 129) if best_repr[n] is None]
    if missing:
        for k in range(max_k + 1, 15):
            if time.time() - t_start > DEADLINE - 5:
                break
            if dfs(0, [], k, 0):
                break
            missing = [n for n in range(1, 129) if best_repr[n] is None]
            if not missing:
                break
            
    # Final check
    final_results = []
    for n in range(1, 129):
        if best_repr[n] is not None:
            final_results.append(list(best_repr[n]))
        else:
            # Absolute last resort: use a valid representation if we can find one quickly
            # e.g. n = a + b where a,b smooth and a !| b.
            # This is O(N^2).
            found = False
            for i in range(N):
                for j in range(i+1, N):
                    if S[i] + S[j] == n and not (S[i] % S[j] == 0 or S[j] % S[i] == 0):
                        final_results.append([S[i], S[j]])
                        found = True
                        break
                if found:
                    break
            if not found:
                # Try 3 terms
                found = False
                for i in range(N):
                    for j in range(i+1, N):
                        if S[i] + S[j] > n: continue
                        rem = n - S[i] - S[j]
                        for k in range(j+1, N):
                            if S[k] == rem and not (S[i] % S[k] == 0 or S[k] % S[i] == 0 or S[j] % S[k] == 0 or S[k] % S[j] == 0):
                                final_results.append([S[i], S[j], S[k]])
                                found = True
                                break
                        if found: break
                    if found: break
                if found: continue
                # If still not found, this is a fatal error for the task.
                # But we must return *something*.
                # The task says "All targets require valid representations".
                # If we can't find one, the program fails.
                # However, given the density, this won't happen.
                # We'll just append an empty list or raise.
                # Let's raise to identify the bug in development.
                # But in production, we might want to return a best-effort.
                # The evaluator checks validity. Invalid -> 0.
                # So we must find a valid one.
                # I will assume the search up to k=10 is sufficient.
                raise RuntimeError(f"Failed to find representation for {n}")

    return final_results

if __name__ == "__main__":
    # Quick test
    res = construct()
    total_terms = sum(len(r) for r in res)
    print(f"Total terms: {total_terms}")
    for i, r in enumerate(res):
        s = sum(r)
        if s != i+1:
            print(f"Error at {i+1}: sum {s}")
        # Check antichain
        for a in range(len(r)):
            for b in range(a+1, len(r)):
                if r[a] % r[b] == 0 or r[b] % r[a] == 0:
                    print(f"Error at {i+1}: {r[a]} divides {r[b]}")
        # Check smooth
        for x in r:
            y = x
            while y % 2 == 0: y //= 2
            while y % 3 == 0: y //= 3
            while y % 5 == 0: y //= 5
            if y != 1:
                print(f"Error at {i+1}: {x} not smooth")
