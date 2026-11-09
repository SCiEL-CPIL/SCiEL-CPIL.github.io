import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Generate 2,3,5-smooth numbers <= 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    n_smooth = len(smooth)
    
    # Precompute divisibility matrix
    # divs[i][j] is True if smooth[i] divides smooth[j]
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                divs[i][j] = True
    
    # solutions[sum] = list of terms
    solutions = [None] * 129
    solved_count = 0
    
    # Check if all solved
    def all_solved():
        return all(solutions[i] is not None for i in range(1, 129))
    
    # Function to check if a set of indices forms an antichain
    def is_antichain(indices):
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i = indices[i]
                idx_j = indices[j]
                if divs[idx_i][idx_j] or divs[idx_j][idx_i]:
                    return False
        return True

    # Iterate by number of terms k
    for k in range(1, 31):
        if time.time() > deadline:
            break
        if all_solved():
            break
            
        # Generate combinations of k indices
        # We can prune combinations that are too large in sum?
        # itertools.combinations doesn't support sum pruning directly.
        # But we can check sum quickly.
        
        for comb in itertools.combinations(range(n_smooth), k):
            # Calculate sum
            s = sum(smooth[idx] for idx in comb)
            if s > 128:
                continue
            if s < 1:
                continue
                
            # If this sum is already solved with fewer terms (which is always true if we iterate k increasing and only set if None), skip
            if solutions[s] is not None:
                continue
                
            # Check antichain property
            if is_antichain(comb):
                solutions[s] = [smooth[idx] for idx in comb]
                solved_count += 1
                
    # Fallback for any unsolved targets (should not happen if k goes high enough, but just in case)
    # If some are unsolved, we might need to use a backtracking search for those specific targets
    # But given the structure, k up to 10 should cover everything.
    # Let's add a simple backtracking for any remaining unsolved just to be safe.
    for n in range(1, 129):
        if solutions[n] is None:
            if time.time() > deadline:
                break
            # Simple backtracking for this specific n
            # We want to find ANY antichain, preferably small.
            # Since we already tried small k globally, we can try larger k or just any valid one.
            # Actually, if the global search didn't find it, it might be because of time or logic error.
            # But the global search is exhaustive for k=1..30.
            # If it's still None, it means no antichain of size <= 30 exists? Unlikely.
            # Or maybe my `is_antichain` or sum logic was too strict?
            # Wait, `itertools.combinations` covers all subsets of size k.
            # So if it's None, no such antichain exists with sum n and size k.
            # If it's None for all k, then no representation exists?
            # The problem states "All targets require valid representations".
            # So there must be a representation.
            
            # Let's try a specific backtracking search for n
            def find_antichain_bt(target, max_k=30):
                # Backtracking to find an antichain summing to target
                # We can reuse the global search logic but targeted.
                # Actually, the global search should have found it.
                # If not, maybe I should just return a dummy? No, must be valid.
                # Let's try a DFS that doesn't care about minimal k, just finds one.
                
                def dfs(start_idx, remaining_sum, current_path):
                    if remaining_sum == 0:
                        return current_path
                    if start_idx >= n_smooth:
                        return None
                    if time.time() > deadline:
                        return None
                        
                    # Option 1: Skip this smooth number
                    res = dfs(start_idx + 1, remaining_sum, current_path)
                    if res:
                        return res
                        
                    # Option 2: Include this smooth number
                    val = smooth[start_idx]
                    if val <= remaining_sum:
                        # Check antichain with current_path
                        valid = True
                        for prev_idx in current_path:
                            if divs[start_idx][prev_idx] or divs[prev_idx][start_idx]:
                                valid = False
                                break
                        if valid:
                            res = dfs(start_idx + 1, remaining_sum - val, current_path + [start_idx])
                            if res:
                                return res
                    return None

                res_indices = dfs(0, n, [])
                if res_indices:
                    return [smooth[i] for i in res_indices]
                return None
            
            sol = find_antichain_bt(n)
            if sol:
                solutions[n] = sol
            else:
                # Last resort fallback (should not happen)
                # Just pick the number itself if it's smooth?
                if n in smooth:
                    solutions[n] = [n]
                else:
                    # Try to find any combination, even if not antichain? No, must be antichain.
                    # If we can't find one, we fail. But problem guarantees existence.
                    pass

    return solutions[1:129]
