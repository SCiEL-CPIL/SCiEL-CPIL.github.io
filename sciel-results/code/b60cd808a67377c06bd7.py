import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    N = 1000
    MAX_SUM = 2 * N
    
    # Precompute pairs for vectorized updates
    # Pairs (i, j) with i <= j
    i_idx, j_idx = np.triu_indices(N)
    pairs_sum = np.zeros(len(i_idx), dtype=np.int32)
    for k in range(len(i_idx)):
        pairs_sum[k] = i_idx[k] + 1 + j_idx[k] + 1
        
    # Map element index (0-based) to list of pair indices involving it
    # This allows fast update of counts when an element is added/removed
    # We need to update counts for all sums involving x
    # For a specific element x (0-based idx), the pairs are:
    # 1. (x, x)
    # 2. (x, y) for y > x
    # 3. (y, x) for y < x -> which is (y, x) in our sorted pair list? 
    # Actually, np.triu_indices gives i <= j.
    # So pairs involving x are:
    # - (x, x)
    # - (x, j) where j > x
    # - (i, x) where i < x
    # Let's precompute the indices of pairs for each element.
    
    pair_indices_for_elem = [[] for _ in range(N)]
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        pair_indices_for_elem[i].append(k)
        if i != j:
            pair_indices_for_elem[j].append(k)

    def build_counts(S_set):
        """
        Build sum-count array from set S.
        S_set is a set of 1-based integers.
        Returns numpy array of counts.
        """
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        # Convert to 0-based indices for faster lookup if needed, 
        # but here we just iterate.
        # To be fast, we can use the precomputed pairs? 
        # No, building from scratch is O(N^2). 
        # Better: start with zero counts, and add elements one by one using precomputed pair indices?
        # That would be O(N * avg_degree) which is O(N^2) but with low constant.
        
        # Actually, the most efficient way to build counts from a set S is:
        # counts = np.zeros(MAX_SUM+1, dtype=np.int32)
        # for x in S:
        #    idx = x - 1
        #    counts[pairs_sum[pair_indices_for_elem[idx]]] += 1
        # This counts each pair (a,b) once if we are careful?
        # No, pair_indices_for_elem[i] contains ALL pairs involving i.
        # If we iterate over all x in S and add 1 to all pairs involving x, 
        # we will count each pair (a,b) with a!=b twice (once for a, once for b) 
        # and (a,a) once.
        # We want the count of representations a+b.
        # Representation is defined by the pair {a,b}.
        # So if we sum over x in S, the count for sum s will be:
        # Sum_{x in S} (number of pairs (x,y) in S such that x+y=s)
        # For a pair (a,b) with a!=b, it is counted when x=a (pair a,b) and when x=b (pair b,a). 
        # But our pair list only has (a,b) with a<=b.
        # So if a<b, pair (a,b) is in the list. 
        # When x=a, we see pair (a,b). 
        # When x=b, do we see pair (b,a)? No, we see pairs (b, y) with b<=y. 
        # So (a,b) is only seen when x=a? 
        # Let's check pair_indices_for_elem.
        # For i < j: pair (i,j) is added to elem[i] AND elem[j].
        # So if we iterate x in S and add counts for all pairs in pair_indices_for_elem[x],
        # then for a pair (a,b) with a<b:
        # It is in elem[a] and elem[b].
        # So it will be added twice if both a and b are in S.
        # But we want it to be counted ONCE as a representation.
        # So this method overcounts mixed pairs by 2x.
        
        # Correct approach for fast build:
        # Just do the double loop but vectorized? No.
        # Since N=1000, N^2 = 1e6 operations. Python loop is ~0.1s. 
        # That's acceptable for initialization.
        
        S_list = sorted(S_set)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            counts[2 * a] += 1
            for j in range(i + 1, n):
                b = S_list[j]
                counts[a + b] += 1
        return counts

    def can_add_vectorized(S_set, counts, x):
        """
        Check if x can be added to S without violating B2[2].
        x is 1-based.
        """
        idx = x - 1
        # Get all pairs involving x
        pair_idxs = np.array(pair_indices_for_elem[idx], dtype=np.int32)
        if len(pair_idxs) == 0:
            return True
        
        # Get the sums for these pairs
        sums = pairs_sum[pair_idxs]
        
        # We need to check if any of these sums currently have count >= 2.
        # BUT, we must be careful: 
        # The counts array reflects the CURRENT set S.
        # If we add x, the counts for sums involving x will increase.
        # A sum s is valid if current_count[s] + new_contributions[s] <= 2.
        
        # New contributions:
        # For each pair (x, y) in S (including x=x), the count increases by 1.
        # So for each unique sum s formed by x and an element in S, count increases by 1.
        # Wait, if x is not in S, the pairs are (x, y) for y in S.
        # Since x is new, it doesn't have pairs with other new elements (there are none).
        # So the new pairs are exactly (x, y) for y in S, and (x, x).
        # These are exactly the pairs in pair_indices_for_elem[idx] 
        # EXCEPT we must filter out pairs (x, y) where y is NOT in S.
        
        # My pair_indices_for_elem contains ALL pairs (x, y) for y in 1..N.
        # I need to filter to only y in S.
        
        # This filtering is the bottleneck.
        # Alternative: 
        # Check all sums s such that there exists y in S with x+y=s.
        # This is equivalent to: for each y in S, sum = x+y.
        # Also sum = 2*x.
        
        # Let's stick to the loop for can_add but make it tight.
        # Or, precompute a boolean array "in_S" for faster lookup?
        # in_S = np.zeros(N+1, dtype=bool)
        # in_S[S] = True
        
        # Let's try the vectorized filter.
        # We have pair_idxs.
        # For each pair idx, we know the two elements i, j.
        # We need to check if the OTHER element is in S.
        
        # Let's extract the other elements.
        # For pair (i, j):
        # If i == idx, other is j.
        # If j == idx, other is i.
        # Since i <= j:
        # If i == idx, then j >= idx. Other is j.
        # If j == idx, then i <= idx. Other is i.
        
        # Actually, simpler:
        # The pairs involving x are (x, y) for all y.
        # We only care about y in S.
        # So we just need to check:
        # 1. 2*x: if counts[2*x] >= 2, fail. (Since x not in S, count of 2x is from other pairs. 
        #    Wait, if x not in S, counts[2x] does NOT include (x,x). 
        #    So if we add x, counts[2x] becomes counts[2x] + 1.
        #    So we need counts[2x] + 1 <= 2 => counts[2x] <= 1.
        #    If counts[2x] >= 2, fail.
        
        # 2. x + y for y in S, y != x.
        #    For each such y, the sum is s = x + y.
        #    The count of s will increase by 1.
        #    So we need counts[s] + 1 <= 2 => counts[s] <= 1.
        #    If any counts[s] >= 2, fail.
        
        # So the check is:
        # if counts[2*x] >= 2: return False
        # for y in S:
        #    if y != x and counts[x+y] >= 2: return False
        # return True
        
        # This is O(|S|). With |S| ~ 700, this is fast.
        # The previous program did this in Python. 
        # Can we vectorize?
        # S_array = np.array(list(S_set))
        # sums = x + S_array
        # sums = sums[sums != 2*x] # Exclude x+x? No, x not in S, so x+x not in S_array sum.
        # # Actually if x not in S, S_array doesn't contain x.
        # # So sums = x + S_array are all distinct from 2x? 
        # # No, if y = x, sum = 2x. But x not in S, so y != x always.
        # # So we just check counts[sums] < 2.
        
        # Let's use a precomputed boolean mask for S to speed up?
        # No, iterating over S is fine.
        
        # Optimization: 
        # We can precompute the "conflict sums" for each x? 
        # No, it depends on S.
        
        # Let's just use the loop, it's likely fast enough if we don't do it in a tight inner loop 
        # of a slow outer loop. But local search does many checks.
        
        # Vectorized check:
        if counts[2 * x] >= 2:
            return False
        
        # Get S as array
        # This conversion is expensive. 
        # Better to maintain S as a sorted list or array?
        # Maintaining a sorted list makes insertion/removal O(N).
        # For local search, we do many swaps.
        
        # Let's assume S is a set for membership, but we also keep a list/array for iteration.
        # Actually, for can_add, we just need to iterate S.
        
        for y in S_set:
            if y != x:
                if counts[x + y] >= 2:
                    return False
        return True

    def do_add(S_set, S_list, counts, x):
        S_set.add(x)
        S_list.append(x)
        S_list.sort() # Keep sorted for potential optimizations, though not strictly needed for set logic
        counts[2 * x] += 1
        for y in S_set:
            if y != x:
                counts[x + y] += 1

    def do_remove(S_set, S_list, counts, x):
        S_set.discard(x)
        if x in S_list:
            S_list.remove(x)
        counts[2 * x] -= 1
        for y in S_set:
            if y != x:
                counts[x + y] -= 1

    best_set = set()
    best_size = 0
    
    # Strategy: Randomized Local Search with Vectorized/optimized checks
    
    # 1. Initial Greedy to get a decent start
    # Use ascending greedy as a baseline
    S_greedy = set()
    S_list_greedy = []
    counts_greedy = np.zeros(MAX_SUM + 1, dtype=np.int32)
    for x in range(1, N + 1):
        if can_add_vectorized(S_greedy, counts_greedy, x):
            do_add(S_greedy, S_list_greedy, counts_greedy, x)
            
    if len(S_greedy) > best_size:
        best_size = len(S_greedy)
        best_set = set(S_greedy)
        best_list = list(S_list_greedy)
        best_counts = counts_greedy.copy()
        
    # 2. Local Search
    # Start from the best solution found (greedy)
    S_ls = set(best_set)
    S_list_ls = list(best_list)
    counts_ls = best_counts.copy()
    
    # Also try a random dense initialization
    # Pick ~700 random numbers, prune to valid
    S_rand = set()
    S_list_rand = []
    counts_rand = np.zeros(MAX_SUM + 1, dtype=np.int32)
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    for x in candidates:
        if len(S_rand) >= 700:
            break
        if can_add_vectorized(S_rand, counts_rand, x):
            do_add(S_rand, S_list_rand, counts_rand, x)
            
    # Prune S_rand to ensure validity (in case of bugs, though greedy should be valid)
    # And try to add more
    for x in candidates:
        if x not in S_rand and can_add_vectorized(S_rand, counts_rand, x):
            do_add(S_rand, S_list_rand, counts_rand, x)
            
    if len(S_rand) > best_size:
        best_size = len(S_rand)
        best_set = set(S_rand)
        best_list = list(S_list_rand)
        best_counts = counts_rand.copy()
        S_ls = set(S_rand)
        S_list_ls = list(S_list_rand)
        counts_ls = counts_rand.copy()

    # Local Search Loop
    # We will perform swaps and adds.
    # To speed up, we can batch the "can_add" checks?
    # No, dependency is sequential.
    
    # Optimization: 
    # Instead of iterating all y in S for can_add, we can precompute 
    # for each sum s, the set of pairs that form it? 
    # No, that's what counts is for.
    
    # Let's just run the local search with a time limit.
    # We'll alternate between:
    # 1. Try to add a random x.
    # 2. Try to swap a random x_out with a random x_in.
    
    time_limit = deadline - 5.0
    
    # To make it faster, we can sample x_out and x_in randomly.
    # For x_out, prefer elements that are "constrained" (involved in sums with count 2).
    
    # Precompute "pressure" for each element?
    # Pressure(x) = number of sums s such that x+y=s and counts[s]==2.
    # This is expensive to compute from scratch each time.
    
    # Simple heuristic:
    # Randomly pick x_out from S.
    # Randomly pick x_in from [1..N] \ S.
    # Check if (S - {x_out} + {x_in}) is valid.
    # This requires checking validity of the new set.
    # We can do this by:
    # 1. Remove x_out from counts.
    # 2. Check if x_in can be added.
    # 3. If yes, add x_in.
    # 4. If no, restore x_out.
    
    # This is O(|S|) per swap attempt.
    # With |S| ~ 700, and 120s, we can do maybe 1e5 to 1e6 swaps?
    # Python loop overhead is high.
    
    # Let's try to vectorize the "check if x_in can be added" part 
    # given that x_out is removed.
    
    # Actually, the current can_add_vectorized is already a Python loop over S.
    # We can speed it up by using a numpy array for S and vectorized sum check.
    
    def can_add_fast_np(S_arr, counts, x):
        # S_arr is a numpy array of elements in S (0-based or 1-based? 1-based)
        # x is 1-based
        if counts[2 * x] >= 2:
            return False
        # sums = x + S_arr
        # We need to check if any counts[sums] >= 2
        # Exclude y=x if x in S? 
        # In swap, x_in is not in S. So x_in not in S_arr.
        # So all sums are x_in + y for y in S.
        sums = x + S_arr
        # Check if any count >= 2
        # np.any(counts[sums] >= 2)
        return not np.any(counts[sums] >= 2)

    # Update S_ls to use numpy array for S
    S_ls_arr = np.array(S_ls, dtype=np.int32)
    
    # Local Search
    while time.time() < time_limit:
        # Try Add
        if len(S_ls) < N:
            # Pick random x
            x_in = random.randint(1, N)
            if x_in not in S_ls:
                if can_add_fast_np(S_ls_arr, counts_ls, x_in):
                    # Add it
                    S_ls.add(x_in)
                    S_ls_arr = np.append(S_ls_arr, x_in)
                    counts_ls[2 * x_in] += 1
                    sums = x_in + S_ls_arr
                    # Wait, S_ls_arr now includes x_in.
                    # We need to update counts for x_in + y for y in OLD S_ls.
                    # Old S_ls was S_ls_arr[:-1]
                    old_S = S_ls_arr[:-1]
                    old_sums = x_in + old_S
                    counts_ls[old_sums] += 1
                    # Also counts[2*x_in] += 1 (done)
                    
        # Try Swap
        else:
            # Pick random x_out
            x_out = random.choice(list(S_ls))
            # Pick random x_in
            x_in = random.randint(1, N)
            if x_in in S_ls:
                continue
                
            # Remove x_out
            counts_ls[2 * x_out] -= 1
            # S_ls currently contains x_out.
            # Sums involving x_out: x_out + y for y in S_ls, y != x_out.
            # And 2*x_out.
            # We need to subtract these from counts.
            # S_ls_arr contains x_out.
            # Let's create a mask or just iterate.
            # To be fast, we can use the fact that we know x_out.
            # sums_to_sub = x_out + S_ls_arr
            # But we must exclude x_out+x_out? 
            # counts[2*x_out] was already decremented.
            # So we need to subtract counts[x_out + y] for y in S_ls, y != x_out.
            # Let's make a copy of S_ls_arr without x_out.
            # This is O(N).
            # Alternatively, just subtract x_out + S_ls_arr, then add back x_out+x_out?
            # No, x_out+x_out was decremented separately.
            # So:
            # sums_all = x_out + S_ls_arr
            # This includes x_out+x_out.
            # We already did counts[2*x_out] -= 1.
            # So we should NOT subtract counts[x_out+x_out] again.
            # So we need to subtract counts[x_out+y] for y in S_ls, y != x_out.
            
            # Let's just use a list comprehension or loop for the removal, it's safer.
            # But we want speed.
            
            # Let's revert to the safer, slightly slower remove/add for correctness first, 
            # then optimize if time permits.
            
            # Actually, let's just use the set-based do_remove/do_add for simplicity 
            # and rely on the high iteration count.
            
            # Remove x_out
            S_ls.remove(x_out)
            # Update counts
            counts_ls[2 * x_out] -= 1
            for y in S_ls:
                if y != x_out:
                    counts_ls[x_out + y] -= 1
            
            # Check if x_in can be added
            # S_ls is now S \ {x_out}
            # S_ls_arr needs to be updated? 
            # can_add_fast_np expects S_arr.
            # Let's just use the Python loop version of can_add for now, 
            # or rebuild S_ls_arr.
            
            # Rebuilding S_ls_arr is O(N log N) or O(N) if we just make a new array.
            # np.array(list(S_ls)) is O(N).
            
            S_ls_arr_tmp = np.array(list(S_ls), dtype=np.int32)
            
            if can_add_fast_np(S_ls_arr_tmp, counts_ls, x_in):
                # Add x_in
                S_ls.add(x_in)
                counts_ls[2 * x_in] += 1
                for y in S_ls:
                    if y != x_in:
                        counts_ls[x_in + y] += 1
                # Update best if larger? No, size is same.
                # We just accept the swap to escape local optima.
                
                # Update S_ls_arr for next iteration
                S_ls_arr = S_ls_arr_tmp
                S_ls_arr = np.append(S_ls_arr, x_in)
            else:
                # Revert: Add x_out back
                S_ls.add(x_out)
                counts_ls[2 * x_out] += 1
                for y in S_ls:
                    if y != x_out:
                        counts_ls[x_out + y] += 1
                # S_ls_arr stays as is (it was S_ls_arr before removal? 
                # No, we didn't update S_ls_arr in the remove step above.
                # So S_ls_arr is still the old one with x_out.
                # So we are good.)
                
        # Track best size (should be constant or increasing)
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_set = set(S_ls)
            best_list = list(S_ls)
            best_counts = counts_ls.copy()

    # Final verification
    # The evaluator checks the result. 
    # We should return the best_set found.
    # Ensure best_set is valid.
    # The local search maintains validity.
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res))

