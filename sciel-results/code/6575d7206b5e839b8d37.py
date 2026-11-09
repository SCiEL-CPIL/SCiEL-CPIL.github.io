import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    limit = 10000

    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, math.isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))

    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)

    if n < 2:
        return primes

    # dp[j][i] = length of longest non-decreasing-gap sequence ending at primes[i], primes[j] (i < j)
    # We use a 2D list for dp and parent to allow easy backtracking.
    # dp[j][i] is only defined for j > i.
    # To save memory and time, we can use lists of length n, where dp[j][i] is stored.
    # However, we only need dp[i][k] to compute dp[j][i].
    # So we can process i in increasing order? No, dp[j][i] depends on dp[i][k].
    # This means we need the entire row dp[i] to compute contributions to dp[j] for all j > i.
    # So we must store all rows or process in a way that keeps previous rows.
    # Since N ~ 1229, N^2 ~ 1.5M. Storing 1.5M integers is fine.
    
    # dp[j][i]
    dp = [[0] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]
    
    # Initialize length 2 sequences
    best_len = 2
    best_i = 0
    best_j = 1
    
    for j in range(1, n):
        for i in range(j):
            dp[j][i] = 2
            parent[j][i] = -1

    # Iterate i from 1 to n-2
    # For each i, we have dp[i][k] for all k < i.
    # We want to update dp[j][i] for all j > i.
    # dp[j][i] = 1 + max{ dp[i][k] : k < i and primes[i] - primes[k] <= primes[j] - primes[i] }
    
    for i in range(1, n - 1):
        if time.time() > deadline:
            break
            
        # We need max of dp[i][k] for k in valid range.
        # Valid range for a given j: primes[k] >= 2*primes[i] - primes[j]
        # Let's iterate j from i+1 to n-1.
        # As j increases, threshold decreases, so valid k range expands to the left.
        
        # We maintain a pointer 'k_left' which is the smallest valid k index.
        # Initially for j=i+1, threshold is 2*p[i] - p[i+1].
        # We can start k_left at i-1 and move left?
        # Actually, let's just iterate k from i-1 down to 0 to find the initial valid range?
        # No, we can use the two-pointer logic.
        
        # Let's precompute nothing, just use the expanding window.
        # For fixed i, let's find the max dp[i][k] for k in [L, i-1].
        # We'll maintain `max_val` and `argmax_k` for the current valid window.
        
        # Start with an empty window? No, the window is [L_j, i-1].
        # For the first j (j=i+1), we need to find L_{i+1}.
        # Then for subsequent j, L decreases.
        
        # Let's initialize the window with the most restrictive case (smallest j).
        # j = i+1. Threshold T = 2*p[i] - p[i+1].
        # We need k such that p[k] >= T.
        # Since p is sorted, these are the largest k's.
        
        # Let's start with k_left = i-1.
        # Check if p[i-1] >= T. If not, we need to move k_left left?
        # Wait, if p[i-1] < T, then NO k in [0, i-1] is valid?
        # If p[i-1] < T, then since p[k] <= p[i-1] for k < i-1, no k is valid.
        # In that case, max_val remains -infinity (or 0, but we want to know if any exist).
        
        # Let's restart the logic for clarity and safety, using binary search for L_j is O(log N).
        # Total O(N^2 log N) was ~1s. O(N^2) should be faster.
        
        # Let's implement the two-pointer correctly.
        # For fixed i:
        # We want to compute dp[j][i] for j = i+1 ... n-1.
        # Let valid_k be the set of k < i such that p[k] >= 2*p[i] - p[j].
        # Let L_j be min(valid_k).
        # We want max(dp[i][k]) for k in [L_j, i-1].
        
        # We can iterate j from i+1 to n-1.
        # L_j is non-increasing.
        # We can maintain a pointer `left` starting at i-1.
        # But we need to know the max in the range [left, i-1].
        # If we move `left` to `left-1`, we include one more element.
        # We can maintain `current_max` and `argmax`.
        
        # Initialization:
        # For j = i+1, find L.
        # Let's just find L for the first j using binary search, then adjust.
        # Or simpler: 
        # We know that as j increases, the condition becomes looser.
        # So the set of valid k's only grows.
        
        # Let's start with the set of valid k's for the LARGEST j (j=n-1).
        # Threshold is smallest. So all k might be valid.
        # Then as we go down to j=i+1, the set shrinks.
        # This direction is harder to maintain max (removing elements).
        
        # So we must go j from i+1 to n-1 (increasing j, loosening constraint).
        
        # Step 1: Determine initial valid range for j=i+1.
        # Threshold T_0 = 2*p[i] - p[i+1].
        # Find smallest k such that p[k] >= T_0.
        # Let this be `left`.
        # If no such k, then for this j, no predecessor exists (max_val = -1).
        # If such k exists, valid range is [left, i-1].
        # Compute max_val and argmax for this range.
        
        # To avoid O(i) initialization for each i, we can just scan from i-1 down to 0 
        # until we find p[k] < T_0.
        # In the worst case (small gap), this is O(i). Total O(N^2).
        
        left = i - 1
        # Move left to the first invalid k (from the right)
        # Actually, we want the smallest valid k.
        # p is sorted. Valid k are those with p[k] >= T.
        # So we want the first k (smallest index) in [0, i-1] with p[k] >= T.
        # Since p is increasing, p[k] >= T holds for k >= some index.
        # So we want the smallest such index.
        
        # Let's just use binary search for the initial `left` for each i?
        # That would be O(N log N) for initialization.
        # Then the loop is O(N) per i. Total O(N^2).
        
        # Let's do that.
        
        # For fixed i:
        # 1. Find left_init for j=i+1.
        # 2. Initialize max_val and argmax for range [left_init, i-1].
        #    This takes O(i) time.
        # 3. Iterate j from i+1 to n-1.
        #    For each j, update left (move left if threshold allows).
        #    Update max_val and argmax if new elements are included.
        #    Update dp[j][i].
        
        # Wait, step 2 is O(i). Step 3 is O(N). Total O(N) per i. Total O(N^2).
        # This is acceptable.
        
        # Optimization: We can precompute suffix maxes for dp[i]?
        # suffix_max[k] = max(dp[i][k], ..., dp[i][i-1]).
        # Then for any range [L, i-1], the max is suffix_max[L].
        # But we also need the argmax.
        # We can store suffix_argmax[L].
        # Building suffix arrays takes O(i).
        # Then querying is O(1).
        # Finding L for each j is O(log N) or O(1) amortized with two pointers.
        
        # Let's build suffix max and argmax for dp[i].
        suffix_max = [0] * i
        suffix_argmax = [0] * i
        
        if i > 0:
            suffix_max[i-1] = dp[i][i-1]
            suffix_argmax[i-1] = i-1
            for k in range(i-2, -1, -1):
                if dp[i][k] > suffix_max[k+1]:
                    suffix_max[k] = dp[i][k]
                    suffix_argmax[k] = k
                else:
                    suffix_max[k] = suffix_max[k+1]
                    suffix_argmax[k] = suffix_argmax[k+1]
            
            # Now iterate j from i+1 to n-1
            # Use two-pointer for L_j.
            # L_j is the smallest k in [0, i-1] such that p[k] >= 2*p[i] - p[j].
            
            # For j=i+1, find L.
            # Since p is sorted, we can start L from i-1 and move left?
            # No, L is the lower bound of the valid indices.
            # As j increases, threshold decreases, so L decreases (moves left).
            
            # Let's maintain `left_ptr`.
            # Initially, for j=i+1, we need to find L.
            # We can just binary search for the first j, then adjust.
            # Or, since we have suffix arrays, we just need the index L.
            
            # Let's use binary search for L for each j? O(N log N) per i. Total O(N^2 log N).
            # We want O(N^2).
            
            # Two pointer:
            # left_ptr starts at i-1? No, L is the smallest valid index.
            # If threshold is very high, L might be i-1 or higher (invalid).
            # If threshold is low, L might be 0.
            
            # Let's start left_ptr at i-1.
            # For j=i+1:
            #   T = 2*p[i] - p[i+1]
            #   While left_ptr > 0 and p[left_ptr-1] >= T:
            #       left_ptr -= 1
            #   Now left_ptr is the smallest index such that p[left_ptr] >= T?
            #   Wait. If p[left_ptr-1] >= T, then left_ptr-1 is also valid.
            #   We want the SMALLEST valid index.
            #   So we should move left_ptr as far left as possible while p[left_ptr] >= T.
            
            # Let's refine:
            # valid k are those with p[k] >= T.
            # Since p is sorted, this is a suffix [L, i-1].
            # We want L.
            # As j increases, T decreases, so L can only decrease.
            
            left_ptr = i - 1
            # Note: It's possible that no k is valid for j=i+1.
            # In that case, L would be i (out of bounds).
            # We can handle this by checking if p[i-1] < T.
            
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                T = 2 * primes[i] - primes[j]
                
                # Move left_ptr left if possible
                # We want the smallest index k in [0, i-1] such that p[k] >= T.
                # Current left_ptr is a valid candidate from the previous step (or initial).
                # Actually, we should start left_ptr at i-1 for each i?
                # No, for fixed i, as j increases, T decreases.
                # So the valid range expands to the left.
                # So L_j is non-increasing.
                
                # Initialize left_ptr for the first j?
                # We can just start left_ptr at i-1 and move left.
                # But we must ensure p[left_ptr] >= T.
                # If p[i-1] < T, then no k is valid.
                
                # Let's restart left_ptr logic for each i to be safe and simple, 
                # leveraging the fact that we can just scan left from i-1 for the first j?
                # No, that's O(i).
                
                # Let's use the property that L_j is non-increasing.
                # We maintain `left_ptr` which is the current L.
                # Initially, for j=i+1, we don't know L.
                # We can find it by moving left_ptr from i-1 downwards while p[left_ptr-1] >= T.
                # But we need to start with a valid left_ptr.
                # If p[i-1] < T, then left_ptr should be i (meaning empty range).
                
                # Let's use a variable `current_left` which is the smallest valid index for the current j.
                # We'll update it as j increases.
                
                # To avoid complex initialization, let's just use binary search for L for each j.
                # O(log N) per j. Total O(N^2 log N).
                # Given N=1229, N^2 log N ~ 1.5M * 10 = 15M ops. Python can do ~10M-50M simple ops/sec.
                # So ~1s. The current program is 1.32 score (likely ~1s).
                # We want to be faster.
                
                # Let's try the two-pointer with proper initialization.
                # For fixed i:
                #   left_ptr = i - 1
                #   # Check if p[i-1] is valid for j=i+1?
                #   # Actually, we can just iterate j and adjust left_ptr.
                #   # But we need to know if the range is empty.
                
                # Let's do this:
                #   left_ptr = i - 1
                #   for j in range(i+1, n):
                #       T = 2*p[i] - p[j]
                #       # Move left_ptr left as much as possible
                #       while left_ptr > 0 and primes[left_ptr - 1] >= T:
                #           left_ptr -= 1
                #       
                #       # Now we need to check if primes[left_ptr] >= T.
                #       # If not, then no k is valid?
                #       # Wait, if primes[i-1] < T, then left_ptr stays at i-1?
                #       # No, if primes[i-1] < T, then the while loop doesn't run (if left_ptr starts at i-1).
                #       # But we need to know if primes[i-1] is valid.
                #       # If primes[i-1] < T, then NO k in [0, i-1] is valid.
                #       # In that case, max_val should be -1 (or 0, but we want to distinguish).
                
                # Let's add a check:
                #   if primes[i-1] < T:
                #       # No valid k
                #       # But wait, if j increases, T decreases.
                #       # So eventually T might become small enough.
                #       # So we can't just skip.
                
                # Let's initialize left_ptr = i - 1.
                # And maintain a flag `has_valid`.
                # Actually, simpler:
                #   left_ptr = i - 1
                #   for j in range(i+1, n):
                #       T = 2*p[i] - p[j]
                #       # Expand left_ptr to the left
                #       while left_ptr > 0 and primes[left_ptr - 1] >= T:
                #           left_ptr -= 1
                #       
                #       # Check if the current left_ptr is valid
                #       if primes[left_ptr] >= T:
                #           # Valid range is [left_ptr, i-1]
                #           max_val = suffix_max[left_ptr]
                #           argmax = suffix_argmax[left_ptr]
                #           new_len = 1 + max_val
                #           if new_len > dp[j][i]:
                #               dp[j][i] = new_len
                #               parent[j][i] = argmax
                #               if new_len > best_len:
                #                   best_len = new_len
                #                   best_i = i
                #                   best_j = j
                #       else:
                #           # No valid k in [0, i-1]
                #           # This can happen if T is very large.
                #           # But as j increases, T decreases, so this state might become valid later.
                #           pass

    # Reconstruct
    if best_len <= 2:
        values = [primes[0]]
        for p in primes[1:]:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
        return values

    seq = [primes[best_j], primes[best_i]]
    i = best_i
    j = best_j
    while parent[j][i] != -1:
        k = parent[j][i]
        seq.append(primes[k])
        j = i
        i = k

    seq.reverse()
    return seq
