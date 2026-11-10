import time
import math
from collections import Counter

def construct(n):
    start_time = time.time()
    deadline = start_time + 110

    # Step 1: Generate classic superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n <= 5:
        return "".join(map(str, word))

    # Step 2: Precompute permutation ranking for efficient validity checks
    # For n=7, we have 5040 permutations. We map each permutation tuple to a rank 0..5039.
    # This allows us to use a bytearray for seen permutations, which is much faster than a set of tuples.
    
    # We need a bijection between permutations of 1..n and integers 0..n!-1.
    # We can use the lexicographic rank (Lehmer code).
    
    # Precompute factorials
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i

    def perm_rank(p):
        """
        Compute the lexicographic rank of permutation p (tuple of 1..n) in 0..n!-1.
        p is a tuple of integers 1..n.
        """
        # To speed up, we can precompute a mapping for n=7 if needed, but computing on the fly is fine
        # if we do it once. However, we need to do it for every window in every check.
        # Let's see if we can optimize this.
        
        # Alternative: Since n is small (<=7), we can precompute a dictionary mapping all permutations to ranks.
        # But 5040 permutations is small.
        # Actually, we can just use a global cache or pass a precomputed dict.
        
        # Let's stick to a simple implementation first.
        used = [False] * (n + 1)
        rank = 0
        for i in range(n):
            count = 0
            for j in range(1, p[i]):
                if not used[j]:
                    count += 1
            rank += count * fact[n - 1 - i]
            used[p[i]] = True
        return rank

    # Optimization: Precompute ranks for all permutations of 1..n
    # This is 5040 entries for n=7, which is trivial.
    # We can build a dict: perm_tuple -> rank
    # But we also need to go from index to perm? No, just rank is enough for seen tracking.
    # We need to compute rank for windows in the word.
    
    # Let's precompute a lookup table for ranks if n is small?
    # Actually, the function perm_rank is O(n^2). For n=7, that's 49 ops.
    # We have ~10000 windows. 10000 * 49 = 490k ops. This is fast.
    # But in the greedy deletion loop, we recompute ranks for windows repeatedly.
    # We should maintain an array of ranks for the current windows.
    
    target_perms = math.factorial(n)
    
    # Initialize rank array and seen counts
    L = len(word)
    window_ranks = []
    for i in range(L - n + 1):
        window_ranks.append(perm_rank(tuple(word[i:i+n])))
    
    # seen_counts: array of size target_perms, storing count of each permutation
    seen_counts = [0] * target_perms
    for r in window_ranks:
        seen_counts[r] += 1
    
    # unique_count: number of permutations with count > 0
    unique_count = sum(1 for c in seen_counts if c > 0)
    
    # Verify initial validity
    if unique_count != target_perms:
        # Should not happen for classic superperm, but just in case
        pass

    # Step 3: Greedy Deletion with efficient rank tracking
    # We try to delete each position i.
    # If we delete word[i]:
    # 1. Windows affected:
    #    - Windows starting at k in [i-n+1, i] are removed (their ranks are window_ranks[k])
    #    - New windows are formed by joining word[k:i] and word[i+1:k+n] for k in [i-n+1, i-1]
    #      These new windows need their ranks computed.
    
    # We iterate i from 0 to L-1.
    # If deletion is valid, we perform it and restart.
    
    # To avoid O(L^2) recomputation of all ranks, we can update the rank array incrementally?
    # No, deletion shifts all subsequent windows. It's easier to just recompute the affected part.
    # But recomputing the whole rank array is O(L*n^2).
    # If we do this for every i, it's O(L^2 * n^2).
    # L ~ 10000, n=7. L^2 = 100M. 100M * 49 = 5B ops. Too slow.
    
    # We need a smarter way.
    # The current program (score 1.0) uses a Counter of tuples and set of tuples.
    # It checks validity by:
    # 1. Building set of added permutations (tuples).
    # 2. Checking if any removed permutation (tuple) has count 1 and is not in added set.
    # This is O(n) per window check, but involves tuple creation/hashing.
    
    # Let's try to optimize the current approach or use the rank approach more carefully.
    # Actually, the rank approach allows us to use integer arrays which are faster than tuple sets.
    # But the bottleneck is the O(L) scan for each potential deletion.
    
    # Let's look at the "Inspiration 2" strategy:
    # It uses a bytearray for seen, and checks validity by:
    # "A permutation is lost if it ONLY appeared in window i (the one that shifts out)"
    # Wait, that logic was incomplete in the inspiration.
    
    # Let's refine the rank-based greedy deletion.
    # For each i:
    #   Removed ranks: R = [window_ranks[k] for k in range(max(0, i-n+1), min(i+1, L-n+1))]
    #   Added ranks: A = [perm_rank(tuple(word[k:i] + word[i+1:k+n])) for k in range(max(0, i-n+1), i)]
    #   
    #   Valid if:
    #   For all r in R:
    #     if seen_counts[r] == 1:
    #       then r must be in A.
    #   Also, if any r in A is not in R, it increases count, which is fine.
    #   If r in A is in R, it cancels out.
    
    #   We can check this quickly:
    #   1. Compute A (list of ranks).
    #   2. Create a set or counter for A? No, A is small (size < n).
    #   3. For each r in R:
    #      if seen_counts[r] == 1 and r not in A:
    #         valid = False
    #         break
    #   
    #   If valid:
    #      Update seen_counts:
    #      for r in R: seen_counts[r] -= 1
    #      for r in A: seen_counts[r] += 1
    #      Update unique_count:
    #      for r in R: if seen_counts[r] becomes 0: unique_count -= 1
    #      for r in A: if seen_counts[r] becomes 1 (was 0): unique_count += 1
    #      (Note: if r in R and r in A, net change is 0, but we must handle carefully)
    #      
    #      Actually, it's safer to just recompute unique_count if we care, but we only need to know if it drops below target.
    #      Since we check validity before applying, unique_count will remain target_perms.
    
    #   Apply deletion:
    #      word = word[:i] + word[i+1:]
    #      L -= 1
    #      Update window_ranks:
    #      We need to rebuild window_ranks or update it.
    #      Rebuilding is O(L*n^2).
    #      If we do this every time we delete, and we delete many times, it's expensive.
    #      However, deletions are relatively rare compared to failed checks?
    #      No, we scan all i.
    
    #   Alternative: Don't update window_ranks immediately. Just keep word.
    #   When checking next i, we need window_ranks.
    #   If we don't update window_ranks, we can't use it.
    #   
    #   So, if we delete, we MUST update window_ranks.
    #   Cost of deletion: O(L * n^2) to rebuild ranks.
    #   Cost of failed check: O(n^2) to compute A ranks + O(n) to check R.
    #   
    #   Total cost:
    #   Let D be number of deletions.
    #   Failed checks: ~ L * D (roughly, since we restart after each deletion).
    #   Actually, we scan from 0 to L-1. If we delete at i, we restart from 0.
    #   So for each deletion, we do ~ i failed checks + 1 successful check + 1 rebuild.
    #   Average i is L/2.
    #   So per deletion: ~ L/2 * (n^2) + L * n^2 (rebuild) ~ 1.5 * L * n^2.
    #   Total: D * 1.5 * L * n^2.
    #   If D is small (e.g., 100), L=10000, n=7.
    #   100 * 1.5 * 10000 * 49 = 73,500,000 ops. This is fast (< 1s in Python? Maybe 5-10s).
    #   If D is 1000, it's 735M ops. Might be slow.
    
    #   Let's try this approach. It should be faster than the tuple-based one due to integer ops.
    
    while time.time() < deadline:
        improved = False
        L = len(word)
        
        # Rebuild window_ranks if we modified word in previous iteration?
        # Actually, we rebuild inside the loop after deletion.
        # But we need it for the first iteration of the while loop too.
        # We initialized it before the while loop.
        
        for i in range(L):
            if time.time() > deadline:
                break
            
            # Calculate indices for removed and added windows
            # Removed windows start at k in [max(0, i-n+1), min(i, L-n)]
            # Added windows start at k in [max(0, i-n+1), min(i-1, L-n-1)]
            # Note: Added windows are formed by word[k:i] + word[i+1:k+n]
            # These windows exist in the NEW word at position k.
            
            k_start = max(0, i - n + 1)
            k_end_removed = min(i, L - n)  # Last valid start index for a window in old word
            k_end_added = min(i - 1, L - n - 1) # Last valid start index for a new window (which is old window start)
            
            # List of ranks of removed windows
            removed_ranks = []
            for k in range(k_start, k_end_removed + 1):
                removed_ranks.append(window_ranks[k])
            
            # List of ranks of added windows
            added_ranks = []
            for k in range(k_start, k_end_added + 1):
                # Construct new window: word[k:i] + word[i+1:k+n]
                # Note: k+n is the end index in the old word.
                # In the new word, this window is at index k.
                w1 = word[k:i]
                w2 = word[i+1:k+n]
                if len(w1) + len(w2) == n:
                    new_window = tuple(w1 + w2)
                    added_ranks.append(perm_rank(new_window))
            
            # Check validity
            valid = True
            # We need to check if any permutation in removed_ranks has count 1 and is NOT in added_ranks.
            # To make this O(1) lookup for added_ranks, we can use a set if n is small, or just a list.
            # n is at most 7, so added_ranks has at most 6 elements.
            added_set = set(added_ranks)
            
            for r in removed_ranks:
                if seen_counts[r] == 1 and r not in added_set:
                    valid = False
                    break
            
            if valid:
                # Update seen_counts and unique_count
                # First, decrement for removed
                for r in removed_ranks:
                    seen_counts[r] -= 1
                    if seen_counts[r] == 0:
                        unique_count -= 1
                
                # Then, increment for added
                for r in added_ranks:
                    if seen_counts[r] == 0:
                        unique_count += 1
                    seen_counts[r] += 1
                
                # Perform deletion
                word = word[:i] + word[i+1:]
                L -= 1
                
                # Rebuild window_ranks for the new word
                # This is O(L * n^2)
                window_ranks = []
                for k in range(L - n + 1):
                    window_ranks.append(perm_rank(tuple(word[k:k+n])))
                
                improved = True
                break # Restart scan
        
        if not improved:
            break
            
    return "".join(map(str, word))
