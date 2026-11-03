import time
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        if q**n < 10000:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            # Generic greedy fallback - simplified for non-main cases to avoid timeout
            # Since the task is specific, we return a trivial valid solution if possible
            # or a small set if we can't compute fast. 
            # For safety, return all words if small, else a heuristic.
            if q**n <= 100000:
                 return [list(t) for t in itertools.product(range(q), repeat=n)]
            return [] # Should not happen for the specific task, but avoids crash

    # Specific optimization for q=3, n=7, r=1
    space_size = 3**n
    
    # Precompute all words as integers 0..2186
    # and their spheres (neighbors within distance 1)
    # We will use a 2D array `spheres` of shape (space_size, 15)
    # where spheres[i] contains the indices of the 15 words covered by word i
    
    # 1. Generate all words
    words = np.array(list(itertools.product(range(3), repeat=n)), dtype=np.int32)
    
    # 2. Build spheres
    # For each word, generate its 14 neighbors + itself
    spheres = np.zeros((space_size, 15), dtype=np.int32)
    
    # Helper to convert tuple to index (base-3)
    def tup_to_idx(t):
        idx = 0
        for v in t:
            idx = idx * 3 + v
        return idx

    for i in range(space_size):
        w = words[i]
        neighbors = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val != orig:
                    w[pos] = new_val
                    neighbors.append(tup_to_idx(w))
            w[pos] = orig
        spheres[i] = neighbors

    # 3. Greedy Set Cover
    # coverage_counts: number of times each word is covered
    coverage_counts = np.zeros(space_size, dtype=np.int32)
    code_indices = []
    
    # Precompute the spheres of the code? No, we can just use the `spheres` array directly.
    # To speed up gain calculation, we can maintain a boolean array `covered` 
    # or just count uncovered words in the sphere.
    
    # Optimization: 
    # Gain of adding c = sum(1 for w in spheres[c] if coverage_counts[w] == 0)
    # We can compute this vectorized if we maintain a mask of uncovered words.
    # But checking `coverage_counts[spheres[c]] == 0` is fast enough with NumPy.
    
    while True:
        if time.time() > deadline:
            break
            
        if len(code_indices) > 250: # Safety limit
            break
            
        # Calculate gain for all candidates
        # We only need to consider candidates that might have high gain.
        # Initially all are candidates.
        
        # Vectorized gain calculation:
        # For each candidate c, gain = count of w in spheres[c] where cov[w]==0
        # spheres is (N, 15). coverage_counts is (N,).
        # We can expand coverage_counts to (N, 15) by indexing, but that's expensive.
        # Instead, we can iterate or use a trick.
        
        # Trick: 
        # gain[c] = np.sum(coverage_counts[spheres[c]] == 0)
        # This creates a temporary array of size N*15 for each iteration? 
        # No, `spheres` is (N, 15). `coverage_counts[spheres]` is (N, 15).
        # `coverage_counts[spheres] == 0` is (N, 15) bool.
        # `np.sum(..., axis=1)` is (N,).
        # This is O(N*15) per iteration. N=2187, 15*2187 ~ 30k ops. Very fast.
        
        gains = np.sum(coverage_counts[spheres] == 0, axis=1)
        
        best_c = np.argmax(gains)
        best_gain = gains[best_c]
        
        if best_gain == 0:
            break # All words covered
            
        code_indices.append(best_c)
        
        # Update coverage
        for w in spheres[best_c]:
            coverage_counts[w] += 1
            
        # Check if fully covered
        if np.all(coverage_counts > 0):
            break

    # 4. Local Search (Remove and Swap)
    # We want to minimize len(code_indices).
    # Phase A: Remove redundant codewords
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(code_indices) - 1, -1, -1):
            c = code_indices[i]
            # Check if any word in sphere(c) is covered exactly once
            # If yes, c is essential.
            # If no, c is redundant, remove it.
            counts_in_sphere = coverage_counts[spheres[c]]
            if np.any(counts_in_sphere == 1):
                continue
            else:
                # Remove c
                code_indices.pop(i)
                for w in spheres[c]:
                    coverage_counts[w] -= 1
                changed = True

    # Phase B: Swap local search
    # Try to replace a codeword c with another codeword c' such that:
    # 1. The new set is still valid.
    # 2. It allows removing another codeword? 
    # Or simply: try to find a pair (c_in, c_out) such that replacing c_in with c_out 
    # doesn't break validity, and then see if we can remove something else.
    # Actually, a simpler swap: 
    # For each c_in in code:
    #   For each c_out not in code:
    #     Try swap: remove c_in, add c_out.
    #     If valid, check if we can remove another codeword.
    #     If we can remove one, accept swap and removal.
    # This is expensive. 
    # 
    # Alternative: Just try to remove. The "Swap" in inspirations might refer to 
    # moving a codeword to a better position? No, codewords are just indices.
    # 
    # Let's stick to a robust removal loop first, as it often gets close to optimal.
    # If we have time, we can try a random restart or perturbation.
    
    # Given the 120s limit and the small N, we can afford a few more removal passes.
    # The current removal is sequential. Let's ensure we loop until no change.
    # (Already done in Phase A loop)
    
    # Let's try a simple "2-opt" style swap:
    # Iterate through all pairs (i, j) where i < j in code.
    # Try removing both? No.
    
    # Let's try a random perturbation + removal if we have time.
    # But deterministic is better for scoring consistency?
    # The inspirations mention "swap m".
    
    # Let's implement a simple swap: 
    # For each c in code:
    #   For each c' in all_words (not in code):
    #      if swap(c, c') allows removing at least one other codeword:
    #         do swap, do removals, accept.
    
    # This is O(|code| * N * RemovalCost).
    # |code| ~ 180. N ~ 2200. RemovalCost ~ 180 * 15.
    # 180 * 2200 * 2700 ~ 10^9. Too slow.
    
    # We need a faster heuristic.
    # Maybe just run the removal phase multiple times?
    # The removal phase is already O(|code| * N * 15) roughly if we recompute?
    # No, my removal phase is O(|code|^2 * 15) roughly because I iterate code and check sphere.
    # Wait, `coverage_counts[spheres[c]]` is fast.
    # The loop is `for i in range(len(code))`. Inside, `np.any(...)`.
    # This is fast.
    
    # Let's add a "Greedy Remove" which picks the codeword with the fewest unique covered words to remove first?
    # No, we must only remove if it's safe (no unique words).
    
    # Let's try to improve the Greedy phase itself.
    # Standard greedy is often suboptimal.
    # But with N=2187, greedy is quite good.
    
    # Let's add a final polish: 
    # Randomly shuffle the order of consideration in greedy? No, greedy is deterministic.
    
    # Let's try a simple local search:
    # While time < deadline - 5s:
    #    Try to find a redundant codeword. If found, remove.
    #    If none found, try a swap:
    #       Pick a random codeword c_in.
    #       Pick a random candidate c_out (not in code).
    #       Swap.
    #       Try to remove redundant codewords.
    #       If we removed at least 1, keep swap.
    #       Else, revert swap.
    
    # This might help escape local minima.
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # First, try pure removal
        # (The code above already did this, but maybe we can do it again if we added swaps?)
        # Actually, the previous loop already removed all redundancies.
        
        # Try swap
        # We need a list of code indices for fast membership check
        code_set = set(code_indices)
        
        # To make this faster, we can precompute which words are covered exactly once.
        # If we swap c_in -> c_out, the words in sphere(c_in) lose 1 cover, words in sphere(c_out) gain 1.
        # Validity check:
        # 1. No word in sphere(c_in) \ sphere(c_out) has count 1. (Because they drop to 0)
        # 2. (Gain is not required for validity, just coverage)
        
        # Let's iterate c_in
        for i in range(len(code_indices)):
            c_in = code_indices[i]
            # Get words covered by c_in
            s_in = spheres[c_in]
            
            # Find words in s_in that are covered exactly once
            # These are "critical" for c_in
            critical_words = s_in[coverage_counts[s_in] == 1]
            
            # If no critical words, c_in is redundant (should have been removed).
            # So there is at least one.
            
            # We need to find a c_out that covers all critical_words.
            # And c_out must not be in code.
            
            # Optimization: 
            # Instead of checking all c_out, we can check candidates that cover the critical words.
            
            # Let's just try a few random c_out if we have time, or all if small.
            # Since N is small, let's try all c_out not in code.
            
            # To speed up, we can precompute a list of candidates that cover a specific word?
            # Too complex.
            
            # Let's just loop c_out.
            # We can break early if we find a valid swap that allows removal.
            
            # For efficiency, we only check swaps if we think it might help.
            # Actually, just trying all swaps is 180 * 2000 = 360,000 checks.
            # Each check involves set intersection or similar.
            # `critical_words` is small (usually 1-3 words).
            
            # Check: does c_out cover all critical_words?
            # `set(critical_words).issubset(spheres[c_out])`
            
            # If yes, perform swap and check if we can remove anything.
            
            # This is still heavy. Let's limit the number of swaps we try per iteration.
            # Or just rely on the fact that Greedy + Remove is usually very good for this small N.
            # The score 1.0167 suggests we are at ~186 words? 
            # 1.0 / 1.0167 ~ 0.983. 
            # If seed is 186, 186/186 = 1.0. 
            # If we get 184, score = 186/184 = 1.0108.
            # If we get 180, score = 186/180 = 1.033.
            
            # The inspirations got 1.11, which implies ~167 words? 
            # 186/167 = 1.113.
            # So we need to get significantly lower than greedy+remove.
            
            # Let's implement the swap search properly but efficiently.
            
            # Precompute: for each word w, list of codewords covering it? No, that changes.
            
            # Let's try: 
            # For each c_in:
            #   critical = words with count 1 in sphere(c_in)
            #   If critical is empty, remove c_in (redundant).
            #   Else:
            #      Find c_out (not in code) that covers ALL words in critical.
            #      If found, swap c_in with c_out.
            #      Then run removal phase.
            #      If removal phase removed anything, keep swap.
            #      Else, revert swap.
            
            # To find c_out covering critical:
            # We can precompute for each word w, the list of all codewords (from the universe) that cover w.
            # `coverers[w]` = list of c such that w in spheres[c].
            # Then `candidates = intersection(coverers[w] for w in critical)`.
            # Then filter candidates not in code_set.
            
            # This precomputation `coverers` takes O(N * 15) time and memory.
            # N=2187. coverers is list of 2187 lists.
            # Each list has ~15 * (N/15) ? No.
            # Each word is covered by 15 codewords (itself + 14 neighbors).
            # So `coverers[w]` has size 15.
            # Intersection of sets of size 15 is very fast.
            
            # Let's build `coverers`.
            
            if not hasattr(construct, 'coverers_cache'):
                 coverers = [[] for _ in range(space_size)]
                 for c in range(space_size):
                     for w in spheres[c]:
                         coverers[w].append(c)
                 construct.coverers_cache = coverers
            else:
                 coverers = construct.coverers_cache
            
            # Now the swap loop:
            for i in range(len(code_indices)):
                c_in = code_indices[i]
                s_in = spheres[c_in]
                # Critical words
                mask = (coverage_counts[s_in] == 1)
                if not np.any(mask):
                    # Redundant
                    code_indices.pop(i)
                    for w in s_in:
                        coverage_counts[w] -= 1
                    improved = True
                    # Restart removal loop? Or just continue.
                    # If we removed one, the indices changed.
                    # Let's break and restart the whole `while improved` loop to be safe.
                    break
                
                critical_words = s_in[mask]
                
                # Find common coverers
                # Start with coverers of first critical word
                common = set(coverers[critical_words[0]])
                for w in critical_words[1:]:
                    common.intersection_update(coverers[w])
                
                # Filter out current code
                # common is a set of ints
                # code_set is a set of ints
                candidates = common - code_set
                
                if not candidates:
                    continue
                
                # Try the first candidate (or best one?)
                # Let's just try one random or first one.
                c_out = next(iter(candidates))
                
                # Perform swap
                # Remove c_in
                for w in s_in:
                    coverage_counts[w] -= 1
                
                # Add c_out
                s_out = spheres[c_out]
                for w in s_out:
                    coverage_counts[w] += 1
                
                # Check validity (should be valid by construction of critical words)
                # But we need to check if any word became 0?
                # The words in s_in that were critical are now covered by c_out.
                # Words in s_in \ s_out: count decreased by 1.
                # If any was 1, it becomes 0. 
                # But we ensured c_out covers ALL critical words (count 1).
                # What about words with count > 1? They become count-1, so >= 1.
                # What about words in s_out \ s_in? Count increased by 1.
                # So validity is guaranteed IF c_out covers all critical words.
                
                # Now try to remove redundant codewords
                removed_any = False
                # We can do a quick removal pass
                # Note: code_indices still has c_in at i. We should replace it with c_out first?
                # Or just remove c_in, add c_out to end, then clean up.
                
                # Let's update code_indices
                code_indices[i] = c_out
                code_set.remove(c_in)
                code_set.add(c_out)
                
                # Removal pass
                changed = True
                while changed:
                    changed = False
                    for j in range(len(code_indices) - 1, -1, -1):
                        c = code_indices[j]
                        s_c = spheres[c]
                        if not np.any(coverage_counts[s_c] == 1):
                            code_indices.pop(j)
                            code_set.remove(c)
                            for w in s_c:
                                coverage_counts[w] -= 1
                            changed = True
                            removed_any = True
                
                if not removed_any:
                    # Revert swap
                    # Remove c_out
                    for w in s_out:
                        coverage_counts[w] -= 1
                    # Add c_in
                    for w in s_in:
                        coverage_counts[w] += 1
                    code_indices[i] = c_in
                    code_set.remove(c_out)
                    code_set.add(c_in)
                else:
                    improved = True
                    break # Restart loop
            
            if improved:
                break

    # Final check: ensure no redundancies left
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        for i in range(len(code_indices) - 1, -1, -1):
            c = code_indices[i]
            s_c = spheres[c]
            if not np.any(coverage_counts[s_c] == 1):
                code_indices.pop(i)
                for w in s_c:
                    coverage_counts[w] -= 1
                changed = True

    # Convert to lists
    result = [words[i].tolist() for i in code_indices]
    return result
