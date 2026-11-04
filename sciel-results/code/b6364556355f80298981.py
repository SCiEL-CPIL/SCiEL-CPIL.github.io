import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: The evaluator kills at 120s. We stop at 115s to ensure return.
    deadline = start_time + 115.0

    def get_direct_sum():
        if n >= 4 and q == 3:
            # Tetracode is a [4,2,3] code? No, tetracode is [4,2,3] but covering radius 1?
            # Tetracode is a ternary code of length 4, size 9, min distance 3.
            # Covering radius 1?
            # Words: 81. Code: 9. 9 * 3^2 = 81. Yes, it's a perfect code for n=4, q=3, R=1.
            # So direct sum with n-4 coordinates gives 9 * 3^(n-4) words.
            # For n=6, 9 * 9 = 81 words.
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    if q == 3 and n == 6 and radius == 1:
        total = 3 ** n
        powers = [3 ** p for p in range(n)]
        
        # Precompute neighbors: neighbors[i] is a list of indices of words within distance 1 of i
        # There are 1 + 6*2 = 13 neighbors.
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1

        # 1. Greedy Set Cover
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []
        
        while uncovered.sum() > 0:
            if time.time() > deadline:
                break
            
            # Find candidate covering most uncovered words
            # uncovered[neighbors] gives a (729, 13) boolean matrix
            # Sum along axis 1 gives count of covered uncovered neighbors for each candidate
            counts = np.sum(uncovered[neighbors], axis=1)
            best_candidate = np.argmax(counts)
            
            if counts[best_candidate] == 0:
                break
                
            codewords.append(best_candidate)
            uncovered[neighbors[best_candidate]] = 0

        if len(codewords) == 0:
            return get_direct_sum()

        # 2. Local Search: Reduction and Swap/Repair
        # We maintain a list of codeword indices.
        # We try to reduce the size.
        
        def try_reduce(cw_list):
            """Try to remove redundant codewords. Returns (new_list, reduced_bool)"""
            # Calculate coverage counts
            coverage_counts = np.zeros(total, dtype=np.int32)
            for idx in cw_list:
                coverage_counts[neighbors[idx]] += 1
            
            # Try to remove from the end to start (arbitrary order, but helps stability)
            # We iterate multiple times because removing one might make another redundant
            changed = True
            while changed:
                changed = False
                for i in range(len(cw_list) - 1, -1, -1):
                    idx = cw_list[i]
                    # If all neighbors of idx are covered by at least 2 codewords, we can remove idx
                    # coverage_counts[neighbors[idx]] >= 2
                    if np.min(coverage_counts[neighbors[idx]]) >= 2:
                        coverage_counts[neighbors[idx]] -= 1
                        cw_list.pop(i)
                        changed = True
                        break # Restart loop to recalculate? No, counts are updated. 
                        # Actually, removing idx decreases counts. It doesn't increase them.
                        # So other codewords remain redundant or become non-redundant?
                        # If they were redundant (min >= 2), they still are (min >= 1? No, min decreases).
                        # Wait. If I remove idx, counts decrease.
                        # A codeword j was redundant if min(counts[neighbors[j]]) >= 2.
                        # After removing idx, counts[neighbors[j]] might drop to 1.
                        # So j is no longer redundant.
                        # Thus, we should restart the scan from the beginning or just break and restart the while loop.
                        # Breaking the for loop and continuing the while loop is correct.
            
            return cw_list, changed

        # Initial reduction
        codewords, _ = try_reduce(codewords)
        best_size = len(codewords)
        best_codewords = codewords[:]

        # 3. Local Search: Remove and Repair
        # Strategy: Pick a codeword, remove it. If invalid, add the best replacement.
        # If size decreases, keep it.
        
        # To speed up, we can precompute a "best replacement" matrix or just compute on the fly.
        # On the fly is O(N) per check. N=729. 
        # If we do 1000 iterations, it's fast.
        
        current_code = best_codewords[:]
        
        # Precompute a list of all possible codewords for quick access
        all_indices = np.arange(total)
        
        # We will iterate until no improvement is found for a certain number of steps or time runs out.
        no_improve_count = 0
        max_no_improve = 500 # Limit to avoid infinite loops on plateaus
        
        while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
            if not current_code:
                break
            
            # Calculate current coverage
            coverage_counts = np.zeros(total, dtype=np.int32)
            for idx in current_code:
                coverage_counts[neighbors[idx]] += 1
            
            # Try to remove a random codeword
            # We pick a random index in the current_code list
            i = np.random.randint(len(current_code))
            c_remove = current_code[i]
            
            # Tentatively remove
            current_code.pop(i)
            coverage_counts[neighbors[c_remove]] -= 1
            
            # Check validity
            # Find words with coverage 0
            uncovered_mask = (coverage_counts == 0)
            
            if not np.any(uncovered_mask):
                # Valid removal!
                # Try to reduce further
                current_code, changed = try_reduce(current_code)
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_codewords = current_code[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                continue # Restart loop with new state
            
            # Invalid removal. We need to repair.
            # Find a codeword c_add that covers the most uncovered words.
            # uncovered_mask is a boolean array of length 729.
            # We want to maximize sum(uncovered_mask[neighbors[c_add]])
            # But we should prefer c_add != c_remove? Not strictly, but if we add c_remove back, we are back to start.
            # Let's just find the best c_add.
            
            # Optimization: Only consider c_add that cover at least 1 uncovered word.
            # counts_add = np.sum(uncovered_mask[neighbors], axis=1)
            # This is (729, 13) sum -> (729,)
            counts_add = np.sum(uncovered_mask[neighbors], axis=1)
            
            # We want the max count.
            # If max count is 0, we can't repair? That shouldn't happen because c_remove covered them.
            # Unless c_remove is the only one covering some word, and we removed it.
            # In that case, c_remove itself has high count for those words.
            
            best_add = np.argmax(counts_add)
            
            # If the best add is the one we removed, and it's the only option, we revert.
            # But maybe there's another one with slightly less coverage that allows global improvement later?
            # For now, greedy repair: add the best.
            
            # If best_add is already in current_code, we don't need to add it again (it's already there? No, we removed c_remove).
            # If best_add == c_remove, we are just reverting.
            # If best_add != c_remove, we swap.
            
            # Check if best_add is already in current_code
            # This check is O(N). We can use a set for O(1).
            current_set = set(current_code)
            
            if best_add not in current_set:
                current_code.append(best_add)
                coverage_counts[neighbors[best_add]] += 1
                
                # After adding, try to reduce
                current_code, changed = try_reduce(current_code)
                
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_codewords = current_code[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # If best_add is already in current_code, it means the uncovered words are covered by existing codewords?
                # No, if they were covered, uncovered_mask would be 0.
                # This case is tricky. If best_add is in current_code, it means the best candidate to cover uncovered words
                # is one that is already present? That implies my coverage_counts update was wrong or logic error.
                # Actually, if best_add is in current_code, its neighbors are already counted in coverage_counts.
                # So coverage_counts[neighbors[best_add]] should be > 0.
                # But uncovered_mask is where coverage_counts == 0.
                # Contradiction?
                # Ah, if I removed c_remove, coverage_counts dropped.
                # If best_add is in current_code, it means some other codeword is covering the "uncovered" spots?
                # No, if it was covering them, coverage_counts would be > 0.
                # So best_add cannot be in current_code if it covers uncovered spots, unless the uncovered spots are NOT in neighbors[best_add].
                # But counts_add[best_add] is the count of uncovered spots in neighbors[best_add].
                # If counts_add[best_add] > 0, then best_add covers uncovered spots.
                # If best_add is in current_code, those spots should have been covered by best_add.
                # So coverage_counts[those spots] should be >= 1.
                # So they wouldn't be in uncovered_mask.
                # Therefore, best_add should NOT be in current_code if counts_add[best_add] > 0.
                # If counts_add[best_add] == 0, then no codeword covers the uncovered spots? Impossible.
                # So this branch is likely unreachable or indicates a bug.
                # Let's just revert if we are in this state.
                current_code.append(c_remove)
                coverage_counts[neighbors[c_remove]] += 1
                no_improve_count += 1

        # Final check: Ensure we return a valid solution
        # If best_codewords is valid, use it.
        # Otherwise, fallback to greedy result or direct sum.
        
        if best_codewords:
            # Verify validity of best_codewords
            final_coverage = np.zeros(total, dtype=np.int8)
            for idx in best_codewords:
                final_coverage[neighbors[idx]] = 1
            if np.all(final_coverage == 1):
                final_codewords = []
                for idx_c in best_codewords:
                    final_codewords.append([(idx_c // powers[p]) % 3 for p in range(n)])
                # Compare with direct sum
                ds = get_direct_sum()
                if len(final_codewords) <= len(ds):
                    return final_codewords
                else:
                    return ds
            else:
                # Invalid, fall back
                pass
        
        # Fallback to the initial greedy result if it was valid
        # (It should be, as greedy covers everything)
        if codewords:
             final_coverage = np.zeros(total, dtype=np.int8)
             for idx in codewords:
                 final_coverage[neighbors[idx]] = 1
             if np.all(final_coverage == 1):
                 final_codewords = []
                 for idx_c in codewords:
                     final_codewords.append([(idx_c // powers[p]) % 3 for p in range(n)])
                 ds = get_direct_sum()
                 if len(final_codewords) <= len(ds):
                     return final_codewords
                 else:
                     return ds
        
        return get_direct_sum()

    return get_direct_sum()
