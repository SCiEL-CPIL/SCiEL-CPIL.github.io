"""Improved program for the football pool / ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for the 120s limit

    # Parameters
    N = q ** n  # Total number of words
    ball_size = 1 + n * (q - 1)  # Size of Hamming ball of radius 1

    # Generate all words as a flat array of base-q digits
    # Using numpy for speed
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)

    # Precompute the indices of words in the Hamming ball of radius 1 for each word.
    # This is the "cover set" for each potential codeword.
    # cover_sets[i] is a list of indices of words within distance 1 of word i.
    cover_sets = []
    # We can compute these efficiently using vectorization or simple loops.
    # Since N=6561 and ball_size=17, a simple loop is fine.
    for i in range(N):
        w = all_words[i]
        indices = [i]
        for pos in range(n):
            original_val = w[pos]
            for alt in range(q):
                if alt == original_val:
                    continue
                # Create new word
                nw = w.copy()
                nw[pos] = alt
                # Find index of nw
                # Since the order is lexicographic (product order), we can compute index directly
                # Index = sum(digit[j] * q^(n-1-j))
                # But converting tuple to index via dict is also fast enough for 6561 items.
                # Let's use a precomputed mapping for speed.
                # Actually, let's just use the direct formula to avoid dict lookup overhead in tight loop?
                # Dict lookup is O(1). Formula involves powers.
                # Let's stick to dict for simplicity, or precompute neighbors.
                # Optimization: Precompute neighbor indices for each position and value.
                pass
        
    # Better approach: Precompute neighbors for each position.
    # neighbors[pos][val][word_idx] -> neighbor_idx ? No, too complex.
    # Let's just compute the covers once.
    
    # Mapping from tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    covers = []
    for i in range(N):
        w = all_words[i]
        c = [i]
        for pos in range(n):
            orig = w[pos]
            for alt in range(q):
                if alt == orig:
                    continue
                nw = w.copy()
                nw[pos] = alt
                c.append(word_to_idx[tuple(nw)])
        covers.append(c)
    
    # Convert to numpy arrays for faster operations
    covers_np = [np.array(c, dtype=np.int32) for c in covers]

    # Incidence matrix A (N x N) where A[i, j] = 1 if j is in covers[i]
    # This allows fast calculation of coverage counts.
    # A[i] is the row for codeword i.
    # If we select a set of codewords S, the coverage count for word j is sum_{i in S} A[i, j].
    # We want min |S| such that for all j, sum_{i in S} A[i, j] >= 1.
    
    A = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        A[i, covers_np[i]] = 1

    # Greedy Set Cover with Random Restarts
    # We will run multiple greedy restarts with different tie-breaking strategies
    # and pick the best (smallest) cover found.
    
    best_cover_size = N
    best_selected = []
    
    # Number of restarts to try within time budget
    # Each greedy pass takes O(N * N) roughly, but with numpy mat-vec it's fast.
    # N=6561. A is 6561x6561. Mat-vec is 6561*6561 ops ~ 43M ops.
    # In numpy, this is very fast (<1ms).
    # Greedy loop runs until covered. Number of iterations ~ K_q(n,r) ~ 500.
    # So each greedy run takes ~500 * 1ms = 0.5s.
    # We have ~110s. So we can do ~200 restarts.
    
    num_restarts = 200
    restarts_done = 0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        covered = np.zeros(N, dtype=bool)
        selected = []
        
        # Greedy Loop
        while not covered.all():
            # Calculate score for each candidate: number of uncovered words it covers
            # scores[i] = sum_{j not covered} A[i, j]
            # uncovered is a boolean vector.
            # A @ uncovered gives the count.
            scores = A @ (not covered).astype(np.int8)
            
            # Mask out already selected
            if selected:
                scores[selected] = 0
            
            # Tie-breaking:
            # 1. Standard greedy: max score. If tie, random.
            # 2. Add some randomness to explore different paths.
            
            max_score = np.max(scores)
            if max_score == 0:
                # Should not happen if not covered.all()
                break
            
            # Get all candidates with max score
            candidates = np.where(scores == max_score)[0]
            
            # Randomly pick one from candidates
            # To introduce diversity, we can bias towards random selection among top-k
            # or just pick random from max.
            # Let's pick random from max for now.
            # Alternatively, pick top-k and random from them.
            # Let's stick to random from max for simplicity and speed.
            
            if len(candidates) > 1:
                # Random selection
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_idx = candidates[0]
                
            selected.append(best_idx)
            covered[covers_np[best_idx]] = True
            
            # Early exit if we already have more than best known
            if len(selected) >= best_cover_size:
                break
        
        # If we completed the cover and it's better
        if covered.all() and len(selected) < best_cover_size:
            best_cover_size = len(selected)
            best_selected = selected[:]
            # Optional: stop early if we hit a very good bound?
            # Known best is 486. If we hit 486, we can try to improve or stop.
            # Let's keep searching for a bit longer to see if we can go lower.
            # But 486 is hard to beat.
            
    # Local Search / Reduction Phase
    # We have a valid cover best_selected.
    # Try to remove codewords one by one if possible.
    # Then try to swap codewords to allow further removals.
    
    if best_selected:
        selected_set = set(best_selected)
        # Compute current coverage counts
        counts = np.zeros(N, dtype=np.int32)
        for idx in best_selected:
            counts[covers_np[idx]] += 1
        
        # Pass 1: Greedy Removal
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # Iterate through a copy to allow removal
            for idx in list(best_selected):
                # Check if removing idx leaves any word with count 0
                # counts[covers_np[idx]] will be decremented
                # If any becomes 0, we can't remove.
                # We can check this quickly:
                # min_count_in_cover = np.min(counts[covers_np[idx]])
                # If min_count_in_cover > 1, we can remove.
                # Wait, if min is 2, after removal it is 1, which is >= 1. So we can remove.
                # If min is 1, after removal it is 0. Cannot remove.
                # So condition: np.min(counts[covers_np[idx]]) > 1
                
                if np.min(counts[covers_np[idx]]) > 1:
                    # Safe to remove
                    counts[covers_np[idx]] -= 1
                    best_selected.remove(idx)
                    selected_set.discard(idx)
                    changed = True
        
        # Pass 2: Local Search Swaps
        # Try to swap a codeword i in selected with a codeword j not in selected
        # such that the new set is valid and allows for further reductions or is just valid.
        # Actually, a simple hill-climbing on the size of the cover:
        # 1. Try to remove.
        # 2. If stuck, try to swap i with j.
        #    - Remove i (tentative).
        #    - Add j.
        #    - Check validity.
        #    - If valid, try to remove more.
        #    - If size decreases, keep. Else revert.
        
        # To make this efficient, we need to identify "critical" words (count=1).
        # Swapping must ensure critical words are still covered.
        
        # Let's do a limited number of random swap attempts.
        max_swap_iterations = 5000
        for _ in range(max_swap_iterations):
            if time.time() > deadline - 5.0:
                break
            
            # Find words with count 1
            critical_words = np.where(counts == 1)[0]
            if len(critical_words) == 0:
                break # No critical words, can we remove more?
                # If no critical words, we might have missed removals?
                # The greedy removal loop above should have handled that.
                # If counts are all >= 2, we can remove any codeword?
                # No, removing one might make some counts 1, but not 0.
                # So we can remove one.
                # Let's run one more removal pass.
                # Actually, if all counts >= 2, we can remove any codeword i
                # because min count in its cover is >= 2, so after removal >= 1.
                # So the while loop above should have removed them all?
                # Wait, the while loop removes if min > 1.
                # If all counts are 2, min is 2 > 1, so it removes.
                # Then counts become 1.
                # Then next iteration, min is 1, not > 1, so it stops.
                # So if we exit the while loop, all remaining codewords have at least one critical word in their cover.
                # So critical_words should not be empty unless we are empty.
                # So if critical_words is empty, we are done? Or we made a mistake.
                # Let's just break.
                pass
            
            # Pick a random critical word
            w_idx = critical_words[random.randint(0, len(critical_words) - 1)]
            
            # Find the codeword in selected that covers w_idx
            # It must be unique since count[w_idx] == 1
            covering_c = None
            for c in covers[w_idx]:
                if c in selected_set:
                    covering_c = c
                    break
            
            if covering_c is None:
                # Should not happen
                continue
            
            # Find a replacement codeword j not in selected that covers w_idx
            # j must be in covers[w_idx] and not in selected_set
            replacements = [c for c in covers[w_idx] if c not in selected_set]
            if not replacements:
                continue
            
            # Pick a random replacement
            new_c = replacements[random.randint(0, len(replacements) - 1)]
            
            # Tentative swap: Remove covering_c, Add new_c
            # Update counts
            counts[covers_np[covering_c]] -= 1
            counts[covers_np[new_c]] += 1
            
            # Check validity: any count < 1?
            if np.min(counts) >= 1:
                # Valid swap
                # Accept swap
                selected_set.discard(covering_c)
                best_selected.remove(covering_c)
                selected_set.add(new_c)
                best_selected.append(new_c)
                
                # Try to remove more
                # Run a quick removal pass
                removed_any = False
                for idx in list(best_selected):
                    if np.min(counts[covers_np[idx]]) > 1:
                        counts[covers_np[idx]] -= 1
                        best_selected.remove(idx)
                        selected_set.discard(idx)
                        removed_any = True
                        break # Just one removal to keep it fast? Or loop?
                        # Let's do a full pass if we have time?
                        # For now, just one removal to see if size drops.
                
                # If size didn't drop, should we revert?
                # The swap itself keeps size same. The removal drops it.
                # If no removal happened, size is same.
                # We can keep the swap if it opens up future possibilities,
                # but for pure size minimization, we should only keep if size decreased.
                # However, sometimes a size-neutral swap is needed to enable a future reduction.
                # Given time constraints, let's only keep if size decreased.
                # To check, we need the size before.
                # old_size = len(best_selected) + 1 (before removal loop)
                # This is getting complicated.
                
                # Simpler strategy:
                # Only accept the swap if it immediately allows a removal.
                # If not, revert.
                if not removed_any:
                    # Revert
                    counts[covers_np[new_c]] -= 1
                    counts[covers_np[covering_c]] += 1
                    selected_set.discard(new_c)
                    best_selected.remove(new_c)
                    selected_set.add(covering_c)
                    best_selected.append(covering_c)
                else:
                    # Size decreased, keep it.
                    # We might want to continue removals?
                    # The outer loop will pick new critical words.
                    pass
            else:
                # Invalid swap, revert
                counts[covers_np[new_c]] -= 1
                counts[covers_np[covering_c]] += 1

    # Final Result
    # Ensure we return a valid list of codewords
    # If best_selected is empty (shouldn't happen), fallback to a large valid cover
    if not best_selected:
        # Fallback: Use the first 486 words? No, that might not be valid.
        # Use a known valid construction or just return all words (too big).
        # Greedy should always find a valid cover.
        # If it didn't, something went wrong.
        # Let's return the greedy result from the first restart if we have it?
        # We didn't store it separately.
        # Let's just re-run a simple greedy without random restarts to ensure validity.
        covered = np.zeros(N, dtype=bool)
        selected = []
        while not covered.all():
            scores = A @ (not covered).astype(np.int8)
            scores[selected] = 0
            best_idx = np.argmax(scores)
            selected.append(best_idx)
            covered[covers_np[best_idx]] = True
        best_selected = selected

    return [all_words[idx].tolist() for idx in best_selected]

# EVOLVE-BLOCK-END
