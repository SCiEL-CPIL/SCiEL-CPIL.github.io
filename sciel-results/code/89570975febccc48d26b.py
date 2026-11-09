"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    N = len(all_words)
    
    # Map word to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute covers
    covers = []
    for idx in range(N):
        w = all_words[idx]
        cover_set = [idx]
        for i in range(n):
            for alt in range(q):
                if alt == w[i]:
                    continue
                nw = list(w)
                nw[i] = alt
                cover_set.append(word_to_idx[tuple(nw)])
        covers.append(cover_set)
        
    covers_np = [np.array(c, dtype=np.int32) for c in covers]
    
    # Build Incidence Matrix A
    # A[i, j] = 1 if j is in covers[i]
    # Using sparse-like logic but dense numpy for speed of mat-vec
    A = np.zeros((N, N), dtype=np.int8)
    for i, cover_set in enumerate(covers):
        A[i, cover_set] = 1
        
    covered = np.zeros(N, dtype=bool)
    selected_indices = []
    
    # Greedy Set Cover
    while not covered.all():
        if time.time() > deadline - 10.0: # Reserve 10s for local search
            break
            
        uncovered = ~covered
        # scores[i] = number of uncovered words covered by codeword i
        scores = A @ uncovered
        # Mask out already selected
        if selected_indices:
            scores[selected_indices] = 0
            
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        covered[covers_np[best_idx]] = True
        
    # If we didn't finish greedy, fallback to simple addition or direct sum?
    # If we broke due to time, we might have an incomplete cover.
    # Ensure validity.
    if not covered.all():
        # Continue adding greedily without the strict time reserve, but check deadline
        while not covered.all() and time.time() < deadline:
            uncovered = ~covered
            scores = A @ uncovered
            if selected_indices:
                scores[selected_indices] = 0
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                # Should not happen if not covered.all()
                # Fallback: pick random uncovered word and add its cover?
                # Or just break and use fallback
                break
            selected_indices.append(best_idx)
            covered[covers_np[best_idx]] = True
            
    if not covered.all():
        # Final safety fallback: Direct Sum Construction (486 words)
        # Tetracode based? 
        # For now, just return what we have if it's valid, else generate a large valid one.
        # The 486 construction:
        # We can generate a simple valid cover by taking all words? No, too big.
        # Let's use the tetracode direct sum logic if we can implement it quickly.
        # Or just return the selected if it covers everything.
        # If it doesn't cover everything, we must return a valid cover.
        # The prompt implies we should return a valid cover.
        # If greedy failed, we can use a known small cover.
        # Let's assume greedy works or we can add a few more.
        
        # Fallback: Add random codewords until covered
        remaining = np.where(~covered)[0]
        while remaining.size > 0 and time.time() < deadline:
            w_idx = random.choice(remaining)
            # Add a codeword that covers w_idx
            # Any c in covers[w_idx] works
            c_idx = random.choice(covers[w_idx])
            if c_idx not in selected_indices:
                selected_indices.append(c_idx)
                covered[covers_np[c_idx]] = True
                remaining = np.where(~covered)[0]
                
    if not covered.all():
        # Absolute fallback: Return all words? No, score will be bad.
        # Return the 486-word construction? 
        # We don't have the exact 486 construction code handy in this block.
        # But we can construct a valid cover of size 3^n / (1 + n(q-1)) * overhead?
        # Let's just return the selected list. It should be valid.
        pass

    # Local Search Phase
    if time.time() < deadline - 5.0:
        # Use boolean mask for selection
        is_selected = np.zeros(N, dtype=bool)
        for idx in selected_indices:
            is_selected[idx] = True
            
        # Calculate initial counts
        # counts[w] = number of selected codewords covering w
        counts = A[:, is_selected].sum(axis=1).astype(np.int32)
        
        # Precompute for each codeword: list of words it covers (already have covers_np)
        # Also precompute: for each word, list of codewords that cover it
        word_to_covers = [np.array([i for i in range(N) if A[i, w] == 1], dtype=np.int32) for w in range(N)]
        
        def try_remove(idx):
            if not is_selected[idx]: 
                return False
            # Check if any word covered by idx has count 1
            if np.any(counts[covers_np[idx]] == 1):
                return False
            # Safe to remove
            is_selected[idx] = False
            counts[covers_np[idx]] -= 1
            return True

        # Greedy Remove Pass - more efficient: iterate in fixed order, skip removed
        changed = True
        while changed:
            changed = False
            for idx in np.where(is_selected)[0]:
                if try_remove(idx):
                    changed = True
        
        # Local Search Swaps - improved: try multiple swaps per pass, better replacement selection
        while time.time() < deadline - 2.0:
            critical_words = np.where(counts == 1)[0]
            if len(critical_words) == 0:
                break
            
            # Try up to 10 swaps per pass to make better progress
            swaps_done = 0
            for _ in range(10):
                if time.time() > deadline - 2.0:
                    break
                critical_words = np.where(counts == 1)[0]
                if len(critical_words) == 0:
                    break
                    
                w_idx = random.choice(critical_words)
                
                # Find unique selected codeword c covering w_idx
                covering_c = -1
                for c in covers[w_idx]:
                    if is_selected[c]:
                        covering_c = c
                        break
                
                if covering_c == -1:
                    continue
                    
                # Find a replacement c' - prefer one that covers more critical words
                replacements = [c for c in covers[w_idx] if not is_selected[c]]
                if not replacements:
                    continue
                
                # Score replacements by how many critical words they cover
                critical_set = set(critical_words.tolist())
                best_new_c = max(replacements, key=lambda c: sum(1 for w in covers_np[c] if w in critical_set))
                
                # Check if swap is valid
                counts_copy = counts.copy()
                counts_copy[covers_np[covering_c]] -= 1
                counts_copy[covers_np[best_new_c]] += 1
                
                if np.any(counts_copy == 0):
                    continue
                else:
                    # Accept swap
                    is_selected[covering_c] = False
                    is_selected[best_new_c] = True
                    counts = counts_copy
                    swaps_done += 1
                    
                    # Try to remove more after each swap
                    changed = True
                    while changed:
                        changed = False
                        for idx in np.where(is_selected)[0]:
                            if try_remove(idx):
                                changed = True
            
            if swaps_done == 0:
                # No swaps made in this pass, try a different strategy
                # Random restart: pick a random selected codeword and try to replace it
                selected_list = np.where(is_selected)[0]
                if len(selected_list) == 0:
                    break
                random_c = random.choice(selected_list.tolist())
                # Try to find a replacement that allows removal of random_c
                # This is complex; just break and let the loop continue
                break

    # Update selected_indices from is_selected
    selected_indices = np.where(is_selected)[0].tolist()

    # Final validity check and repair if needed
    if not covered.all():
        # Recompute covered from final selection
        covered = np.zeros(N, dtype=bool)
        for idx in selected_indices:
            covered[covers_np[idx]] = True
        
        if not covered.all():
            # Add more codewords to cover remaining
            while not covered.all() and time.time() < deadline:
                uncovered = np.where(~covered)[0]
                w_idx = random.choice(uncovered.tolist())
                # Find a codeword covering w_idx that's not already selected
                candidates = [c for c in covers[w_idx] if c not in set(selected_indices)]
                if candidates:
                    c_idx = random.choice(candidates)
                    selected_indices.append(c_idx)
                    covered[covers_np[c_idx]] = True
                else:
                    # All covering codewords are already selected? That shouldn't happen
                    break

    # Return result
    return [all_words[idx].tolist() for idx in selected_indices]

# EVOLVE-BLOCK-END
