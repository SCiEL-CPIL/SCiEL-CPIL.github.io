"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # For the specific case q=3, n=6, radius=1
    total_words = q ** n
    
    def word_to_idx(word, q, n):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx, q, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute neighbors for each word (for radius=1)
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx, q, n)
        neighbors[idx].append(idx)  # self
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w[pos] = new_val
                        nb_idx = word_to_idx(w, q, n)
                        neighbors[idx].append(nb_idx)
                w[pos] = orig
    
    # Convert neighbors to numpy arrays for faster operations
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    def greedy_cover(restart_seed=0, initial_codewords=None, symmetry_tie_break=False):
        """Greedy set cover with coverage counting and optional symmetry-guided tie-breaking."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(restart_seed)
        
        # Coverage count: how many codewords cover each word
        coverage = np.zeros(total_words, dtype=np.int32)
        codewords = []
        covered_mask = np.zeros(total_words, dtype=bool)
        
        # If we have initial codewords, add them
        if initial_codewords:
            for cw in initial_codewords:
                if cw not in codewords:
                    codewords.append(cw)
                    for nb in neighbors_np[cw]:
                        coverage[nb] += 1
                        covered_mask[nb] = True
        
        # Uncovered words as a boolean array (True = uncovered)
        uncovered_arr = np.ones(total_words, dtype=bool)
        for cw in codewords:
            uncovered_arr[neighbors_np[cw]] = False
        
        # Precompute all neighbor arrays as a 2D array for vectorized operations
        # neighbors_2d[i] is the list of neighbors of word i
        # For efficient counting, we'll use a different approach
        
        # Greedy: repeatedly pick the codeword that covers the most uncovered words
        while time.time() < deadline:
            # Check if all covered
            if not uncovered_arr.any():
                break
            
            # For each candidate, count how many of its neighbors are uncovered
            # We need to count for each candidate i: sum(uncovered_arr[neighbors_np[i]])
            # This is expensive to do naively. Let's use a smarter approach.
            
            # Instead of checking all 729 candidates each iteration, 
            # we can use the fact that only candidates near uncovered words matter.
            # But for now, let's do a vectorized version using precomputed structure.
            
            # Build a 2D array where neighbor_matrix[i, j] = 1 if j is a neighbor of i
            # This is 729x729 = 531441 entries, which is fine.
            # Actually, let's just do it more efficiently.
            
            # For each candidate, the count is the number of its neighbors that are uncovered.
            # We can compute this as: for each candidate i, count = sum(uncovered_arr[neighbors_np[i]])
            # To vectorize, we need a 2D boolean matrix.
            
            # Let's precompute a 2D array: is_neighbor[i, j] = True if j is in neighbors[i]
            # Then count[i] = sum(is_neighbor[i, :] & uncovered_arr)
            # This is a matrix-vector product essentially.
            
            # Precompute neighbor matrix once
            if not hasattr(greedy_cover, '_neighbor_matrix'):
                nm = np.zeros((total_words, total_words), dtype=bool)
                for i in range(total_words):
                    nm[i, neighbors_np[i]] = True
                greedy_cover._neighbor_matrix = nm
            
            nm = greedy_cover._neighbor_matrix
            
            # Count uncovered neighbors for each candidate
            # count[i] = sum(nm[i, :] & uncovered_arr)
            counts = nm.dot(uncovered_arr.astype(np.int32))
            
            # Find the candidate with max count
            best_count = counts.max()
            if best_count == 0:
                break
            
            # Get all candidates with max count
            candidates = np.where(counts == best_count)[0]
            
            # Tie-breaking: Choose the candidate that maximizes overlap with existing codewords
            # This promotes redundancy later.
            # Overlap[i] = sum(nm[i, :] & covered_by_existing)
            # covered_by_existing is (coverage > 0)
            # But we have `covered_mask` which is equivalent to (coverage > 0) in the end,
            # but `coverage` is updated incrementally.
            
            # Let's use the current coverage array to measure "potential redundancy"
            # We want a candidate that covers words already covered.
            # score[i] = sum(nm[i, :] & (coverage > 0))
            
            if len(candidates) > 1:
                # Vectorized score calculation for candidates
                # nm[candidates, :] is shape (K, 729)
                # (coverage > 0) is shape (729,)
                # Score = nm[candidates, :].dot((coverage > 0).astype(np.int32))
                
                # To save time, only do this if we have many candidates.
                # If few, just pick the first.
                if len(candidates) > 5:
                    existing_covered_mask = (coverage > 0)
                    scores = nm[candidates, :].dot(existing_covered_mask.astype(np.int32))
                    best_idx_in_cands = int(np.argmax(scores))
                    best_cw = int(candidates[best_idx_in_cands])
                else:
                    best_cw = int(candidates[0])
            else:
                best_cw = int(candidates[0])
            
            # Add best_cw
            codewords.append(best_cw)
            uncovered_arr[neighbors_np[best_cw]] = False
            # Update coverage for tie-breaking in next step
            coverage[neighbors_np[best_cw]] += 1
            covered_mask[neighbors_np[best_cw]] = True
        
        return codewords
    
    def remove_redundant(codewords):
        """Remove redundant codewords iteratively using coverage counts."""
        if time.time() > deadline:
            return codewords
        
        if not codewords:
            return codewords
        
        # Build coverage count: for each word, how many codewords cover it
        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for nb in neighbors_np[cw]:
                coverage[nb] += 1
        
        # Use the precomputed neighbor matrix for faster checks
        if not hasattr(greedy_cover, '_neighbor_matrix'):
            nm = np.zeros((total_words, total_words), dtype=bool)
            for i in range(total_words):
                nm[i, neighbors_np[i]] = True
            greedy_cover._neighbor_matrix = nm
        
        nm = greedy_cover._neighbor_matrix
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                
                cw = codewords[i]
                
                # Check if all words in ball(cw) are covered by at least 2 codewords
                # If so, cw is redundant
                # coverage[neighbors_np[cw]] gives coverage for all words cw covers
                # If min coverage >= 2, then cw is redundant
                if coverage[neighbors_np[cw]].min() >= 2:
                    # Remove cw and update coverage
                    coverage[neighbors_np[cw]] -= 1
                    codewords.pop(i)
                    changed = True
                    break
        
        return codewords
    
    def local_search(codewords, max_iters=200):
        """
        Active local search: Try to replace codewords to enable further removals.
        Strategy: 
        1. Remove redundancies.
        2. For each codeword, try replacing it with a neighbor.
        3. If replacement allows removing another codeword (net -1), accept.
        """
        if time.time() > deadline:
            return codewords
        
        codewords = list(codewords)
        nm = greedy_cover._neighbor_matrix
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            codewords = remove_redundant(codewords)
            
            # If we hit a low count, we might want to be more aggressive, but let's just iterate.
            if len(codewords) <= 73:
                break
                
            # Compute coverage
            coverage = np.zeros(total_words, dtype=np.int32)
            for cw in codewords:
                coverage[neighbors_np[cw]] += 1
            
            # Find codewords that are "candidates" for removal/replacement
            # We want to move codewords that cover many words already covered by others (high redundancy)
            # Or simply try moving every codeword? 729 * 729 is ~500k ops per pass, which is fast.
            # But we only have ~75 codewords. 75 * 729 = 54k checks.
            
            candidates_to_move = []
            for cw in codewords:
                # Count how many words in ball(cw) are covered by >= 2 codewords
                # This indicates "slack"
                slack = (coverage[neighbors_np[cw]] >= 2).sum()
                candidates_to_move.append((slack, cw))
            
            # Try moving codewords with highest slack first
            candidates_to_move.sort(reverse=True)
            
            for slack_val, cw in candidates_to_move:
                if time.time() > deadline:
                    return codewords
                
                # Try replacing cw with each of its neighbors (distance 1)
                # Also try some random positions? Let's stick to neighbors first.
                # If we have time, we can expand to distance 2.
                neighbors_of_cw = neighbors_np[cw]
                
                for new_cw in neighbors_of_cw:
                    if new_cw in codewords:
                        continue
                    
                    # Simulate move: remove cw, add new_cw
                    # Check if this move allows us to remove any other codeword
                    # To do this efficiently, we can update coverage locally?
                    # coverage_new = coverage - ball(cw) + ball(new_cw)
                    
                    # Check if any other codeword becomes redundant
                    # A codeword d is redundant if min(coverage_new[ball(d)]) >= 2
                    # This is expensive to check for all d.
                    
                    # Alternative: Just do the move, run remove_redundant, and see if size drops.
                    # This is O(N * M) where N is set size, M is words per ball.
                    # 75 * 18 = 1350 ops per redundancy check.
                    # 729 neighbors * 1350 = ~1M ops per codeword.
                    # 75 codewords * 1M = 75M ops. Too slow for Python loops.
                    
                    # Optimization: Only try moves if the new codeword covers words that are currently "critical" 
                    # or if the old codeword was "loose".
                    
                    # Let's use a heuristic: 
                    # If the new codeword covers a word that is currently covered ONLY by cw, 
                    # then we MUST ensure that word is covered by new_cw.
                    # If new_cw does NOT cover that unique word, the move is invalid (coverage lost) 
                    # UNLESS that unique word is covered by some other codeword after we remove cw? 
                    # No, if it was unique to cw, removing cw leaves it uncovered. 
                    # So new_cw MUST cover all words in ball(cw) \ intersection(ball(cw), union(other balls)).
                    
                    unique_to_cw = 0
                    for nb in neighbors_np[cw]:
                        if coverage[nb] == 1:
                            unique_to_cw += 1
                            
                    if unique_to_cw > 0:
                        # Check if new_cw covers all these unique words
                        valid_move = True
                        for nb in neighbors_np[cw]:
                            if coverage[nb] == 1:
                                if not nm[new_cw, nb]: # new_cw does not cover nb
                                    valid_move = False
                                    break
                        if not valid_move:
                            continue
                    
                    # If valid, perform the move and check for reduction
                    # We don't want to actually modify the list inside the loop without rollback.
                    # Instead, we can just count how many codewords become redundant.
                    
                    # Approximation: 
                    # The move is good if it increases the "redundancy" of the system.
                    # Let's just try the most promising few moves per codeword.
                    
                    # Heuristic: Prefer new_cw that covers more words with coverage >= 2
                    new_slack = (nm[new_cw, :].dot((coverage >= 2).astype(np.int32)))
                    
                    # We need a better way to evaluate. 
                    # Let's just pick the best few new_cw based on slack and try them.
                    
                    # To keep it simple and fast, let's just try ONE best move per codeword per iteration.
                    # Find new_cw in neighbors_of_cw that maximizes new_slack.
                    # But we need to ensure validity.
                    
                    best_new_cw = -1
                    best_score = -1
                    for new_cw in neighbors_of_cw:
                        if new_cw in codewords:
                            continue
                        # Check validity (covers unique words of cw)
                        if unique_to_cw > 0:
                            # Fast check: 
                            # unique_mask = (coverage == 1) & nm[cw, :]
                            # if not (nm[new_cw, :] & unique_mask).all(): continue
                            # This is vectorized.
                            unique_mask = (coverage == 1) & nm[cw, :]
                            if not np.all(nm[new_cw, :][unique_mask]):
                                continue
                            
                        # Score: How many words does new_cw cover that are already covered by >=2?
                        # This suggests it might make others redundant.
                        score = (nm[new_cw, :] & (coverage >= 2)).sum()
                        if score > best_score:
                            best_score = score
                            best_new_cw = new_cw
                    
                    if best_new_cw != -1:
                        # Execute move
                        codewords.remove(cw)
                        codewords.append(best_new_cw)
                        
                        # Remove redundancies
                        new_codewords = remove_redundant(codewords)
                        
                        if len(new_codewords) < len(codewords):
                            codewords = new_codewords
                            improved = True
                            break # Restart the loop as coverage changed
                        else:
                            # Rollback
                            codewords.remove(best_new_cw)
                            codewords.append(cw)
                            # Mark this cw as "tried" to avoid infinite loops?
                            # Since we break on improvement, and rollback on no improvement,
                            # we might cycle. 
                            # To prevent cycling, we can keep a history or just limit iterations.
                            # The max_iters guard handles this.
        
        return remove_redundant(codewords)
    
    # Strategy:
    # 1. Start with the tetracode-based construction (81 words)
    # 2. Aggressively remove redundant codewords
    # 3. Try multiple random restarts of greedy to find better solutions
    # 4. Combine best results
    
    best_result = None
    best_count = float('inf')
    
    # First, try the tetracode construction with redundancy removal
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for tail_a in range(3):
        for tail_b in range(3):
            for h in tetracode:
                word = h + [tail_a, tail_b]
                idx = word_to_idx(word, q, n)
                initial_codewords.append(idx)
    
    # Remove redundant from initial
    reduced = remove_redundant(initial_codewords.copy())
    if len(reduced) < best_count:
        best_count = len(reduced)
        best_result = reduced
    
    # Try greedy from scratch with different seeds
    # The vectorized greedy is much faster, so we can do more restarts
    for seed in range(50):
        if time.time() > deadline:
            break
        
        result = greedy_cover(restart_seed=seed, initial_codewords=None, symmetry_tie_break=False)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Try hybrid: start with partial tetracode, then greedy
    for seed in range(20):
        if time.time() > deadline:
            break
        
        # Random subset of tetracode as initial
        rng = random.Random(seed)
        initial = rng.sample(initial_codewords, k=min(len(initial_codewords), rng.randint(30, 60)))
        result = greedy_cover(restart_seed=seed, initial_codewords=initial, symmetry_tie_break=False)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Try additional greedy restarts with different initial random seeds
    for seed in range(50, 100):
        if time.time() > deadline:
            break
        
        result = greedy_cover(restart_seed=seed, initial_codewords=None, symmetry_tie_break=False)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Final local search on best result
    if best_result is not None:
        # Apply local search to the best result
        # Also try local search on the second best if we have time?
        # For now, just the best.
        best_result = local_search(best_result)
        
        # One more pass of redundancy removal
        best_result = remove_redundant(best_result)
    
    # Convert back to lists
    if best_result is None:
        # Fallback: return the tetracode construction
        return [idx_to_word(idx, q, n) for idx in initial_codewords]
    
    return [idx_to_word(idx, q, n) for idx in best_result]
# EVOLVE-BLOCK-END
