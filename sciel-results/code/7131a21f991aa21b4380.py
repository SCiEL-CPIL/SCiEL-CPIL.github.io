import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # Mapping functions
    def int_to_word(x):
        w = [0] * n
        for k in range(n):
            w[k] = x % q
            x //= q
        return w

    def word_to_int(w):
        x = 0
        base = 1
        for k in range(n):
            x += w[k] * base
            base *= q
        return x

    # If the space is too large, return a trivial (but large) solution to avoid timeout
    if total_words > 50000:
        return [[i // (q ** (n - 1 - k)) % q for k in range(n)] for i in range(total_words)]

    # Precompute ball indices: ball_indices[i] is a numpy array of indices within distance radius of word i
    ball_indices = []
    for i in range(total_words):
        wi = [0] * n
        x = i
        for k in range(n):
            wi[k] = x % q
            x //= q
        
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = wi[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    wi[pos] = new_val
                    ball.append(word_to_int(wi))
                wi[pos] = orig_val
        ball_indices.append(np.array(ball, dtype=np.int32))
    
    # Precompute boolean matrix for vectorized operations
    # ball_bool[i, j] = 1 if j in ball_indices[i]
    ball_bool = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        ball_bool[i, ball_indices[i]] = 1
    
    # Also precompute the transpose for faster coverage updates if needed, 
    # but ball_bool[i, :] is already efficient for gain calculation.
    
    best_codewords = None
    best_size = float('inf')
    
    def remove_redundant(selected_list, selected_set, coverage):
        """Remove redundant codewords. Returns updated list, set, and coverage."""
        if not selected_list:
            return selected_list, selected_set, coverage
        
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                ball = ball_indices[idx]
                # Check if all words covered by idx are covered by at least one other selected codeword
                if np.all(coverage[ball] > 1):
                    # Remove
                    selected_list.pop(i)
                    selected_set.remove(idx)
                    coverage[ball] -= 1
                    changed = True
                    break
        return selected_list, selected_set, coverage

    def greedy_set_cover(seed):
        """Run greedy set cover with random tie-breaking."""
        rng = random.Random(seed)
        selected_set = set()
        selected_list = []
        uncovered = np.ones(total_words, dtype=bool)
        coverage = np.zeros(total_words, dtype=np.int32)
        
        while uncovered.any():
            if time.time() > deadline - 10:
                break
            # Compute gain for each candidate: number of uncovered words covered by candidate i
            # gains[i] = sum(uncovered[j] for j in ball_indices[i])
            gains = (ball_bool @ uncovered.astype(np.uint8)).astype(np.int32)
            
            # Mask out already selected
            for idx in selected_set:
                gains[idx] = -1
            
            # Find best
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_idx = int(rng.choice(candidates))
            else:
                best_idx = int(candidates[0])
            
            selected_set.add(best_idx)
            selected_list.append(best_idx)
            coverage[ball_indices[best_idx]] += 1
            uncovered[ball_indices[best_idx]] = False
        
        return selected_list, selected_set, coverage

    def kick_and_repair(selected_list, selected_set, coverage, rng, num_kicks=3):
        """
        Local search using kick-and-repair.
        1. Remove k random codewords.
        2. Identify uncovered words.
        3. Greedily add back codewords to cover them.
        4. If size decreases, accept.
        """
        current_list = selected_list[:]
        current_set = selected_set.copy()
        current_coverage = coverage.copy()
        
        improved = False
        for _ in range(num_kicks):
            if time.time() > deadline - 5:
                break
                
            k = rng.randint(2, 5)
            if len(current_list) < k + 5:
                continue
                
            # Kick: remove k random codewords
            kick_indices = rng.sample(current_list, k)
            for idx in kick_indices:
                current_list.remove(idx)
                current_set.remove(idx)
                current_coverage[ball_indices[idx]] -= 1
            
            # Find uncovered words
            uncovered_mask = (current_coverage == 0)
            if not uncovered_mask.any():
                # Perfect removal, accept immediately
                improved = True
                continue
                
            uncovered_indices = np.where(uncovered_mask)[0]
            
            # Repair: greedily cover uncovered words
            added_list = []
            temp_uncovered = uncovered_mask.copy()
            
            while temp_uncovered.any():
                # Calculate gains for all candidates on the current uncovered set
                gains = (ball_bool @ temp_uncovered.astype(np.uint8)).astype(np.int32)
                
                # Mask out already selected (in current_set)
                for idx in current_set:
                    gains[idx] = -1
                
                max_gain = np.max(gains)
                if max_gain <= 0:
                    # Should not happen if repair is possible, but break to be safe
                    break
                    
                candidates = np.where(gains == max_gain)[0]
                # Random tie-breaking for diversity
                best_idx = int(rng.choice(candidates))
                
                current_set.add(best_idx)
                current_list.append(best_idx)
                current_coverage[ball_indices[best_idx]] += 1
                temp_uncovered[ball_indices[best_idx]] = False
                added_list.append(best_idx)
                
                # Early exit if we added more than we removed
                if len(added_list) >= k:
                    break
            
            # Check if the move was beneficial
            if len(current_list) < len(selected_list):
                # Accept the move
                improved = True
                # Update global state for next iteration of this function call
                selected_list = current_list[:]
                selected_set = current_set.copy()
                coverage = current_coverage.copy()
                
                # Reset current state to the new best for the next kick
                current_list = selected_list[:]
                current_set = selected_set.copy()
                current_coverage = coverage.copy()
                
                # Remove redundancies immediately
                selected_list, selected_set, coverage = remove_redundant(selected_list, selected_set, coverage)
                current_list = selected_list[:]
                current_set = selected_set.copy()
                current_coverage = coverage.copy()
                
            else:
                # Revert
                current_list = selected_list[:]
                current_set = selected_set.copy()
                current_coverage = coverage.copy()
                
        return selected_list, selected_set, coverage, improved

    # Main loop: multiple trials with different seeds
    num_trials = 10
    
    for trial in range(num_trials):
        if time.time() > deadline - 20:
            break
            
        # 1. Greedy Construction
        selected_list, selected_set, coverage = greedy_set_cover(seed=trial * 123 + 7)
        
        # 2. Initial Redundancy Removal
        selected_list, selected_set, coverage = remove_redundant(selected_list, selected_set, coverage)
        
        if len(selected_list) < best_size:
            best_size = len(selected_list)
            best_codewords = selected_list[:]
            
        # 3. Local Search: Kick and Repair
        rng = random.Random(trial * 456 + 13)
        
        # Run kick-and-repair for a while
        max_kick_iterations = 200
        for _ in range(max_kick_iterations):
            if time.time() > deadline - 10:
                break
            
            selected_list, selected_set, coverage, improved = kick_and_repair(
                selected_list, selected_set, coverage, rng, num_kicks=3
            )
            
            if len(selected_list) < best_size:
                best_size = len(selected_list)
                best_codewords = selected_list[:]
                
            # If no improvement in several rounds, we might be stuck, 
            # but kick-and-repair is stochastic so we keep trying with new random kicks.
            
        # 4. Final Polish: Try single deletions and 1-for-1 swaps
        if time.time() < deadline - 5:
            # Try removing any remaining redundant codewords
            selected_list, selected_set, coverage = remove_redundant(selected_list, selected_set, coverage)
            
            # Try 1-for-1 swaps
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                if len(selected_list) < 10:
                    break
                    
                # Try removing one
                remove_idx = rng.choice(selected_list)
                ball_remove = ball_indices[remove_idx]
                
                # Temporarily remove
                selected_list.remove(remove_idx)
                selected_set.remove(remove_idx)
                coverage[ball_remove] -= 1
                
                uncovered_mask = (coverage == 0)
                if not uncovered_mask.any():
                    # Perfect removal
                    if len(selected_list) < best_size:
                        best_size = len(selected_list)
                        best_codewords = selected_list[:]
                    # Keep it removed
                    continue
                else:
                    # Try to add one to cover all uncovered
                    gains = (ball_bool @ uncovered_mask.astype(np.uint8)).astype(np.int32)
                    for idx in selected_set:
                        gains[idx] = -1
                    
                    max_gain = np.max(gains)
                    if max_gain == np.sum(uncovered_mask):
                        # Found a perfect replacement
                        add_candidates = np.where(gains == max_gain)[0]
                        add_idx = int(rng.choice(add_candidates))
                        
                        # Apply swap
                        selected_list.append(add_idx)
                        selected_set.add(add_idx)
                        coverage[ball_indices[add_idx]] += 1
                        
                        if len(selected_list) < best_size:
                            best_size = len(selected_list)
                            best_codewords = selected_list[:]
                        # Do not revert, continue with new state
                    else:
                        # Revert
                        selected_list.append(remove_idx)
                        selected_set.add(remove_idx)
                        coverage[ball_remove] += 1

    if best_codewords is None:
        return [int_to_word(i) for i in range(total_words)]
        
    return [int_to_word(i) for i in best_codewords]
