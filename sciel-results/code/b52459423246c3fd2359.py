import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Handle the specific known case: q=3, n=6, radius=1
    # Known optimal is 73. We aim to find a cover of size 73 or less.
    
    if q == 3 and n == 6 and radius == 1:
        # Precompute index conversions
        def idx_to_tuple(idx):
            t = [0] * 6
            rem = idx
            for i in range(6):
                t[i] = rem % 3
                rem //= 3
            return tuple(t)
        
        def tuple_to_idx(t):
            idx = 0
            pow3 = 1
            for i in range(6):
                idx += t[i] * pow3
                pow3 *= 3
            return idx

        total_words = 729
        
        # Precompute balls as sets of indices
        # Ball size is 1 + 6*2 = 13
        balls = [None] * total_words
        for i in range(total_words):
            w = idx_to_tuple(i)
            ball = {i}
            for pos in range(6):
                orig_val = w[pos]
                for new_val in range(3):
                    if new_val != orig_val:
                        new_w = list(w)
                        new_w[pos] = new_val
                        new_idx = tuple_to_idx(tuple(new_w))
                        ball.add(new_idx)
            balls[i] = ball

        # Helper to check if a set of indices covers the space
        def is_cover(indices):
            covered = set()
            for idx in indices:
                covered.update(balls[idx])
            return len(covered) == total_words

        # Helper to prune redundant codewords
        # This is crucial to get down to 73 from greedy results which might be ~75-80
        def prune_cover(indices):
            indices = list(indices)
            # Count how many times each word is covered
            cover_counts = [0] * total_words
            for idx in indices:
                for w in balls[idx]:
                    cover_counts[w] += 1
            
            changed = True
            while changed:
                changed = False
                new_indices = []
                for idx in indices:
                    # Check if this idx is necessary
                    # It is necessary if it covers any word with count 1
                    necessary = False
                    for w in balls[idx]:
                        if cover_counts[w] == 1:
                            necessary = True
                            break
                    if necessary:
                        new_indices.append(idx)
                    else:
                        # Remove it
                        for w in balls[idx]:
                            cover_counts[w] -= 1
                        changed = True
                indices = new_indices
            return indices

        best_code = None
        best_size = float('inf')
        
        # Strategy 1: Direct Product Construction (Tetracode based)
        # This gives 9 * 3^(6-4) = 27 words? No, wait.
        # Tetracode is length 4. If we take direct sum with length 2, size is 9 * 3^2 = 81.
        # But the tetracode is a perfect code of radius 1 in Z_3^4.
        # So the direct sum is a valid covering code of radius 1 in Z_3^6 with size 81.
        # 81 is worse than 73.
        # However, we can use it as a starting point or a fallback.
        
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        dp_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                word = h + list(tail)
                dp_indices.append(tuple_to_idx(tuple(word)))
        
        # Prune the direct product code
        pruned_dp = prune_cover(dp_indices)
        if len(pruned_dp) < best_size:
            best_size = len(pruned_dp)
            best_code = pruned_dp
            
        # Strategy 2: Randomized Greedy
        # We run multiple restarts of a randomized greedy algorithm.
        # The greedy step picks a word that covers the maximum number of uncovered words.
        # Randomization helps escape local minima and find different structures.
        
        num_restarts = 100 # We have ~110 seconds. Each restart is fast for n=6.
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            uncovered = set(range(total_words))
            code_indices = []
            
            while uncovered:
                # Calculate gains for all candidates
                # Optimization: Instead of sorting all 729, we can find the max gain.
                # But sorting 729 items is fast.
                gains = []
                for c in range(total_words):
                    gain = len(balls[c] & uncovered)
                    if gain > 0:
                        gains.append((gain, c))
                
                if not gains:
                    break # Should not happen if uncovered is not empty
                
                # Sort by gain descending
                gains.sort(key=lambda x: x[0], reverse=True)
                
                # Randomized selection
                # Pick from the top K candidates with max gain, or top K overall
                # Let's pick from the top 5 candidates with the highest gain to introduce diversity
                max_gain = gains[0][0]
                top_candidates = [c for g, c in gains if g == max_gain]
                
                # If there are many max gain candidates, pick randomly from them
                # If we want more diversity, we could pick from top 3 overall, but usually max gain is the right move.
                # Let's stick to random among max gain for stability, or maybe top 2-3.
                # Actually, just picking random among max gain is a good heuristic.
                best_c = random.choice(top_candidates)
                
                code_indices.append(best_c)
                uncovered -= balls[best_c]
                
                # Time check inside the loop
                if time.time() > deadline - 10.0:
                    break
            
            if uncovered:
                # Incomplete cover, discard
                continue
                
            # Prune the result
            pruned = prune_cover(code_indices)
            
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned
                
                # If we hit 73, we can stop early?
                # The lower bound is 71, so 72 or 71 might be possible but unlikely.
                # 73 is the best known. If we find 73, we are good.
                # Let's continue to see if we can find 72 or 71.
                
        if best_code is None:
            # Fallback to direct product
            best_code = pruned_dp
        else:
            # Final verification
            if not is_cover(best_code):
                # Should not happen, but just in case
                best_code = pruned_dp

        # Convert indices back to list of lists
        result = [list(idx_to_tuple(c)) for c in best_code]
        return result

    else:
        # Generic fallback for other parameters
        # If the space is small, use greedy
        total_words = q ** n
        if total_words <= 10000:
            # Re-use logic from above but parameterized? 
            # For simplicity and safety, just return a large set or use a simple greedy
            # Since the task is specific to 3,6,1, this branch is unlikely.
            # But let's implement a basic greedy to be safe.
            
            powers = [1] * n
            for i in range(1, n):
                powers[i] = powers[i-1] * q
                
            def idx_to_tuple_gen(idx):
                t = [0] * n
                rem = idx
                for i in range(n):
                    t[i] = rem % q
                    rem //= q
                return tuple(t)
                
            def tuple_to_idx_gen(t):
                idx = 0
                for i in range(n):
                    idx += t[i] * powers[i]
                return idx

            balls = [None] * total_words
            for i in range(total_words):
                w = idx_to_tuple_gen(i)
                ball = {i}
                for pos in range(n):
                    orig_val = w[pos]
                    for new_val in range(q):
                        if new_val != orig_val:
                            new_w = list(w)
                            new_w[pos] = new_val
                            new_idx = tuple_to_idx_gen(tuple(new_w))
                            ball.add(new_idx)
                balls[i] = ball

            uncovered = set(range(total_words))
            code_indices = []
            
            while uncovered:
                best_c = -1
                max_gain = 0
                for c in range(total_words):
                    gain = len(balls[c] & uncovered)
                    if gain > max_gain:
                        max_gain = gain
                        best_c = c
                if best_c == -1:
                    break
                code_indices.append(best_c)
                uncovered -= balls[best_c]
                
                if time.time() > deadline:
                    break
            
            result = [list(idx_to_tuple_gen(c)) for c in code_indices]
            return result
        else:
            # Fallback for large spaces: return a grid or something
            # This is a placeholder
            return [[0] * n]
