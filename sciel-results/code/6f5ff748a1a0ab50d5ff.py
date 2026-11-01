import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius
    'radius'. 
    Optimized specifically for q=3, n=7, radius=1 using bitset-based greedy set cover
    and iterative local search (removal and swap heuristics).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        # 1. Precomputation
        all_words = list(itertools.product(range(3), repeat=7))
        N = len(all_words)  # 2187
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute balls as bitsets (Python ints)
        # bit i is set if word i is in the ball
        balls = [0] * N
        for w in all_words:
            idx = word_to_idx[w]
            mask = 1 << idx
            for i in range(7):
                for val in range(3):
                    if val != w[i]:
                        new_word = list(w)
                        new_word[i] = val
                        new_idx = word_to_idx[tuple(new_word)]
                        mask |= (1 << new_idx)
            balls[idx] = mask
        
        # Total mask for all words
        full_mask = (1 << N) - 1

        best_code_size = N
        best_codewords = list(range(N)) # Fallback: all words

        # We will run several rounds of Greedy + Local Search
        # Each round has a time budget
        max_rounds = 20
        round_time_limit = (deadline - time.time()) / max_rounds
        
        for seed in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            random.seed(seed)
            
            # 2. Greedy Set Cover with Bitsets
            covered_mask = 0
            codewords = []
            
            # To speed up greedy, we can just scan all candidates each time
            # Since N=2187, and we pick ~180 codewords, this is 180 * 2187 operations.
            # Each operation is an AND and POPCOUNT (bit_count in Python 3.8+).
            # This is very fast.
            
            while covered_mask != full_mask:
                best_cw = -1
                best_gain = -1
                
                # Optimization: If we have many candidates, checking all is fine.
                # We want to maximize (balls[cw] & ~covered_mask).bit_count()
                
                # We can also use a heuristic: prioritize candidates that cover many
                # currently uncovered words.
                
                for i in range(N):
                    # Skip if already in codewords? 
                    # Greedy usually allows re-picking, but since balls are fixed, 
                    # picking the same one again yields 0 gain.
                    # However, checking `i in codewords` is O(K). 
                    # It's faster to just check gain. If gain is 0, it's redundant.
                    
                    gain_mask = balls[i] & ~covered_mask
                    gain = gain_mask.bit_count()
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_cw = i
                        
                        # Early break if perfect gain? Max gain is 15.
                        if best_gain == 15:
                            # Can't do better than covering 15 new words
                            # (unless there are overlaps, but max ball size is 15)
                            pass 
                            
                if best_cw == -1 or best_gain == 0:
                    break # Should not happen if covered_mask != full_mask
            
                codewords.append(best_cw)
                covered_mask |= balls[best_cw]
                
                if time.time() > deadline - 5:
                    break

            if len(codewords) < best_code_size:
                best_code_size = len(codewords)
                best_codewords = list(codewords)

            # 3. Local Search Optimization
            # We want to minimize the size.
            # Strategy: 
            # a) Try to remove any redundant codeword.
            # b) If no removal, try to swap a codeword to a better position 
            #    that might enable a removal later.
            
            current_code = list(codewords)
            
            # Helper to check if a codeword can be removed
            # A codeword c can be removed if for all w in ball(c), 
            # there exists another c' in code \ {c} such that w in ball(c').
            
            def can_remove(idx_to_remove, code_list):
                b = balls[code_list[idx_to_remove]]
                # Check if b is covered by union of other balls
                # Union of other balls = (OR of all balls) & ~(balls[idx_to_remove])? No.
                # We need: b | (OR of others) == OR of all? 
                # Actually, simpler: 
                # Let U = OR of all balls in code_list.
                # Let U_no_i = OR of balls in code_list except i.
                # Can remove i if (b_i & ~U_no_i) == 0. i.e., b_i is subset of U_no_i.
                
                # Calculating U_no_i from scratch is O(K).
                # We can maintain the OR mask.
                # But for simplicity in this loop, let's just iterate if K is small.
                # K is ~180. 180 * 180 = 32400 ops. Very fast.
                
                others_mask = 0
                for j, cw in enumerate(code_list):
                    if j != idx_to_remove:
                        others_mask |= balls[cw]
                
                return (b & ~others_mask) == 0

            def try_remove_pass(code_list):
                removed = False
                # Shuffle to avoid bias
                indices = list(range(len(code_list)))
                random.shuffle(indices)
                
                for idx in indices:
                    if idx >= len(code_list): continue
                    if can_remove(idx, code_list):
                        code_list.pop(idx)
                        removed = True
                        # After removal, restart scanning? 
                        # Or just continue? Removing one might make others redundant.
                        # Restarting is safer and often faster to convergence.
                        return True
                return removed

            # Aggressive removal
            while try_remove_pass(current_code):
                pass

            if len(current_code) < best_code_size:
                best_code_size = len(current_code)
                best_codewords = list(current_code)

            # If stuck, try swaps
            # A swap replaces c_old with c_new.
            # Valid if coverage is maintained.
            # We want to find a swap that allows a subsequent removal.
            
            # Heuristic: 
            # 1. Calculate "slack" for each word: how many codewords cover it.
            # 2. Identify words with slack 1 (critical).
            # 3. A codeword covering only critical words is "essential".
            # 4. Try to move non-essential codewords to positions that cover more 
            #    words with slack > 1, potentially freeing up other codewords.
            
            # Simpler Swap Heuristic:
            # Randomly pick a codeword c_old.
            # Randomly pick a new codeword c_new.
            # If c_new is not in code:
            #   Simulate swap.
            #   If valid, accept.
            #   Then try to remove.
            
            local_search_time_limit = time.time() + 2.0 # 2 seconds per round
            while time.time() < local_search_time_limit:
                if not current_code:
                    break
                
                improved = False
                # Try removal first
                if try_remove_pass(current_code):
                    improved = True
                    if len(current_code) < best_code_size:
                        best_code_size = len(current_code)
                        best_codewords = list(current_code)
                    continue
                
                # If no removal, try a swap
                idx_old = random.randint(0, len(current_code) - 1)
                cw_old = current_code[idx_old]
                
                # Choose cw_new
                # Prefer cw_new that covers words not well covered by others?
                # Or just random.
                cw_new = random.randint(0, N - 1)
                
                if cw_new == cw_old or cw_new in current_code:
                    continue
                
                # Simulate swap
                # New coverage = (Old Coverage - Ball(cw_old)) + Ball(cw_new)
                # We need New Coverage to cover all words.
                # i.e., (Old Coverage - Ball(cw_old) + Ball(cw_new)) == Full Mask?
                # Actually, we just need it to cover all words that were previously covered.
                # Since Old Coverage == Full Mask, we need:
                # (Full Mask - Ball(cw_old) + Ball(cw_new)) == Full Mask
                # This is equivalent to: Ball(cw_old) is subset of (Old Coverage - Ball(cw_old) + Ball(cw_new))?
                # No. 
                # Let C be the set of covered words. C = Full.
                # C' = (C \ Ball(cw_old)) \ Ball(cw_old) | Ball(cw_new) ?? No.
                # C' = (C \ Ball(cw_old)) | Ball(cw_new)
                # We need C' == C.
                # This holds if Ball(cw_old) \ C' is empty.
                # Words in Ball(cw_old) that are NOT in C' are those in Ball(cw_old) that are not in (C \ Ball(cw_old)) and not in Ball(cw_new).
                # Since C = Full, C \ Ball(cw_old) = Full \ Ball(cw_old).
                # So words in Ball(cw_old) not in C' are:
                # (Ball(cw_old) \ (Full \ Ball(cw_old))) \ Ball(cw_new)
                # = Ball(cw_old) \ Ball(cw_new)
                # So the swap is valid if Ball(cw_old) \ Ball(cw_new) is covered by the *other* codewords.
                
                # Let Others = Union of balls of all codewords except idx_old.
                # Condition: (Ball(cw_old) & ~Ball(cw_new)) & ~Others == 0
                
                others_mask = 0
                for j, cw in enumerate(current_code):
                    if j != idx_old:
                        others_mask |= balls[cw]
                
                # Check validity
                if (balls[cw_old] & ~balls[cw_new] & ~others_mask) == 0:
                    # Valid swap
                    current_code[idx_old] = cw_new
                    improved = True
                    # After swap, try to remove
                    if try_remove_pass(current_code):
                        if len(current_code) < best_code_size:
                            best_code_size = len(current_code)
                            best_codewords = list(current_code)

            # If no improvement in this round, we can just let it go.
            # The restarts handle escaping local minima.

        # Final cleanup: ensure no duplicates (shouldn't happen) and return
        unique_codewords = list(set(best_codewords))
        
        # One final removal pass on the best found
        final_code = unique_codewords
        while try_remove_pass(final_code):
            pass

        return [list(all_words[cw]) for cw in final_code]

    # General fallback (not optimized for this specific case)
    # This part is unlikely to be reached for the specific task contract
    # but keeps the function general as per signature.
    
    all_words = list(itertools.product(range(q), repeat=n))
    
    def get_coverage_general(codeword):
        covered = set()
        covered.add(codeword)
        for i in range(n):
            for val in range(q):
                if val != codeword[i]:
                    new_word = list(codeword)
                    new_word[i] = val
                    covered.add(tuple(new_word))
        return covered
    
    codewords = []
    covered = set()
    
    while time.time() < deadline - 10:
        best_cw = None
        best_count = -1
        best_new_covered = set()
        
        for candidate in all_words:
            new_covered = get_coverage_general(candidate) - covered
            count = len(new_covered)
            if count > best_count:
                best_count = count
                best_cw = candidate
                best_new_covered = new_covered
        
        if best_count <= 0:
            break
        
        codewords.append(best_cw)
        covered.update(best_new_covered)
        
        if len(covered) == len(all_words):
            break
    
    # Redundancy removal
    i = 0
    while i < len(codewords) and time.time() < deadline - 5:
        cw_to_remove = codewords[i]
        coverage_of_cw = get_coverage_general(cw_to_remove)
        
        can_remove = True
        for w in coverage_of_cw:
            is_covered_by_other = False
            for j, other_cw in enumerate(codewords):
                if i == j:
                    continue
                if w in get_coverage_general(other_cw):
                    is_covered_by_other = True
                    break
            if not is_covered_by_other:
                can_remove = False
                break
        
        if can_remove:
            codewords.pop(i)
        else:
            i += 1
    
    return [list(cw) for cw in codewords]
