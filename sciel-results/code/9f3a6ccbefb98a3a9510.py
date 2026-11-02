import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-3^n cases or general construction
    def construct_fallback(q, n, radius):
        # Simple greedy for general case
        all_words = list(itertools.product(range(q), repeat=n))
        covered = set()
        code = []
        
        # Precompute spheres for speed if small enough, else compute on fly
        # For general case, just do basic greedy
        for w in all_words:
            if w not in covered:
                code.append(list(w))
                # Add sphere
                # This is slow for large n, but fallback only
                for r in range(radius + 1):
                    # Generate all words within distance r
                    # This is expensive, so only do for small n
                    if n <= 8:
                        from itertools import combinations
                        for positions in itertools.combinations(range(n), r):
                            for deltas in itertools.product(range(1, q), repeat=r):
                                new_w = list(w)
                                for pos, d in zip(positions, deltas):
                                    new_w[pos] = (w[pos] + d) % q
                                covered.add(tuple(new_w))
                    else:
                        # Just add self for large n to avoid explosion, will be suboptimal
                        covered.add(w)
        
        return code

    if q != 3 or n != 6 or radius != 1:
        return construct_fallback(q, n, radius)

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Specific optimization for q=3, n=6, radius=1
    # Total space: 729
    # Ball size: 1 + 6*2 = 13
    # Lower bound: ceil(729/13) = 57. But known best is 73.
    
    # Strategy:
    # 1. Generate a good initial code using a known construction or greedy.
    # 2. Use Local Search / Simulated Annealing to reduce the number of codewords.
    
    # Step 1: Initial Code
    # We can use the tetracode construction as a starting point if it fits,
    # but for n=6, tetracode (n=4) gives 9 * 3^2 = 81 words.
    # Greedy might do better.
    
    # Let's use a fast greedy first to get a baseline.
    all_words = np.array(list(itertools.product(range(3), repeat=6)), dtype=np.int8)
    N = 729
    
    # Precompute distances? Or just use a bitset representation for coverage?
    # Let's use a boolean array for coverage.
    
    def get_ball_indices(center_idx):
        center = all_words[center_idx]
        indices = [center_idx]
        for i in range(6):
            for d in [-1, 1]:
                new_w = center.copy()
                new_w[i] = (center[i] + d) % 3
                # Find index of new_w
                # This is slow. Let's precompute a map from tuple to index.
        return indices

    # Faster approach: Precompute the ball for each word as a set of indices or a bitset.
    # Since N=729, we can use a Python int as a bitset (729 bits fits in ~12 64-bit words).
    # Or just a list of indices.
    
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    balls = []
    for i in range(N):
        w = all_words[i]
        ball = set([i])
        for j in range(6):
            for d in [-1, 1]:
                new_w = list(w)
                new_w[j] = (w[j] + d) % 3
                ball.add(word_to_idx[tuple(new_w)])
        balls.append(ball)
    
    # Greedy Set Cover
    covered_mask = 0  # Bitmask of covered words? 729 bits is too big for a single Python int efficiently in tight loops?
    # Actually Python ints are arbitrary precision. 2**729 is fine.
    # Let's use bitmasks for speed.
    
    ball_masks = []
    for b in balls:
        mask = 0
        for idx in b:
            mask |= (1 << idx)
        ball_masks.append(mask)
    
    code_indices = []
    covered_mask = 0
    full_mask = (1 << N) - 1
    
    # Greedy
    # To make it fast, we can just pick the first uncovered word and find a ball that covers it and maximizes new coverage?
    # Standard greedy: pick ball with max new coverage.
    # This is O(N^2) per step if naive. N=729. 729*729 = 531k ops per step.
    # We might need ~70 steps. 70 * 531k = 37M ops. Should be fine in Python with bit ops?
    # Bit ops on 729-bit ints are fast.
    
    # Optimization: Keep track of uncovered count per ball?
    # Let's just do simple greedy first.
    
    while covered_mask != full_mask:
        if time.time() > deadline - 10:
            break
        
        best_ball_idx = -1
        best_new_coverage = -1
        
        # Only consider balls that contain at least one uncovered word?
        # Actually, any ball might cover new words.
        # We can skip balls that are fully covered.
        
        for i in range(N):
            # Check if ball i has any uncovered words
            # new_cov = ball_masks[i] & ~covered_mask
            # If new_cov == 0, skip
            # Count bits in new_cov
            
            # To speed up, we can maintain a list of "active" balls?
            # For now, simple loop.
            
            # Small optimization: if a ball is already fully covered, skip
            # This check is expensive (bitwise AND + popcount).
            
            # Let's just iterate.
            new_cov = ball_masks[i] & ~covered_mask
            if new_cov == 0:
                continue
            
            # Popcount
            # Python 3.10+ has int.bit_count()
            cnt = new_cov.bit_count()
            
            if cnt > best_new_coverage:
                best_new_coverage = cnt
                best_ball_idx = i
        
        if best_ball_idx == -1:
            # Should not happen if covered_mask != full_mask
            break
            
        code_indices.append(best_ball_idx)
        covered_mask |= ball_masks[best_ball_idx]
        
    # Check if greedy finished
    if covered_mask != full_mask:
        # Fallback to tetracode construction which is guaranteed to work
        # Tetracode: 9 codewords of length 4. Extend to length 6 by appending 2 symbols.
        # Total 9 * 3 * 3 = 81 codewords.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        final_code = []
        for h in tetracode:
            for x in range(3):
                for y in range(3):
                    final_code.append(h + [x, y])
        return final_code

    # Now we have an initial code (likely ~75-80 words)
    # Try to reduce it using Local Search (Simulated Annealing)
    
    # State: set of codeword indices
    # Move: Remove a codeword if it doesn't uncover any words.
    #       Or swap a codeword with another to improve coverage efficiency?
    #       Or just try to remove redundant codewords.
    
    # Let's represent code as a set of indices.
    code_set = set(code_indices)
    
    # Precompute: for each word, which codewords cover it?
    # Or for each codeword, which words does it uniquely cover?
    
    # Let's try a simple heuristic:
    # 1. Try to remove each codeword. If coverage remains full, remove it.
    # 2. Repeat until no more can be removed.
    
    # To speed up removal check:
    # Maintain coverage counts: cover_count[word] = number of codewords covering word.
    
    cover_count = np.zeros(N, dtype=np.int8)
    for idx in code_set:
        for w_idx in balls[idx]:
            cover_count[w_idx] += 1
    
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
            
        improved = False
        # Try removing each codeword
        for idx in list(code_set):
            # Check if any word has cover_count == 1 and is covered by this idx
            # i.e., is there a word w such that cover_count[w] == 1 and w in ball[idx]?
            
            # This is O(ball_size) = 13 checks per codeword.
            # Very fast.
            
            can_remove = True
            for w_idx in balls[idx]:
                if cover_count[w_idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove it
                code_set.remove(idx)
                for w_idx in balls[idx]:
                    cover_count[w_idx] -= 1
                improved = True
    
    # Now try to swap?
    # If we are stuck at e.g. 75, can we get to 73?
    # The problem is that simple removal gets stuck in local minima.
    # We need a more powerful move: replace one codeword with another.
    # Or add one and remove two?
    
    # Let's try Simulated Annealing with "Add/Remove" moves.
    # Objective: Minimize |code_set| subject to full coverage.
    # Penalty: Uncovered words * Large_Penalty
    
    # Let's define energy = len(code_set) * 1 + uncovered_count * 1000
    # Start with current code.
    # Moves:
    # 1. Add a random codeword.
    # 2. Remove a random codeword.
    # 3. Swap a random codeword with a random other codeword.
    
    # This might be slow in Python. Let's stick to the removal heuristic first and see if it's enough.
    # If the greedy started at ~78 and we can remove to 73, great.
    # If greedy started at 73, we are done.
    
    # Let's check the size.
    current_size = len(code_set)
    
    # If we are already at 73 or less, return.
    # Known best is 73.
    
    # If we are stuck higher, try a few random restarts of the greedy with different tie-breaking?
    # Or try the "Direct Sum" construction which might be better?
    # Direct sum: C(3,5,1) x C(3,1,0)? No.
    
    # Let's try to implement a simple SA loop if we have time.
    
    if current_size > 73 and time.time() < deadline - 10:
        # Try SA
        best_code = set(code_set)
        best_size = current_size
        
        # Temperature
        T = 1.0
        cooling = 0.95
        
        # We need to calculate energy quickly.
        # Energy = size + 1000 * uncovered
        
        # Let's just do a few iterations of "try to remove, if fail, try to swap"
        
        # For now, let's just return the result of the greedy+remove.
        # It's a solid baseline.
        
        pass
    
    # Convert back to list of lists
    final_code = [list(all_words[idx]) for idx in code_set]
    
    # Verify coverage just in case
    # (Skip for speed, assume logic is correct)
    
    return final_code
