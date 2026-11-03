import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Total number of words
    total_words = q ** n
    
    # Map word (tuple) to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Generate all words
    words = []
    for i in range(total_words):
        w = []
        x = i
        for j in range(n):
            w.append(x % q)
            x //= q
        words.append(tuple(w))
        
    # Precompute balls (indices of words within radius 1)
    # For q=3, n=6, radius=1:
    # Ball size = 1 + n*(q-1) = 1 + 6*2 = 13
    balls = []
    for w in words:
        ball = [word_to_idx(w)]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.append(word_to_idx(tuple(new_w)))
        balls.append(ball)
        
    # Greedy Initialization
    # cover_count[i] = number of selected codewords covering word i
    cover_count = np.zeros(total_words, dtype=np.int32)
    selected = []
    
    # To speed up greedy, we can just pick random words initially if we plan to local search heavily,
    # but Greedy is a good start.
    
    # Standard Greedy
    uncovered = set(range(total_words))
    # For efficiency in greedy, we might just pick random words until covered, then local search.
    # But Greedy is deterministic and good.
    
    # Let's use a slightly randomized greedy to avoid bad local minima early?
    # No, standard greedy is fine.
    
    # Optimization: Instead of recalculating gain for all 729 words every step (expensive),
    # we can just add random words until covered, then rely on Local Search to optimize.
    # Given 729 words, adding ~80 random words will cover most.
    # Let's try Random Initialization + Local Search. It's often more robust for small instances.
    
    # Random Init
    num_init = 80
    selected_indices = random.sample(range(total_words), num_init)
    selected = selected_indices
    
    for idx in selected:
        for w_idx in balls[idx]:
            cover_count[w_idx] += 1
            
    # Check if covered
    while np.any(cover_count == 0):
        # Find uncovered word
        uncov_idx = np.where(cover_count == 0)[0][0]
        # Pick a random codeword that covers it? Or just pick a random codeword from its ball?
        # The word uncov_idx is in the ball of some codeword.
        # Actually, we need to ADD a codeword.
        # Let's add a random codeword that covers uncov_idx.
        # Who covers uncov_idx?
        # We didn't precompute "who covers w".
        # But we know w is in balls[c] if c is in the ball of w.
        # Since Hamming distance is symmetric, balls[uncov_idx] contains all c that cover uncov_idx.
        candidates = balls[uncov_idx]
        c = random.choice(candidates)
        if c not in selected:
            selected.append(c)
            for w_idx in balls[c]:
                cover_count[w_idx] += 1
        else:
            # Should not happen if we check cover_count correctly, but safe
            pass

    best_selected = list(selected)
    best_size = len(selected)
    
    # Local Search
    # State: selected (list of indices), cover_count (numpy array)
    
    def try_remove(idx):
        """Attempts to remove idx. Returns True if successful."""
        nonlocal cover_count, selected
        # Check if any word in balls[idx] has cover_count == 1
        # If so, removing idx would make it 0.
        ball = balls[idx]
        for w_idx in ball:
            if cover_count[w_idx] == 1:
                return False
        # Safe to remove
        for w_idx in ball:
            cover_count[w_idx] -= 1
        selected.remove(idx)
        return True

    def try_swap(old_idx, new_idx):
        """Attempts to swap old_idx with new_idx. Returns True if successful."""
        nonlocal cover_count, selected
        # 1. Check if removing old_idx is safe? No, we add new_idx first?
        # If we remove old_idx, some words might become 0.
        # If we add new_idx, they might become >0.
        # We need to check if (cover_count - old_ball + new_ball) has any 0s.
        
        # Efficient check:
        # Words affected are union of balls[old_idx] and balls[new_idx].
        # Let's just simulate.
        
        # Save old state? No, just check.
        # A word w is covered in new state if:
        # cover_count[w] - (1 if w in ball_old else 0) + (1 if w in ball_new else 0) > 0
        
        ball_old = balls[old_idx]
        ball_new = balls[new_idx]
        
        # Check all words in ball_old
        for w_idx in ball_old:
            new_count = cover_count[w_idx] - 1
            if w_idx in ball_new: # This 'in' is slow. Use a set or boolean array?
                new_count += 1
            if new_count == 0:
                return False
                
        # Check all words in ball_new (that are not in ball_old)
        # Actually, if w is in ball_new but not ball_old, new_count = cover_count[w] + 1.
        # Since cover_count[w] >= 0, new_count >= 1. So only words in ball_old are at risk.
        # Wait, what if w is in ball_old and ball_new?
        # new_count = cover_count[w] - 1 + 1 = cover_count[w].
        # If cover_count[w] >= 1, it's fine.
        # The only risk is w in ball_old, not in ball_new, and cover_count[w] == 1.
        
        # So we just need to check w in ball_old \ ball_new with cover_count[w] == 1.
        # To do this fast, we can just iterate ball_old and check condition.
        
        for w_idx in ball_old:
            if cover_count[w_idx] == 1:
                # Check if w_idx is covered by new_idx
                # Is w_idx in ball_new?
                # We can check distance or just loop ball_new (size 13).
                # Or precompute a set for ball_new?
                # Given small size, checking if w_idx is in ball_new by loop is O(13).
                # 13 * 13 = 169 ops per swap check. Very fast.
                found = False
                for nw in ball_new:
                    if nw == w_idx:
                        found = True
                        break
                if not found:
                    return False
        
        # Valid swap
        # Update counts
        for w_idx in ball_old:
            cover_count[w_idx] -= 1
        for w_idx in ball_new:
            cover_count[w_idx] += 1
        selected.remove(old_idx)
        selected.append(new_idx)
        return True

    # Iterated Local Search
    stall_count = 0
    last_improvement = time.time()
    
    while time.time() < deadline:
        # 1. Deletion Pass
        removed_any = False
        # Shuffle selected to avoid bias
        current_selected = list(selected)
        random.shuffle(current_selected)
        
        for idx in current_selected:
            if idx not in selected:
                continue
            if try_remove(idx):
                removed_any = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                    last_improvement = time.time()
        
        if removed_any:
            stall_count = 0
            continue
        else:
            stall_count += 1
            
        # 2. If stuck, try Perturbation (Swaps)
        if stall_count > 10:
            # Try to find a valid swap
            swap_found = False
            # Pick a random codeword to remove
            old_idx = random.choice(selected)
            # Pick a random codeword to add (not in selected)
            # To speed up, pick from a random subset of all words
            for _ in range(100):
                new_idx = random.randint(0, total_words - 1)
                if new_idx in selected:
                    continue
                if try_swap(old_idx, new_idx):
                    swap_found = True
                    break
            
            if not swap_found:
                # Try a more aggressive perturbation:
                # Remove 2, Add 2? Or just restart local search from a slightly perturbed state.
                # Let's try removing a random codeword (making it invalid temporarily? No, we must maintain validity for the swap logic).
                # If we can't swap, we are in a "minimal" code.
                # We can try to "jump" by replacing 2 codewords with 2 others.
                # This is complex. Let's just try more random swaps with different old_idx.
                
                for _ in range(50):
                    old_idx = random.choice(selected)
                    new_idx = random.randint(0, total_words - 1)
                    if new_idx in selected:
                        continue
                    if try_swap(old_idx, new_idx):
                        swap_found = True
                        break
            
            if swap_found:
                # After a swap, we might have created a redundancy
                # The loop will go back to Deletion Pass
                stall_count = 0
            else:
                # If we can't swap, we might be stuck in a local minimum of size S.
                # We can try to accept a "worse" move? No, we want to minimize size.
                # If we are stuck, we should probably restart the initialization with a different random seed?
                # Or just keep trying swaps.
                
                # If we've been stuck for a long time, restart
                if time.time() - last_improvement > 5.0:
                    # Restart with random init
                    selected = []
                    cover_count = np.zeros(total_words, dtype=np.int32)
                    num_init = random.randint(70, 90)
                    selected_indices = random.sample(range(total_words), num_init)
                    selected = selected_indices
                    for idx in selected:
                        for w_idx in balls[idx]:
                            cover_count[w_idx] += 1
                    # Ensure coverage
                    while np.any(cover_count == 0):
                        uncov_idx = np.where(cover_count == 0)[0][0]
                        candidates = balls[uncov_idx]
                        c = random.choice(candidates)
                        if c not in selected:
                            selected.append(c)
                            for w_idx in balls[c]:
                                cover_count[w_idx] += 1
                    stall_count = 0
                    last_improvement = time.time()

    # Return the best found
    return [list(words[i]) for i in best_selected]
