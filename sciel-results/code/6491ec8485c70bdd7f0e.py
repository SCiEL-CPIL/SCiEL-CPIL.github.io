import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 120.0  # Safety margin from 150s limit
    
    # Fallback for other parameters
    if q != 3 or n != 6 or radius != 1:
        return _greedy_cover(q, n, radius)

    # Specific optimization for 3, 6, 1
    N = q ** n  # 729
    
    # 1. Generate all words and their indices
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # 2. Precompute balls as bitsets (Python integers)
    # Ball size is 1 + n*(q-1) = 1 + 6*2 = 13
    balls = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig: continue
                new_w = w.copy()
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        balls[i] = mask
        
    full_mask = (1 << N) - 1
    
    # 3. Initialization: Tetracode Direct Sum
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # Direct sum with length 2 tail
    init_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    init_code_indices = [word_to_idx[tuple(w)] for w in init_code_words]
    
    # 4. State Initialization
    code_set = set(init_code_indices)
    # Coverage counts: how many codewords cover each word
    coverage_counts = np.zeros(N, dtype=np.int32)
    for idx in code_set:
        # Efficiently update counts using numpy?
        # balls are bitsets, hard to vectorize directly with numpy without unpacking.
        # But N=729 is small. We can just iterate.
        # However, doing this for 81 codewords * 13 neighbors is fast.
        mask = balls[idx]
        # Extract indices from mask
        # Faster: precompute list of neighbors for each ball?
        # Let's precompute neighbor lists for fast count updates.
        pass

    # Precompute neighbor lists for faster count updates
    neighbor_lists = []
    for i in range(N):
        mask = balls[i]
        neighbors = []
        # Extract set bits
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            neighbors.append(idx)
            m ^= lsb
        neighbor_lists.append(neighbors)
        
    # Re-initialize counts
    for idx in code_set:
        for nb in neighbor_lists[idx]:
            coverage_counts[nb] += 1
            
    # Helper: Check if a word can be removed
    def can_remove(idx):
        # Check if any neighbor has count 1
        for nb in neighbor_lists[idx]:
            if coverage_counts[nb] == 1:
                return False
        return True

    # Helper: Perform removal
    def do_remove(idx):
        code_set.remove(idx)
        for nb in neighbor_lists[idx]:
            coverage_counts[nb] -= 1

    # Helper: Perform addition
    def do_add(idx):
        code_set.add(idx)
        for nb in neighbor_lists[idx]:
            coverage_counts[nb] += 1

    # Helper: Check if swap (out, in) is valid
    # Condition: All words covered ONLY by 'out' must be covered by 'in'
    # Words covered only by 'out' are those in Ball[out] where count==1.
    # Note: 'in' is not in code_set yet, so its coverage is 0 in counts.
    # So "count==1" in current state means "covered only by 'out' among current code".
    # If we remove 'out', these words become 0. If we add 'in', they become 1 if in Ball[in].
    # So we need: For all w in Ball[out] with count[w]==1, w must be in Ball[in].
    def is_valid_swap(idx_out, idx_in):
        # Quick check: if idx_in is in code_set, it's not a swap, it's a no-op or error.
        # We assume idx_in not in code_set.
        # Iterate neighbors of idx_out
        for nb in neighbor_lists[idx_out]:
            if coverage_counts[nb] == 1:
                # This word is uniquely covered by idx_out
                # Check if idx_in covers it
                if not (balls[idx_in] & (1 << nb)):
                    return False
        return True

    best_size = len(code_set)
    best_code_set = set(code_set)
    
    # Local Search Loop
    # We use a simple loop: try removals, then swaps.
    # To avoid getting stuck, we randomize the order of checking removals and swaps.
    
    no_improve_steps = 0
    max_no_improve = 5000 # If stuck for too long, maybe restart? Or just keep going.
    
    while time.time() - start_time < deadline - 5.0:
        # 1. Try to find a removable codeword
        removed = False
        # Randomize the order of codewords to check for removal to avoid bias
        # We can just iterate through a shuffled list of current code
        current_list = list(code_set)
        random.shuffle(current_list)
        
        for idx in current_list:
            if can_remove(idx):
                do_remove(idx)
                removed = True
                no_improve_steps = 0
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code_set = set(code_set)
                break # Restart loop with new state
                
        if removed:
            continue
            
        # 2. If no removal found, try a swap
        # Pick a random codeword to remove
        if not code_set:
            break
            
        idx_out = random.choice(list(code_set))
        
        # Find a valid idx_in
        # We can bias the search for idx_in.
        # Ideally, idx_in should cover the "unique" words of idx_out.
        # Let's just iterate through all possible idx_in not in code_set.
        # To make it faster, we can pre-filter candidates.
        # But 729 is small.
        
        found_swap = False
        # Shuffle potential candidates?
        # We want to find ANY valid swap to escape the local minimum.
        # Let's try a random subset first for speed, then all if needed.
        # Actually, checking all 729 is very fast with bitsets.
        
        # Optimization: Only check idx_in that cover at least one unique word of idx_out?
        # No, if idx_out has no unique words, it's removable (handled above).
        # So idx_out MUST have unique words.
        
        unique_mask = 0
        for nb in neighbor_lists[idx_out]:
            if coverage_counts[nb] == 1:
                unique_mask |= (1 << nb)
                
        # If unique_mask is 0, it should have been removable. 
        # (This can happen if counts are inconsistent, but we maintain them).
        
        # We need an idx_in such that (unique_mask & ~balls[idx_in]) == 0
        # i.e., unique_mask is a subset of balls[idx_in]
        
        # Iterate through all indices not in code_set
        # To speed up, we can iterate through all N and check if in code_set
        # Or maintain a list of non-code indices? 
        # Given N=729, iterating all N is fine.
        
        valid_candidates = []
        for idx_in in range(N):
            if idx_in in code_set:
                continue
            # Check subset
            if (unique_mask & ~balls[idx_in]) == 0:
                valid_candidates.append(idx_in)
                
        if valid_candidates:
            # Pick a candidate.
            # Heuristic: Pick the one that maximizes overlap with the rest of the code?
            # Or just random to explore.
            # Let's pick random to avoid bias.
            idx_in = random.choice(valid_candidates)
            
            # Perform swap
            # First remove idx_out, then add idx_in
            # We must be careful with counts.
            # do_remove updates counts. do_add updates counts.
            do_remove(idx_out)
            do_add(idx_in)
            
            no_improve_steps = 0
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code_set = set(code_set)
                
            # After a swap, try to remove again immediately
            # (Loop continues, so it will check removals in next iteration)
            
        else:
            # No valid swap found for this idx_out.
            # Try another idx_out?
            # To avoid infinite loops on "no swap", we can just let the loop continue.
            # The random choice of idx_out will eventually find a path.
            no_improve_steps += 1
            
            # If we are stuck for a long time, we might want to do a "kick"
            # e.g., remove 2, add 1? Or just accept we are in a local minimum.
            # For now, we rely on the randomness of idx_out selection.

    # Convert best_code_set back to list of words
    best_code_words = [all_words[i] for i in best_code_set]
    
    # Final validity check
    # Re-calculate coverage to be safe
    final_counts = np.zeros(N, dtype=np.int32)
    for idx in best_code_set:
        for nb in neighbor_lists[idx]:
            final_counts[nb] += 1
            
    if np.all(final_counts > 0):
        return best_code_words
    else:
        # Fallback to initial if something went wrong
        return [all_words[i] for i in init_code_indices]

def _greedy_cover(q, n, radius):
    # Fallback for other parameters
    N = q ** n
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute balls
    balls = []
    for w in all_words:
        ball = set()
        ball.add(word_to_idx[tuple(w)])
        for delta in itertools.product(range(-radius, radius+1), repeat=n):
            dist = sum(abs(d) for d in delta)
            if dist <= radius:
                new_w = [0]*n
                valid = True
                for i in range(n):
                    val = w[i] + delta[i]
                    if 0 <= val < q:
                        new_w[i] = val
                    else:
                        valid = False
                        break
                if valid:
                    ball.add(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    uncovered = set(range(N))
    code_indices = []
    
    while uncovered:
        best_word = -1
        max_covered = 0
        for i in range(N):
            if i in code_indices: continue
            inter = len(balls[i] & uncovered)
            if inter > max_covered:
                max_covered = inter
                best_word = i
        if best_word == -1: break
        code_indices.append(best_word)
        uncovered -= balls[best_word]
        
    return [all_words[i] for i in code_indices]

if __name__ == "__main__":
    # Simple test
    code = construct(3, 6, 1)
    print(f"Code size: {len(code)}")
