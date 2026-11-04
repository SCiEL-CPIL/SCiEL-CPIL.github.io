import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    # Fallback for other parameters or if specific optimization fails
    # For the specific task K_3(6,1), we use the optimized path.
    if q == 3 and n == 6 and radius == 1:
        return _solve_3_6_1()
    else:
        # Generic greedy fallback
        return _generic_greedy(q, n, radius)

def _solve_3_6_1():
    q = 3
    n = 6
    radius = 1
    total_words = q ** n
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # Map word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute balls: for each word index, list of indices covered
    balls = []
    for i, w in enumerate(all_words):
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                w_new = w[:]
                w_new[pos] = new_val
                ball.append(word_to_idx[tuple(w_new)])
        balls.append(ball)
        
    # Greedy Initialization with random tie-breaking
    def greedy_init(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        remaining = set(range(total_words))
        code = []
        # Pre-compute coverage counts for greedy? No, just count intersection with remaining.
        while remaining:
            best_word = -1
            max_covered = -1
            # To speed up, we can just iterate all words and count how many in 'remaining' are in its ball
            # Since total_words is 729, and ball size 13, this is fast.
            # Optimization: pick a random word from remaining to break ties?
            # Or just pick the one with max coverage. If tie, pick random among max.
            candidates = []
            max_cov = 0
            for w_idx in range(total_words):
                covered_in_remaining = sum(1 for x in balls[w_idx] if x in remaining)
                if covered_in_remaining > max_cov:
                    max_cov = covered_in_remaining
                    candidates = [w_idx]
                elif covered_in_remaining == max_cov and max_cov > 0:
                    candidates.append(w_idx)
            
            if not candidates:
                break
                
            chosen = random.choice(candidates)
            code.append(chosen)
            # Remove covered words from remaining
            for x in balls[chosen]:
                if x in remaining:
                    remaining.remove(x)
                    
        return code

    # Local Search
    def local_search(initial_code, deadline):
        code = list(initial_code)
        # cover_count[i] = number of codewords in code that cover word i
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cover_count[balls[c]] += 1
            
        best_code = list(code)
        best_size = len(code)
        
        # Check if valid
        if np.min(cover_count) < 1:
             # Should not happen with greedy, but safety
             pass

        # Removal phase
        def try_removals():
            nonlocal code, cover_count, best_code, best_size
            removed_any = False
            # Shuffled indices to randomize removal order
            indices = list(range(len(code)))
            random.shuffle(indices)
            for idx in indices:
                c = code[idx]
                # Check if removable
                # A word is removable if all words in its ball have cover_count > 1
                # Note: if we remove c, cover_count for balls[c] decreases by 1.
                # So we need min(cover_count[balls[c]]) > 1
                if np.min(cover_count[balls[c]]) > 1:
                    # Remove it
                    cover_count[balls[c]] -= 1
                    code.pop(idx)
                    removed_any = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    # Break to restart scanning? Or continue?
                    # Continuing might be dangerous if indices shift.
                    # We popped idx, so indices > idx are shifted.
                    # Let's just break and restart the outer loop iteration.
                    break
            return removed_any

        # Swap phase
        def try_swap():
            nonlocal code, cover_count
            if len(code) == 0:
                return False
            # Pick random codeword to swap out
            i = random.randint(0, len(code) - 1)
            c_old = code[i]
            
            # Pick random word to swap in
            # It must not be in code
            # Simple: pick random index, if in code, retry a few times
            c_new = random.randint(0, total_words - 1)
            if c_new in code:
                # Try a few times
                for _ in range(10):
                    c_new = random.randint(0, total_words - 1)
                    if c_new not in code:
                        break
                else:
                    return False # Couldn't find a new word

            # Test swap: remove c_old, add c_new
            # Tentatively update
            old_ball = balls[c_old]
            new_ball = balls[c_new]
            
            # Check if any word in old_ball becomes 0
            # And any word in new_ball is currently 0 (which would be bad, but we are adding)
            # Actually, we just need to check if min(cover_count) >= 1 after swap.
            
            # Optimization:
            # Words affected are old_ball union new_ball.
            # For w in old_ball: count decreases. If it was 1, it becomes 0. BAD.
            # For w in new_ball: count increases.
            # So we only need to check if any w in old_ball has cover_count[w] == 1 
            # AND w is NOT in new_ball (because if it's in new_ball, it increases back).
            
            # Let's do a quick check
            removable = True
            for w in old_ball:
                if cover_count[w] == 1:
                    # Check if w is in new_ball
                    # If w is in new_ball, the count will go 1 -> 0 -> 1. Safe.
                    # If w is not in new_ball, count goes 1 -> 0. Unsafe.
                    if w not in new_ball:
                        removable = False
                        break
            
            if not removable:
                return False
            
            # Perform swap
            code[i] = c_new
            cover_count[old_ball] -= 1
            cover_count[new_ball] += 1
            return True

        start_time = time.time()
        
        # Initial removals
        while try_removals():
            pass
        
        # Update best
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        # Main loop
        while time.time() < deadline:
            if try_removals():
                # If we removed, we improved size. Update best.
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                # Try more removals
                while try_removals():
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
            else:
                # No removals, try swaps
                # Do a batch of swaps to move around
                swaps_per_iter = 1000
                for _ in range(swaps_per_iter):
                    if time.time() >= deadline:
                        break
                    try_swap()
                
                # After some swaps, try removals again
                if try_removals():
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    while try_removals():
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = list(code)

        return [all_words[i] for i in best_code]

    deadline = time.time() + 140
    
    # Run multiple iterations of Greedy + Local Search
    best_overall = []
    best_size_overall = float('inf')
    
    start = time.time()
    iter_count = 0
    while time.time() < deadline - 5: # Leave 5s margin
        seed = iter_count
        code = greedy_init(seed)
        # Run local search with a sub-deadline
        sub_deadline = time.time() + 10 # 10s per iteration approx
        if time.time() + 10 > deadline - 5:
            sub_deadline = deadline - 5
            
        result = local_search(code, sub_deadline)
        
        if len(result) < best_size_overall:
            best_size_overall = len(result)
            best_overall = result
            
        iter_count += 1
        if time.time() > deadline - 5:
            break
            
    if best_overall:
        return best_overall
    else:
        # Fallback to seed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

def _generic_greedy(q, n, radius):
    # Simple fallback
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    total = len(all_words)
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute balls
    balls = []
    for i, w in enumerate(all_words):
        ball = [i]
        # Generate neighbors within radius
        # For radius 1, it's simple. For general, use itertools.
        # Given constraints, assume radius is small or n is small.
        # For safety, just use Hamming distance check if n*q is small.
        for j, w2 in enumerate(all_words):
            dist = sum(1 for a, b in zip(w, w2) if a != b)
            if dist <= radius:
                ball.append(j)
        balls.append(ball)
        
    remaining = set(range(total))
    code = []
    while remaining:
        best_word = -1
        max_cov = -1
        for w_idx in range(total):
            cov = sum(1 for x in balls[w_idx] if x in remaining)
            if cov > max_cov:
                max_cov = cov
                best_word = w_idx
        if best_word == -1:
            break
        code.append(best_word)
        for x in balls[best_word]:
            if x in remaining:
                remaining.remove(x)
                
    return [all_words[i] for i in code]
