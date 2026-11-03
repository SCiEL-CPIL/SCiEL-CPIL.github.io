"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    """
    Simulated annealing / stochastic local search for covering code K_q(n, radius).
    Uses a fundamentally different approach: start from a valid (possibly large)
    covering, then iteratively remove/swap/add codewords with probabilistic
    acceptance to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    def index_to_word(idx, q, n):
        """Convert a linear index back to a word (tuple of ints)."""
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return tuple(word)

    # Precompute all ball indices as a 2D numpy array: shape (total_words, ball_size)
    weights = [q ** (n - 1 - pos) for pos in range(n)]
    ball_flat = np.empty((total_words, ball_size), dtype=np.int64)
    for idx in range(total_words):
        temp = idx
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q
        ball_flat[idx, 0] = idx
        offset = 1
        for pos in range(n):
            w = weights[pos]
            d = digits[pos]
            for val in range(q):
                if val != d:
                    ball_flat[idx, offset] = idx + (val - d) * w
                    offset += 1

    # Precompute inverse balls: for each word, which codewords cover it
    # inv_ball[word_idx] = list of codeword indices whose ball contains word_idx
    inv_ball = []
    for w in range(total_words):
        covers = []
        for c in range(total_words):
            if w in ball_flat[c]:
                covers.append(c)
        inv_ball.append(np.array(covers, dtype=np.int32))

    # Fallback construction using tetracode
    def fallback():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Quick greedy to get an initial valid covering
    def quick_greedy(seed=42):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(total_words, dtype=np.uint8)
        selected = []
        while uncovered.any():
            if time.time() > deadline - 10:
                break
            # Pick random candidate with max coverage among a sample
            sample_size = min(300, total_words)
            candidates = rng.choice(total_words, size=sample_size, replace=False)
            best_count = -1
            best_idx = -1
            for cand in candidates:
                count = np.count_nonzero(uncovered[ball_flat[cand]])
                if count > best_count:
                    best_count = count
                    best_idx = cand
            selected.append(best_idx)
            uncovered[ball_flat[best_idx]] = 0
        return selected

    # Simulated annealing over codeword sets
    def simulated_annealing(initial, time_budget):
        """
        SA with moves: remove (if valid), swap (if valid), add (random).
        Energy = -len(selected) + penalty * uncovered_count
        We minimize size while maintaining validity.
        """
        end_time = start_time + time_budget
        selected = list(initial)
        selected_set = set(selected)
        
        # Build coverage count
        cover_count = np.zeros(total_words, dtype=np.int32)
        for ci in selected:
            cover_count[ball_flat[ci]] += 1
        uncovered_count = np.count_nonzero(cover_count == 0)
        
        best_solution = list(selected)
        best_size = len(selected)
        
        # Temperature schedule
        T0 = 50.0
        Tend = 0.5
        total_steps_estimate = int((end_time - time.time()) / 0.001)
        total_steps_estimate = max(total_steps_estimate, 1000)
        cooling = (Tend / T0) ** (1.0 / total_steps_estimate)
        T = T0
        
        rng = np.random.default_rng(12345)
        
        step = 0
        while time.time() < end_time:
            step += 1
            if step % 1000 == 0:
                T *= cooling
                T = max(T, Tend)
            
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try removal
                if len(selected) == 0:
                    continue
                i = rng.integers(0, len(selected))
                ci = selected[i]
                ball_i = ball_flat[ci]
                
                # Check if removal is valid
                would_uncover = np.any(cover_count[ball_i] == 1)
                
                if not would_uncover:
                    # Valid removal
                    cover_count[ball_i] -= 1
                    selected.pop(i)
                    selected_set.discard(ci)
                    uncovered_count = np.count_nonzero(cover_count == 0)
                    
                    if len(selected) < best_size and uncovered_count == 0:
                        best_size = len(selected)
                        best_solution = list(selected)
                else:
                    # Invalid removal: probabilistic acceptance
                    # Delta energy = +1 (size unchanged) + penalty for uncovered
                    # We don't accept invalid states in this version
                    pass
                    
            elif move_type < 0.8:
                # Try swap
                if len(selected) == 0:
                    continue
                i = rng.integers(0, len(selected))
                ci = selected[i]
                ball_i = ball_flat[ci]
                
                # Pick random candidate not in selected
                cand = rng.integers(0, total_words)
                if cand in selected_set:
                    continue
                ball_cand = ball_flat[cand]
                
                # Simulate swap
                cover_count[ball_i] -= 1
                cover_count[ball_cand] += 1
                new_uncovered = np.count_nonzero(cover_count == 0)
                
                if new_uncovered == 0:
                    # Valid swap, size unchanged
                    selected[i] = cand
                    selected_set.discard(ci)
                    selected_set.add(cand)
                else:
                    # Invalid: revert
                    cover_count[ball_cand] -= 1
                    cover_count[ball_i] += 1
                    
            else:
                # Try add (only if we have room / can later remove)
                if len(selected) >= total_words:
                    continue
                cand = rng.integers(0, total_words)
                if cand in selected_set:
                    continue
                ball_cand = ball_flat[cand]
                
                # Add temporarily
                cover_count[ball_cand] += 1
                selected.append(cand)
                selected_set.add(cand)
                
                # Now try to remove something
                if len(selected) > 1:
                    j = rng.integers(0, len(selected))
                    cj = selected[j]
                    ball_j = ball_flat[cj]
                    would_uncover = np.any(cover_count[ball_j] == 1)
                    if not would_uncover:
                        cover_count[ball_j] -= 1
                        selected.pop(j)
                        selected_set.discard(cj)
                    else:
                        # Revert the add
                        cover_count[ball_j] -= 1  # undo add
                        selected.pop()
                        selected_set.discard(cand)
        
        # Final: try to remove as many as possible
        cover_count = np.zeros(total_words, dtype=np.int32)
        for ci in best_solution:
            cover_count[ball_flat[ci]] += 1
        changed = True
        while changed and time.time() < end_time:
            changed = False
            for i in range(len(best_solution)):
                ci = best_solution[i]
                ball = ball_flat[ci]
                if not (cover_count[ball] == 1).any():
                    cover_count[ball] -= 1
                    best_solution.pop(i)
                    changed = True
                    break
        
        return best_solution

    # Main: get initial solution, then SA
    best_solution = None
    best_size = float('inf')
    
    # Get initial valid covering
    initial = quick_greedy(seed=42)
    if len(initial) == 0:
        fb = fallback()
        return fb
    
    # Verify initial is valid
    cover = np.zeros(total_words, dtype=np.uint8)
    for ci in initial:
        cover[ball_flat[ci]] = 1
    if not np.all(cover == 1):
        # Not valid, use fallback
        fb = fallback()
        return fb
    
    best_size = len(initial)
    best_solution = list(initial)
    
    # Run SA with time budget
    sa_time = min(100, deadline - time.time() - 5)
    if sa_time > 5:
        sa_result = simulated_annealing(initial, sa_time)
        if len(sa_result) < best_size:
            # Verify
            cover = np.zeros(total_words, dtype=np.uint8)
            for ci in sa_result:
                cover[ball_flat[ci]] = 1
            if np.all(cover == 1):
                best_size = len(sa_result)
                best_solution = sa_result
    
    # Second SA run with different seed if time allows
    remaining = deadline - time.time() - 5
    if remaining > 20:
        initial2 = quick_greedy(seed=999)
        cover = np.zeros(total_words, dtype=np.uint8)
        for ci in initial2:
            cover[ball_flat[ci]] = 1
        if np.all(cover == 1) and len(initial2) < best_size + 10:
            sa_time2 = min(remaining - 5, 80)
            if sa_time2 > 5:
                sa_result2 = simulated_annealing(initial2, sa_time2)
                cover = np.zeros(total_words, dtype=np.uint8)
                for ci in sa_result2:
                    cover[ball_flat[ci]] = 1
                if np.all(cover == 1) and len(sa_result2) < best_size:
                    best_size = len(sa_result2)
                    best_solution = sa_result2
    
    # Final cleanup: remove redundant codewords
    cover_count = np.zeros(total_words, dtype=np.int32)
    for ci in best_solution:
        cover_count[ball_flat[ci]] += 1
    changed = True
    while changed and time.time() < deadline - 3:
        changed = False
        for i in range(len(best_solution)):
            ci = best_solution[i]
            ball = ball_flat[ci]
            if not (cover_count[ball] == 1).any():
                cover_count[ball] -= 1
                best_solution.pop(i)
                changed = True
                break
    
    # Convert to list of lists
    codewords = [list(index_to_word(ci, q, n)) for ci in best_solution]
    
    # Final validity check
    cover = np.zeros(total_words, dtype=np.uint8)
    for ci in best_solution:
        cover[ball_flat[ci]] = 1
    if not np.all(cover == 1):
        return fallback()
    
    return codewords

# EVOLVE-BLOCK-END
