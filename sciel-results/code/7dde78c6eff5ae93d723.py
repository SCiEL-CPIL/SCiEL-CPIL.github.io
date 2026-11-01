import time
import random
import itertools
import numpy as np
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = 118.0  # Safety margin from 150s limit

    # Fallback for other parameters
    if q != 3 or n != 6 or radius != 1:
        return _greedy_cover(q, n, radius)

    # Specific optimization for 3, 6, 1
    N = q ** n  # 729
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

    # Precompute balls (sets of indices within Hamming distance 1)
    # Use bitsets via Python integers for fast intersection
    ball_masks = [0] * N
    for i, w in enumerate(all_words):
        mask = 0
        mask |= (1 << i)
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        ball_masks[i] = mask

    # Also keep balls as lists for iteration
    balls = [0] * N
    for i in range(N):
        # Convert mask to list of set bits
        m = ball_masks[i]
        lst = []
        while m:
            lsb = m & (-m)
            lst.append(lsb.bit_length() - 1)
            m ^= lsb
        balls[i] = lst

    # Greedy set cover to get an initial solution
    uncovered_mask = (1 << N) - 1
    code_indices = []
    picked = set()

    while uncovered_mask:
        best_word = -1
        max_covered = 0
        for i in range(N):
            if i in picked:
                continue
            covered = (ball_masks[i] & uncovered_mask).bit_count()
            if covered > max_covered:
                max_covered = covered
                best_word = i
        if best_word == -1:
            break
        code_indices.append(best_word)
        picked.add(best_word)
        uncovered_mask &= ~ball_masks[best_word]

    # Greedy reduction: remove redundant codewords
    def is_valid_mask(mask):
        # Check if the union of balls for all set bits covers all N words
        union = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            union |= ball_masks[idx]
            m ^= lsb
        return union == (1 << N) - 1

    code_mask = 0
    for idx in code_indices:
        code_mask |= (1 << idx)

    changed = True
    while changed:
        changed = False
        m = code_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            new_mask = code_mask ^ lsb
            if is_valid_mask(new_mask):
                code_mask = new_mask
                changed = True
                break  # Restart
            m ^= lsb

    # Local search with swaps
    best_mask = code_mask
    best_size = code_mask.bit_count()
    current_mask = code_mask

    def count_cover(mask):
        # Return numpy array of coverage counts
        counts = np.zeros(N, dtype=np.int32)
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            for w in balls[idx]:
                counts[w] += 1
            m ^= lsb
        return counts

    current_counts = count_cover(current_mask)

    def try_remove(mask, counts, idx):
        for w in balls[idx]:
            if counts[w] == 1:
                return False
        return True

    def do_remove(mask, counts, idx):
        mask ^= (1 << idx)
        for w in balls[idx]:
            counts[w] -= 1
        return mask

    def do_add(mask, counts, idx):
        mask |= (1 << idx)
        for w in balls[idx]:
            counts[w] += 1
        return mask

    # Simulated annealing-like local search
    temperature = 1.0
    cooling = 0.998
    no_improve = 0

    while time.time() - start_time < deadline:
        if current_mask.bit_count() < best_size:
            best_size = current_mask.bit_count()
            best_mask = current_mask
            no_improve = 0
            temperature = 1.0
        else:
            no_improve += 1

        if not current_mask:
            break

        # Try removal first
        candidates = []
        m = current_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            m ^= lsb

        random.shuffle(candidates)
        removed = False
        for idx in candidates:
            if try_remove(current_mask, current_counts, idx):
                current_mask = do_remove(current_mask, current_counts, idx)
                removed = True
                break

        if removed:
            continue

        # Try swap: remove one, add one
        # Pick a random codeword to remove
        idx_remove = random.choice(candidates)

        # Find a good replacement
        # We want to add a word that covers the "uncovered" parts after removal
        # Simulate removal temporarily
        temp_counts = current_counts.copy()
        for w in balls[idx_remove]:
            temp_counts[w] -= 1

        # Find words that would restore coverage
        # A word idx_add is valid if for all w in balls[idx_remove] where temp_counts[w] == 0,
        # w is in balls[idx_add]
        # And ideally idx_add covers many words with low count
        best_add = -1
        best_score = -1

        # Sample some candidates for addition
        sample_size = min(50, N)
        for _ in range(sample_size):
            idx_add = random.randint(0, N - 1)
            if idx_add in current_mask:  # Check bit
                if (current_mask >> idx_add) & 1:
                    continue

            # Check if swap is valid
            valid = True
            for w in balls[idx_remove]:
                if temp_counts[w] == 0:
                    # This word is only covered by idx_remove, so idx_add must cover it
                    if (ball_masks[idx_add] >> w) & 1 == 0:
                        valid = False
                        break

            if not valid:
                continue

            # Score: how many words with low count does idx_add cover?
            score = 0
            for w in balls[idx_add]:
                if temp_counts[w] <= 2:
                    score += (3 - temp_counts[w])
            
            if score > best_score:
                best_score = score
                best_add = idx_add

        if best_add != -1:
            # Perform swap
            current_mask = do_remove(current_mask, current_counts, idx_remove)
            current_mask = do_add(current_mask, current_counts, best_add)

            # After swap, try to remove redundant words again
            for idx in list(candidates):
                if (current_mask >> idx) & 1:
                    if try_remove(current_mask, current_counts, idx):
                        current_mask = do_remove(current_mask, current_counts, idx)
                        break
        else:
            temperature *= cooling

        # Occasionally try a 2-swap (remove 2, add 1) to escape local minima
        if no_improve > 100 and random.random() < 0.1:
            if len(candidates) >= 2:
                idx_rem1, idx_rem2 = random.sample(candidates, 2)
                # Check if removing both is possible (shouldn't be, but check)
                temp_counts = current_counts.copy()
                for w in balls[idx_rem1]:
                    temp_counts[w] -= 1
                for w in balls[idx_rem2]:
                    temp_counts[w] -= 1

                # Find a word that covers all words with count 0 after removal
                best_add = -1
                best_score = -1
                for _ in range(30):
                    idx_add = random.randint(0, N - 1)
                    if (current_mask >> idx_add) & 1:
                        continue

                    valid = True
                    for w in range(N):
                        if temp_counts[w] == 0:
                            if (ball_masks[idx_add] >> w) & 1 == 0:
                                valid = False
                                break

                    if not valid:
                        continue

                    score = 0
                    for w in balls[idx_add]:
                        if temp_counts[w] <= 2:
                            score += (3 - temp_counts[w])

                    if score > best_score:
                        best_score = score
                        best_add = idx_add

                if best_add != -1:
                    current_mask = do_remove(current_mask, current_counts, idx_rem1)
                    current_mask = do_remove(current_mask, current_counts, idx_rem2)
                    current_mask = do_add(current_mask, current_counts, best_add)

    # Final validation
    if not is_valid_mask(best_mask):
        # Fallback to initial
        return [all_words[i] for i in code_indices]

    # Convert best_mask to list of words
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        result.append(all_words[idx])
        m ^= lsb

    return result

def _greedy_cover(q, n, radius):
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
            if i in code_indices:
                continue
            inter = len(balls[i] & uncovered)
            if inter > max_covered:
                max_covered = inter
                best_word = i
        if best_word == -1:
            break
        code_indices.append(best_word)
        uncovered -= balls[best_word]

    return [all_words[i] for i in code_indices]

if __name__ == "__main__":
    code = construct(3, 6, 1)
    print(f"Code size: {len(code)}")
