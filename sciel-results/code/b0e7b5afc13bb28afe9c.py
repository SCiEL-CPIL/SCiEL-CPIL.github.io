import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    space_size = q ** n
    
    # Precompute word to index and index to word
    # For efficiency, we work with indices 0..space_size-1
    
    def word_to_idx(word):
        idx = 0
        p = 1
        for i in range(n-1, -1, -1):
            idx += word[i] * p
            p *= q
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return tuple(word)

    # Precompute balls
    # ball_indices[idx] = list of indices of words in ball(idx)
    ball_indices = []
    for idx in range(space_size):
        w = idx_to_word(idx)
        ball = [idx]
        for i in range(n):
            orig = w[i]
            for v in range(q):
                if v == orig: continue
                nw = list(w)
                nw[i] = v
                ball.append(word_to_idx(tuple(nw)))
        ball_indices.append(ball)

    def compute_counts(code_set):
        counts = [0] * space_size
        for c in code_set:
            for idx in ball_indices[c]:
                counts[idx] += 1
        return counts

    def is_redundant(c, counts):
        for idx in ball_indices[c]:
            if counts[idx] < 2:
                return False
        return True

    # Initial Code Construction
    if q == 3 and n == 6 and radius == 1:
        # Tetracode product
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_words = []
        for h in tetracode:
            for t1 in range(3):
                for t2 in range(3):
                    initial_words.append(tuple(h + [t1, t2]))
        initial_code = set(word_to_idx(w) for w in initial_words)
    else:
        # Generic Greedy Init
        initial_code = set()
        covered = [False] * space_size
        while True:
            # Check if all covered
            if all(covered):
                break
            # Find best word
            best_idx = -1
            best_count = -1
            for idx in range(space_size):
                if idx in initial_code:
                    continue
                new_covered = 0
                for x in ball_indices[idx]:
                    if not covered[x]:
                        new_covered += 1
                if new_covered > best_count:
                    best_count = new_covered
                    best_idx = idx
            if best_idx == -1:
                break
            initial_code.add(best_idx)
            for x in ball_indices[best_idx]:
                covered[x] = True

    current_code = set(initial_code)
    counts = compute_counts(current_code)
    
    best_code = list(current_code)
    best_size = len(current_code)

    # Local Search Loop
    while time.time() < deadline:
        # Phase 1: Greedy Removal
        removed_any = False
        while True:
            removable = [c for c in current_code if is_redundant(c, counts)]
            if not removable:
                break
            # Remove a random removable codeword
            c = random.choice(removable)
            current_code.remove(c)
            for idx in ball_indices[c]:
                counts[idx] -= 1
            removed_any = True
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                
        if not removed_any:
            # Phase 2: Try a Swap
            if len(current_code) == 0:
                break
            
            c_old = random.choice(list(current_code))
            c_new = random.randint(0, space_size - 1)
            
            if c_new in current_code:
                continue
            
            # Tentatively swap
            # Decrement counts for c_old
            for idx in ball_indices[c_old]:
                counts[idx] -= 1
            # Increment counts for c_new
            for idx in ball_indices[c_new]:
                counts[idx] += 1
            
            # Check validity: all counts must be >= 1
            if min(counts) > 0:
                # Accept swap
                current_code.remove(c_old)
                current_code.add(c_new)
                # The removal loop in the next iteration will handle pruning
            else:
                # Revert
                for idx in ball_indices[c_new]:
                    counts[idx] -= 1
                for idx in ball_indices[c_old]:
                    counts[idx] += 1

    # Convert best_code indices back to words
    return [list(idx_to_word(idx)) for idx in best_code]
