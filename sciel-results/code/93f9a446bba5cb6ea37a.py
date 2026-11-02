import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    else:
        return _generic_construct(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    """
    Specialized solver for q=3, n=7, radius=1.
    Uses NumPy-based Greedy Set Cover + Iterative Reduction.
    """
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # 1. Precomputation
    # Generate all words and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute ball members as a list of lists of indices
    # For q=3, n=7, r=1, each ball has 1 + 7*2 = 15 points.
    ball_members = []
    for i in range(N):
        w = all_words[i]
        members = [i]
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val != orig_val:
                    new_word = list(w)
                    new_word[pos] = val
                    members.append(word_to_idx[tuple(new_word)])
        ball_members.append(members)
        
    # Precompute coverage matrix: coverage[point][word] = 1 if word covers point
    # This is a N x N sparse matrix, but we can use a list of sets or just the ball_members.
    # For greedy, we need to know which words cover which uncovered points.
    # It's faster to maintain a count of how many times each point is covered by the current set.
    
    # 2. Greedy Initialization
    # We want to select a subset of words such that every point is covered.
    # Greedy strategy: At each step, pick the word that covers the most *uncovered* points.
    
    uncovered = np.ones(N, dtype=bool)
    chosen_indices = []
    
    # To speed up greedy, we can precompute for each word the list of points it covers.
    # We maintain a count of coverage for each point.
    # However, calculating "newly covered" points for every candidate word at every step is O(N^2 * ball_size).
    # N=2187, ball_size=15. 2187 * 2187 * 15 ~ 71M operations. This might be too slow in pure Python loops.
    
    # Alternative: Use a heap? No, counts change.
    # Alternative: Randomized Greedy?
    # Let's try a simpler heuristic first: Randomized greedy with a limited scan.
    # Or, since N is small, we can use a bitset representation with Python integers?
    # 2187 bits fits in an integer.
    
    # Let's use Python integers for bitsets.
    # point_mask[p] = 1 << p
    # word_mask[w] = sum(1 << p for p in ball_members[w])
    
    word_masks = [0] * N
    for w in range(N):
        mask = 0
        for p in ball_members[w]:
            mask |= (1 << p)
        word_masks[w] = mask
        
    all_points_mask = (1 << N) - 1
    uncovered_mask = all_points_mask
    is_chosen = [False] * N
    
    # Greedy loop
    # To avoid O(N) scan every time, we can just scan all N words.
    # 180 steps * 2187 words * (bitwise AND + bit_count)
    # bit_count is fast (C implementation in Python 3.10+).
    # 180 * 2187 = ~393,060 iterations. Each does an AND and bit_count on ~274-byte ints.
    # This should be fast enough (sub-second).
    
    while uncovered_mask:
        if time.time() > deadline - 30:
            break
            
        best_i = -1
        best_count = 0
        
        # Scan all words
        for i in range(N):
            if is_chosen[i]:
                continue
            # Count how many uncovered points this word covers
            new_covered = word_masks[i] & uncovered_mask
            if new_covered:
                cnt = new_covered.bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_i = i
                    if cnt == uncovered_mask.bit_count():
                        break # Can't do better than covering all remaining
        
        if best_i == -1:
            # Should not happen if code is valid, but if time runs out in greedy
            # we might have a problem.
            # Fallback: add random words
            random.seed(42)
            while uncovered_mask:
                i = random.randint(0, N - 1)
                if not is_chosen[i]:
                    chosen_indices.append(i)
                    is_chosen[i] = True
                uncovered_mask &= ~word_masks[i]
                if time.time() > deadline - 5:
                    break
            break
            
        chosen_indices.append(best_i)
        is_chosen[best_i] = True
        uncovered_mask &= ~word_masks[best_i]
        
    # Ensure validity if greedy didn't finish
    if uncovered_mask:
        random.seed(123)
        while uncovered_mask:
            i = random.randint(0, N - 1)
            if not is_chosen[i]:
                chosen_indices.append(i)
                is_chosen[i] = True
            uncovered_mask &= ~word_masks[i]
            if time.time() > deadline - 2:
                break

    # 3. Local Search: Iterative Reduction
    # We have a valid code. Now try to remove redundant codewords.
    # A codeword is redundant if all points in its ball are covered by at least one other codeword.
    
    # Maintain coverage counts
    cover_count = np.zeros(N, dtype=np.int32)
    for idx in chosen_indices:
        for p in ball_members[idx]:
            cover_count[p] += 1
            
    # Iteratively scan and remove redundant words
    # To make this efficient, we can check redundancy in O(ball_size) per word.
    # Total O(N * len(chosen) * ball_size) if we do a full scan each time.
    # But we can be smarter: only check words that might have become redundant.
    # Actually, a simple loop that removes any redundant word found is often sufficient for a good result.
    
    # Let's do multiple passes of reduction.
    # Pass 1: Remove any word that is currently redundant.
    
    def try_reduce():
        removed = False
        # Iterate backwards to safely remove from list
        for i in range(len(chosen_indices) - 1, -1, -1):
            cw = chosen_indices[i]
            # Check if all points in ball(cw) have count > 1
            redundant = True
            for p in ball_members[cw]:
                if cover_count[p] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for p in ball_members[cw]:
                    cover_count[p] -= 1
                chosen_indices.pop(i)
                removed = True
        return removed

    # Run reduction until no more can be removed
    # This is fast because each check is small.
    while time.time() < deadline - 5:
        if not try_reduce():
            break

    # 4. Optional: Simple Swap Search
    # If we have time, try to swap a word with another to allow further reduction.
    # This is the "iterated local search" part, but kept very simple to avoid timeout.
    
    if time.time() < deadline - 10:
        random.seed(456)
        # Try a few random swaps
        for _ in range(2000): # Limit iterations
            if time.time() > deadline - 5:
                break
            if not chosen_indices:
                break
            
            i = random.randint(0, len(chosen_indices) - 1)
            old_cw = chosen_indices[i]
            
            # Find critical points (covered only by old_cw)
            critical = [p for p in ball_members[old_cw] if cover_count[p] == 1]
            
            if not critical:
                # Should have been removed
                continue
                
            # Find candidates that cover all critical points
            # Start with coverers of first critical point
            # We need a precomputed list of which words cover which point.
            # Let's build this on the fly or precompute?
            # Precomputing `coverers` (list of lists) is O(N * ball_size).
            
            # For simplicity and speed, let's just pick a random word from the whole space
            # that covers the first critical point, and check if it covers the rest.
            # This is a heuristic.
            
            # Get all words that cover critical[0]
            # We didn't precompute coverers. Let's just iterate through all words? No, too slow.
            # Let's skip complex swaps to ensure safety. The reduction alone usually gets close to optimal.
            
            # Instead, let's do a "kick": randomly replace a word with a random other word,
            # if the code remains valid? No, that breaks validity.
            
            # Better: Try to replace a word with another word that covers its critical points.
            # Since we don't have coverers, let's just skip swaps to be safe.
            # The greedy + reduction is a solid baseline.
            
            pass # Skipping swaps to prevent timeout risk

    # Final Result
    # Convert indices to words
    result = [list(all_words[idx]) for idx in chosen_indices]
    
    # Ensure distinctness
    seen = set()
    final_result = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            final_result.append(w)
            
    return final_result

def _generic_construct(q, n, radius, deadline):
    """
    Generic greedy + reduction for arbitrary q, n, radius.
    """
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls
    ball_members = []
    for i in range(total_words):
        w = all_words[i]
        members = set()
        # Enumerate all words within Hamming distance `radius`
        for dist in range(radius + 1):
            for positions in itertools.combinations(range(n), dist):
                for deltas in itertools.product(range(1, q), repeat=dist):
                    new_word = list(w)
                    for pos, d in zip(positions, deltas):
                        new_word[pos] = (w[pos] + d) % q
                    members.add(word_to_idx[tuple(new_word)])
        ball_members.append(list(members))
        
    # Greedy set cover
    uncovered_set = set(range(total_words))
    chosen = []
    
    while uncovered_set and time.time() < deadline - 30:
        best_ball = -1
        best_count = -1
        for i in range(total_words):
            count = sum(1 for m in ball_members[i] if m in uncovered_set)
            if count > best_count:
                best_count = count
                best_ball = i
        if best_ball == -1 or best_count <= 0:
            break
        chosen.append(best_ball)
        for m in ball_members[best_ball]:
            if m in uncovered_set:
                uncovered_set.discard(m)
        if not uncovered_set:
            break
    
    # Reduction
    coverage = [[] for _ in range(total_words)]
    for ci, ball_idx in enumerate(chosen):
        for m in ball_members[ball_idx]:
            coverage[m].append(ci)
    
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for ci in range(len(chosen) - 1, -1, -1):
            redundant = True
            for m in ball_members[chosen[ci]]:
                if len(coverage[m]) == 1 and coverage[m][0] == ci:
                    redundant = False
                    break
            if redundant:
                for m in ball_members[chosen[ci]]:
                    coverage[m].remove(ci)
                chosen.pop(ci)
                changed = True
    
    result = [list(all_words[idx]) for idx in chosen]
    seen = set()
    final_result = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            final_result.append(w)
    return final_result
