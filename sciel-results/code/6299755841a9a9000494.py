"""Improved program for the suite task; uses Greedy Set Cover."""
# EVOLVE-BLOCK-START
import time
import heapq
import itertools

# Deadline guard: stop if we exceed this fraction of the 120s limit
# The evaluator kills at 150s, but we want a safety margin.
DEADLINE_SECONDS = 100.0

def construct(q, n, radius):
    start_time = time.time()
    
    # Parameters are fixed by the task, but we handle general case structure
    # Task: q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic for unexpected parameters to ensure validity
        # Though the task contract implies specific values
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Precompute all words and their indices
    total_words = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Map word tuple to index for fast lookup
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute the "ball" of radius 1 for each word
    # ball[i] is a list of indices of words within distance <= 1 of words[i]
    balls = [[] for _ in range(total_words)]
    
    for i, w in enumerate(words):
        balls[i].append(i) # Include itself
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                # Create neighbor
                neighbor = w[:]
                neighbor[pos] = new_val
                idx = word_to_idx[tuple(neighbor)]
                balls[i].append(idx)
        
    # State tracking
    # covered[j] = 1 if word j is covered, 0 otherwise
    covered = bytearray(total_words)
    uncovered_count = total_words
    
    # Greedy Set Cover
    # We maintain a max-heap of (gain, word_index)
    # Python heapq is a min-heap, so we push -gain
    # gain[i] = number of uncovered words in balls[i]
    
    # Initial gains: all balls have size 1 + n*(q-1) = 1 + 6*2 = 13
    initial_gain = 1 + n * (q - 1)
    
    heap = []
    for i in range(total_words):
        heapq.heappush(heap, (-initial_gain, i))
    
    selected_codewords = []
    
    while uncovered_count > 0:
        # Check time
        if time.time() - start_time > DEADLINE_SECONDS:
            break
            
        # Get the word with max gain
        neg_gain, idx = heapq.heappop(heap)
        gain = -neg_gain
        
        # If gain is 0, no more improvement possible (should not happen if uncovered > 0)
        if gain == 0:
            break
            
        # Select this word
        selected_codewords.append(words[idx][:])
        
        # Update coverage
        # Mark all neighbors as covered
        for neighbor_idx in balls[idx]:
            if not covered[neighbor_idx]:
                covered[neighbor_idx] = 1
                uncovered_count -= 1
        
        # If all covered, stop
        if uncovered_count == 0:
            break
        
        # Update heap for other words
        # A word's gain decreases if any of its neighbors were just covered
        # We don't update the heap entries immediately (lazy deletion).
        # When we pop an item, we check if its gain is still valid.
        # However, calculating the current gain requires scanning the ball.
        # To keep it efficient, we can just push updated gains for affected words?
        # Or simpler: Lazy deletion. When we pop, we re-calculate the gain.
        # If the re-calculated gain matches the popped gain, we use it.
        # If not, we push the new gain and continue.
        
        # Actually, a simpler and robust approach for small N:
        # Since N=729 is small, we can just re-calculate gains for all words in the next iteration?
        # No, that's O(N^2).
        
        # Let's stick to the standard lazy heap update:
        # When a word `idx` is selected, only words `j` such that `idx` is in `balls[j]` (i.e., dist(j, idx) <= 1) can have their gain reduced.
        # These are exactly the neighbors of `idx`.
        # So we only need to update gains for `j in balls[idx]`.
        
        # We push new entries for these neighbors.
        for j in balls[idx]:
            # Calculate current gain for j
            # gain_j = count of k in balls[j] such that covered[k] == 0
            current_gain = 0
            for k in balls[j]:
                if not covered[k]:
                    current_gain += 1
            
            # Push to heap
            if current_gain > 0:
                heapq.heappush(heap, (-current_gain, j))
        
        # Note: The old entries for these j's will remain in the heap.
        # When we pop an old entry for j, we will see that its gain is stale (too high).
        # We need a way to detect staleness.
        # We can store the gain in the heap and compare with a stored current_gain array?
        # Let's maintain a `current_gains` array.
        
    # The above lazy deletion logic needs a `current_gains` array to verify pops.
    # Let's restart the loop logic with this correction.
    
    # Reset state for correct implementation
    covered = bytearray(total_words)
    uncovered_count = total_words
    selected_codewords = []
    
    # current_gains[i] = number of uncovered words in balls[i]
    current_gains = [0] * total_words
    for i in range(total_words):
        current_gains[i] = len(balls[i]) # All uncovered initially
        
    heap = []
    for i in range(total_words):
        heapq.heappush(heap, (-current_gains[i], i))
        
    while uncovered_count > 0:
        if time.time() - start_time > DEADLINE_SECONDS:
            break
            
        neg_gain, idx = heapq.heappop(heap)
        gain = -neg_gain
        
        # Check if this entry is stale
        if gain != current_gains[idx]:
            continue
            
        if gain == 0:
            break
            
        # Select idx
        selected_codewords.append(words[idx][:])
        
        # Update coverage
        newly_covered = []
        for neighbor_idx in balls[idx]:
            if not covered[neighbor_idx]:
                covered[neighbor_idx] = 1
                uncovered_count -= 1
                newly_covered.append(neighbor_idx)
                
        if uncovered_count == 0:
            break
            
        # Update gains for words affected by newly_covered
        # A word j's gain decreases if any of its neighbors (k in balls[j]) is in newly_covered.
        # This means dist(j, k) <= 1.
        # So for each k in newly_covered, we need to update gains for all j in balls[k].
        
        # To avoid duplicates in updates, we can just iterate over newly_covered and update balls[k].
        # But a word j might be in balls[k1] and balls[k2].
        # We just push updated gains. Stale check handles duplicates.
        
        for k in newly_covered:
            # For every word j within distance 1 of k, its gain might have dropped
            for j in balls[k]:
                # Recalculate gain for j
                # Optimization: instead of full scan, we know gain dropped by 1 for each newly covered neighbor.
                # But it's safer to just rescan if the number of newly covered is small?
                # Actually, we can just decrement if we know k was the one covered.
                # But `current_gains[j]` counts ALL uncovered neighbors.
                # If k was uncovered and is now covered, and k is in balls[j], then gain[j] decreases by 1.
                
                # We need to be careful: did k contribute to gain[j]?
                # Yes, if k was uncovered before this step.
                # We know k was uncovered because it's in `newly_covered`.
                # So we can simply do:
                current_gains[j] -= 1
                
                # Push new gain to heap
                if current_gains[j] > 0:
                    heapq.heappush(heap, (-current_gains[j], j))
                    
    return selected_codewords
# EVOLVE-BLOCK-END
