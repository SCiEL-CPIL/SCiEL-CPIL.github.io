import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 100.0  # Safety margin against 120s limit

    # 1. Construct the baseline set of codewords
    # Direct sum of ternary tetracode (len 4, rad 1) and full space (len n-4)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    if n < 4:
        # Fallback for very small n, though task specifies n=8
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    base_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            base_codewords.append(h + list(tail))
    
    # Convert to tuples for hashing and efficient set operations
    initial_set = set(base_codewords)
    
    # If we are already at a very small number, just return
    if len(initial_set) <= 500:
        return [list(c) for c in initial_set]

    # Pre-compute neighborhoods for all possible words? No, too much memory/time for large q^n.
    # Instead, compute neighborhood on the fly or cache.
    # For q=3, n=8, total space is 6561. We can afford to pre-compute neighborhoods for all words in the space.
    
    all_words = list(itertools.product(range(q), repeat=n))
    # Map word to index for fast counting
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Pre-compute list of indices in the neighborhood of radius 'radius' for each word in the space
    # Since radius is 1, we just need single-position changes.
    # neigh[idx] = list of indices in the space at distance <= radius from all_words[idx]
    neigh = [[] for _ in range(len(all_words))]
    
    # Optimization: Generate neighbors for each word
    # For radius 1, neighbors are the word itself + words with 1 position changed.
    for i, w in enumerate(all_words):
        neigh[i].append(i)
        for pos in range(n):
            for val in range(q):
                if w[pos] == val:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_idx = word_to_idx[tuple(new_w)]
                neigh[i].append(new_idx)
        
    # Initialize cover counts
    cover_count = [0] * len(all_words)
    
    # Add all initial codewords
    current_codewords = list(initial_set)
    
    for c in current_codewords:
        idx = word_to_idx[c]
        for n_idx in neigh[idx]:
            cover_count[n_idx] += 1
            
    # Greedy Removal Loop
    # We try to remove codewords one by one.
    # To be efficient, we iterate through the current list.
    # If removal is valid, we remove it. We might need multiple passes if removing one codeword
    # "unlocks" the removal of another (though in covering codes, removal is generally monotonic 
    # in the sense that if a set is a cover, removing a word makes it less likely to be a cover, 
    # but sometimes redundant words exist in clusters).
    # Actually, if we remove a word, the counts decrease. If a word was removable before, it might 
    # become non-removable after removing another word? No.
    # If S is a cover, and we remove c1 to get S1, S1 is a cover.
    # If c2 was removable in S, is it removable in S1?
    # Cover(S1) = Cover(S) \ N(c1).
    # If removing c2 from S leaves all counts >= 1, let min_count_in_S be the min count.
    # If min_count_in_S >= 2 for all points in N(c2), then c2 is removable.
    # If we remove c1 first, counts in N(c1) decrease by 1.
    # If N(c2) intersects N(c1), counts in that intersection decreased.
    # So c2 might NOT be removable in S1 even if it was in S.
    # Therefore, we must re-evaluate.
    
    # Strategy: 
    # 1. Check all codewords. If removable, mark for removal.
    # 2. Apply removals.
    # 3. Repeat until no more removals or time limit.
    
    # To speed up, we can process in batches or just one by one.
    # One by one is safer for correctness of the "removable" check.
    
    changed = True
    while changed:
        changed = False
        # Shuffle to avoid bias? Or just iterate. Iterating is fine.
        # We iterate over a copy of the list because we might remove items.
        # However, removing an item changes cover_count, which affects the removability of subsequent items.
        # So we must process sequentially and update counts immediately.
        
        i = 0
        while i < len(current_codewords):
            if time.time() - start_time > time_limit - 5:
                break
            
            c = current_codewords[i]
            idx = word_to_idx[c]
            
            # Check if we can remove c
            # We need to see if any neighbor count drops to 0
            # We simulate removal
            removable = True
            for n_idx in neigh[idx]:
                if cover_count[n_idx] == 1:
                    removable = False
                    break
            
            if removable:
                # Commit removal
                for n_idx in neigh[idx]:
                    cover_count[n_idx] -= 1
                current_codewords.pop(i)
                changed = True
                # Do not increment i, check the next element at the same index
            else:
                i += 1
        
        if time.time() - start_time > time_limit - 5:
            break

    # If time remains, we can try a different order or local search?
    # For now, the sequential greedy is a solid baseline improvement.
    
    return [list(c) for c in current_codewords]

# Helper to run if executed directly for testing
if __name__ == "__main__":
    # Test for q=3, n=8, radius=1
    code = construct(3, 8, 1)
    print(f"Size: {len(code)}")
    
    # Verify coverage
    import itertools
    q, n, radius = 3, 8, 1
    all_words = list(itertools.product(range(q), repeat=n))
    covered = set()
    for c in code:
        c_tuple = tuple(c)
        # Add all neighbors
        for i, w in enumerate(all_words):
            if sum(1 for a, b in zip(c_tuple, w) if a != b) <= radius:
                covered.add(w)
    if len(covered) == len(all_words):
        print("Valid cover")
    else:
        print(f"Invalid cover: {len(all_words) - len(covered)} missing")
