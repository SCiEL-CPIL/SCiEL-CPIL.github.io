"""Optimized ternary covering code K_3(7,1) via randomized greedy set-cover with bitsets."""
import time
import random
import itertools

# Deadline guard: keep a safety margin before the 150s kill.
_DEADLINE = 150.0
_SAFETY = 10.0  # seconds


def _ball_indices(q, n, radius, center):
    """Yield integer indices of all words within Hamming distance <= radius of center."""
    # Precompute: for a given center (tuple of ints), generate all words at distance <= radius.
    # For efficiency, we precompute the list of neighbor offsets (differences) once per radius.
    pass  # We'll use a different approach below.


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE - _SAFETY

    total = q ** n  # 3^7 = 2187

    # Precompute all words as tuples and their integer indices
    # We'll represent each word as an integer in [0, total)
    # Map: word (w0,...,w_{n-1}) -> index = w0 + w1*q + ... (little-endian) or big-endian
    # Let's use: index = sum(w_i * q^i) for i in 0..n-1 (little-endian)

    # Precompute all words as lists
    all_words = list(itertools.product(range(q), repeat=n))
    # Map tuple -> index
    tuple_to_idx = {w: i for i, w in enumerate(all_words)}

    # For each word, precompute the set of words within distance <= radius
    # This is a bitset of 'total' bits. total=2187, so we can use Python ints as bitsets.
    
    # Precompute neighbor bitsets for each word
    # neighbors[i] = bitset of all words j such that dist(word_i, word_j) <= radius
    # Actually, we want: for each candidate center c, the set of words it covers.
    # Let's precompute cover_set[i] = bitset of words covered by center i.
    
    # To compute cover_set efficiently:
    # For each word w (as index), generate all words at distance <= radius.
    
    # Precompute: for each position and each symbol change, the index difference
    # Actually, let's just precompute the list of words at each distance for each center.
    # Since total is only 2187, and radius=1, each center covers 1 + n*(q-1) = 1 + 7*2 = 15 words.
    
    # Build cover_sets as list of Python ints (bitsets)
    cover_sets = [0] * total
    
    # Precompute powers of q for index computation
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
    
    def tuple_to_int(w):
        idx = 0
        for i in range(n):
            idx += w[i] * q_powers[i]
        return idx
    
    # Precompute all word tuples to int
    word_to_int = {}
    for i, w in enumerate(all_words):
        word_to_int[w] = i
    
    # For each center, compute its cover set
    for center_idx in range(total):
        center = all_words[center_idx]
        bits = 1 << center_idx  # The center itself
        
        # Words at distance 1: change one position to a different symbol
        for pos in range(n):
            for sym in range(q):
                if sym == center[pos]:
                    continue
                # Create new word
                new_word = list(center)
                new_word[pos] = sym
                new_idx = word_to_int[tuple(new_word)]
                bits |= (1 << new_idx)
        
        if radius >= 2:
            # For radius 2, also add distance 2 words
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for sym1 in range(q):
                        if sym1 == center[pos1]:
                            continue
                        for sym2 in range(q):
                            if sym2 == center[pos2]:
                                continue
                            new_word = list(center)
                            new_word[pos1] = sym1
                            new_word[pos2] = sym2
                            new_idx = word_to_int[tuple(new_word)]
                            bits |= (1 << new_idx)
        
        cover_sets[center_idx] = bits
    
    # Now we have cover_sets[i] = bitset of words covered by center i.
    
    # Greedy set cover with randomization
    # We'll do multiple restarts with different random orderings
    
    best_solution = None
    best_size = total + 1
    
    # Number of restarts: we'll do as many as we can before deadline
    # Each restart: greedy with random tie-breaking
    
    # For speed, we'll use a fixed number of restarts but break if time runs out
    max_restarts = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random permutation of centers
        centers_order = list(range(total))
        random.shuffle(centers_order)
        
        # Greedy: at each step, pick the center that covers the most uncovered words
        # Track uncovered words as a bitset
        uncovered = (1 << total) - 1  # All words initially uncovered
        selected = []
        
        # For efficiency, we'll maintain a list of (coverage_count, center_idx) 
        # but recomputing each step is O(total) which is 2187 * ~200 steps = ~437k ops per restart
        # That's too slow for 5000 restarts. Let's optimize.
        
        # Alternative: precompute for each center, the list of words it covers (as indices)
        # Then use a counting approach.
        
        # Actually, let's use a simpler approach: 
        # Maintain uncovered as a set of indices (or bitset)
        # At each step, for each candidate center not yet selected, compute |cover_set[c] & uncovered|
        # Pick the one with max coverage.
        
        # To speed this up, we can use the fact that total is small (2187).
        # 2187 * 200 = 437,400 bitset AND operations per restart. Each is fast in Python with big ints.
        # 5000 restarts * 437,400 = 2.187 billion ops... too slow.
        
        # Let's reduce: do fewer restarts but smarter.
        # Or use a different strategy: random greedy with limited restarts.
        
        # Let's try: 200 restarts, each greedy.
        # 200 * 2187 * 200 = 87 million bitset ops. Might be okay.
        
        # Actually, let's just do it and break on time.
        
        selected_set = set()
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_center = -1
            best_count = 0
            best_cover = 0
            
            for c in centers_order:
                if c in selected_set:
                    continue
                # Count how many uncovered words this center covers
                cover = cover_sets[c] & uncovered
                # Count bits
                cnt = cover.bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_center = c
                    best_cover = cover
            
            if best_center == -1:
                break
            
            selected.append(best_center)
            selected_set.add(best_center)
            uncovered &= ~best_cover
        
        if not uncovered:  # Valid cover
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = selected[:]
        
        # Early stop if we've reached a very good solution
        if best_size <= 186:
            break
    
    # Convert selected indices back to word lists
    if best_solution is None:
        # Fallback: return all words (shouldn't happen)
        return [list(w) for w in all_words]
    
    # Verify validity
    covered = 0
    for c in best_solution:
        covered |= cover_sets[c]
    
    if covered != (1 << total) - 1:
        # Fallback
        return [list(w) for w in all_words]
    
    return [list(all_words[c]) for c in best_solution]
