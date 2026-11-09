import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    N = 17
    MIN_D = 6
    NUM_WORDS = 1 << N  # 131072
    
    # Precompute Hamming weights using numpy
    words = np.arange(NUM_WORDS, dtype=np.uint32)
    # Compute weights via popcount
    weights = np.zeros(NUM_WORDS, dtype=np.uint8)
    for bit in range(N):
        weights |= ((words >> bit) & 1).astype(np.uint8)
    
    # Precompute forbidden zones: for each word w, which words are within distance < MIN_D
    # This is expensive to precompute fully, so we'll use a different approach:
    # For each word, we can quickly check if it's within distance < MIN_D of any codeword
    # by using a bitset approach with numpy
    
    # Strategy: Use a bitset representation of the code
    # code_bitset[i] = True if word i is in the code
    # For checking if a new word w can be added, we need to verify that
    # no codeword c has Hamming distance < MIN_D from w.
    
    # Precompute for each word w, the list of words within distance < MIN_D
    # This is too large to precompute fully (131072 * ~20000 entries), so we'll use
    # a different approach: for each candidate w, check against all codewords using numpy
    
    # Actually, let's use a smarter approach:
    # 1. Generate the 256-word linear code as baseline
    # 2. Use a "forbidden set" approach: maintain a set of all words that are within
    #    distance < MIN_D of any codeword. A new word can be added iff it's not in this set.
    
    # Precompute the "ball" of radius MIN_D-1 around each word? That's still large.
    # Instead, let's use numpy to vectorize the distance check.
    
    def generate_linear_code():
        """Generate the [17,8,6] linear code with 256 words."""
        # We need to find 8 linearly independent vectors of weight >= 6
        # such that all their sums also have weight >= 6.
        # Use a known construction or greedy approach.
        
        basis = []
        current_code = [0]
        
        # Try all vectors in increasing order of weight, then value
        candidates = sorted(range(1, NUM_WORDS), key=lambda x: (weights[x], x))
        
        for g in candidates:
            if len(basis) >= 8:
                break
            valid = True
            for c in current_code:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                basis.append(g)
                new_code = []
                for c in current_code:
                    new_code.append(c)
                    new_code.append(c ^ g)
                current_code = new_code
        
        return current_code
    
    # Generate baseline linear code
    best_code = generate_linear_code()
    best_size = len(best_code)
    
    # Now try to improve using a "forbidden zone" greedy approach
    # Maintain a boolean array: forbidden[w] = True if w is within distance < MIN_D of some codeword
    
    def build_forbidden_from_code(code):
        """Build forbidden array from a list of codewords."""
        forbidden = np.zeros(NUM_WORDS, dtype=bool)
        code_arr = np.array(code, dtype=np.uint32)
        
        # For each codeword, mark all words within distance < MIN_D as forbidden
        # This is O(|code| * NUM_WORDS) which is 256 * 131072 = 33M operations
        # We can vectorize this with numpy
        
        for c in code_arr:
            # Compute distance from c to all words
            diffs = words ^ c
            dists = np.zeros(NUM_WORDS, dtype=np.uint8)
            for bit in range(N):
                dists |= ((diffs >> bit) & 1).astype(np.uint8)
            forbidden |= (dists < MIN_D)
        
        return forbidden
    
    def greedy_pass_with_forbidden(forbidden, order):
        """Greedy pass using precomputed forbidden array. Returns list of codewords."""
        code = []
        for w in order:
            if not forbidden[w]:
                code.append(w)
                # Mark forbidden zone for w
                diffs = words ^ w
                dists = np.zeros(NUM_WORDS, dtype=np.uint8)
                for bit in range(N):
                    dists |= ((diffs >> bit) & 1).astype(np.uint8)
                forbidden |= (dists < MIN_D)
        return code
    
    # Build initial forbidden from linear code
    forbidden = build_forbidden_from_code(best_code)
    
    # Create orders to try
    all_words_list = list(range(NUM_WORDS))
    
    # Weight-sorted orders
    order_asc = sorted(all_words_list, key=lambda x: weights[x])
    order_desc = sorted(all_words_list, key=lambda x: -weights[x])
    
    # Try structured orders first
    for order in [order_asc, order_desc]:
        if time.time() > deadline - 5:
            break
        # We need to reset forbidden for each pass
        # But rebuilding forbidden from scratch each time is expensive
        # Instead, let's do a fresh greedy from scratch for each ordering
        
        # Actually, let's just do random greedy passes with the forbidden approach
        # but rebuild forbidden each time from the current best code
        pass
    
    # Better approach: for each random ordering, start from scratch
    # but use the forbidden array to speed up checks
    
    # Precompute the "ball" masks more efficiently
    # For a given word w, the set of words within distance < MIN_D can be computed
    # by enumerating all subsets of positions to flip
    
    def get_forbidden_mask(w):
        """Return a boolean array where True = within distance < MIN_D of w."""
        mask = np.zeros(NUM_WORDS, dtype=bool)
        # Enumerate all words within distance < MIN_D of w
        # This is sum_{i=0}^{MIN_D-1} C(17, i) = C(17,0)+C(17,1)+...+C(17,5)
        # = 1 + 17 + 136 + 680 + 2380 + 6188 = 9402
        # For each such word, set mask[word] = True
        
        # Use itertools to generate combinations
        import itertools
        positions = list(range(N))
        for r in range(MIN_D):
            for combo in itertools.combinations(positions, r):
                flipped = w
                for pos in combo:
                    flipped ^= (1 << pos)
                mask[flipped] = True
        return mask
    
    # This is still slow for each word. Let's precompute for all words?
    # 131072 * 9402 = 1.2 billion operations - too slow.
    
    # Alternative: use the linear code as a starting point and try to add words
    # one by one, checking against the existing code
    
    def can_add_to_code(w, code_arr, weights_arr):
        """Check if w can be added to code_arr."""
        diffs = code_arr ^ w
        dists = np.zeros(len(code_arr), dtype=np.uint8)
        for bit in range(N):
            dists |= ((diffs >> bit) & 1).astype(np.uint8)
        return np.all(dists >= MIN_D)
    
    # Let's try a different approach: start with the linear code, then try to add
    # additional words that are far from all existing codewords
    
    code_arr = np.array(best_code, dtype=np.uint32)
    
    # Find all words that are at distance >= MIN_D from all codewords in the linear code
    # These are "available" words
    
    # For each word, check if it's compatible with the entire linear code
    # This is O(NUM_WORDS * 256) = 33M checks, each involving 17 bit operations
    # With numpy, we can vectorize over the 256 codewords
    
    # Compute distance from each word to each codeword
    # dists[i, j] = Hamming distance between words[i] and code_arr[j]
    # This is a 131072 x 256 matrix of uint8 - about 33 MB, manageable
    
    # But computing it naively is slow. Let's use a different approach:
    # For each word w, compute the minimum distance to any codeword
    # min_dist[w] = min_j dist(w, code_arr[j])
    # w is available if min_dist[w] >= MIN_D
    
    # We can compute this in batches
    batch_size = 10000
    available_words = []
    
    for start in range(0, NUM_WORDS, batch_size):
        if time.time() > deadline - 5:
            break
        end = min(start + batch_size, NUM_WORDS)
        batch = words[start:end]
        
        # For each word in batch, compute min distance to code_arr
        # dists has shape (batch_size, len(code_arr))
        diffs = batch[:, None] ^ code_arr[None, :]  # (batch_size, 256)
        
        dists = np.zeros((len(batch), len(code_arr)), dtype=np.uint8)
        for bit in range(N):
            dists |= ((diffs >> bit) & 1).astype(np.uint8)
        
        min_dists = np.min(dists, axis=1)
        avail_mask = min_dists >= MIN_D
        available_words.extend(batch[avail_mask].tolist())
    
    # Now try to greedily add words from available_words to the code
    # We need to check compatibility with both the original linear code AND
    # any newly added words
    
    # Start with the linear code
    current_code = list(best_code)
    current_code_arr = np.array(current_code, dtype=np.uint32)
    
    # Shuffle available words and try to add them
    random.shuffle(available_words)
    
    for w in available_words:
        if time.time() > deadline - 5:
            break
        # Check if w is compatible with all current codewords
        if can_add_to_code(w, current_code_arr, weights):
            current_code.append(w)
            current_code_arr = np.append(current_code_arr, w)
            if len(current_code) > best_size:
                best_code = current_code
                best_size = len(current_code)
    
    # Try multiple random restarts with different orderings
    while time.time() < deadline - 3:
        random.shuffle(available_words)
        current_code = list(best_code)
        current_code_arr = np.array(current_code, dtype=np.uint32)
        
        for w in available_words:
            if time.time() > deadline - 3:
                break
            if can_add_to_code(w, current_code_arr, weights):
                current_code.append(w)
                current_code_arr = np.append(current_code_arr, w)
        
        if len(current_code) > best_size:
            best_code = current_code
            best_size = len(current_code)
    
    return best_code
