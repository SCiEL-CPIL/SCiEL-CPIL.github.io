import time
import random
import itertools

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Build the extended quadratic-residue code of length 18
    # QR(17) is a [17, 6, 5] code. Its extension is [18, 6, 6] with 64 words.
    row0_positions = [0, 1, 3, 7, 8, 12, 14, 15]
    
    G = []
    for i in range(6):
        row = 0
        for pos in row0_positions:
            shifted = (pos + i) % 17
            row |= (1 << shifted)
        G.append(row)
    
    qr_words_18 = []
    for mask in range(64):
        word = 0
        for i in range(6):
            if (mask >> i) & 1:
                word ^= G[i]
        parity = word.bit_count() % 2
        extended = word | (parity << 17)
        qr_words_18.append(extended)
    
    best_words = list(qr_words_18)
    best_size = len(best_words)
    
    # Bitset-based Bron-Kerbosch for maximum independent set
    # In the Hamming graph H(18,6), we want a maximum set of vertices
    # where no two are adjacent (distance < 6).
    # Equivalently, find maximum independent set in the graph where
    # edges connect pairs with distance < 6.
    
    # Precompute for each vertex, the set of vertices at distance < MIN_DIST
    # Using bitsets for efficiency
    
    # Strategy: Use the symmetry of the hypercube.
    # Fix vertex 0 in the code (by translation invariance of Hamming distance).
    # Then search for maximum independent set in the remaining vertices
    # that are all at distance >= MIN_DIST from 0 and from each other.
    
    # For a vertex v, define N(v) = {u : dist(v,u) < MIN_DIST}
    # We need an independent set in the complement graph.
    
    # Use a greedy + local search hybrid with bitset acceleration
    
    # Precompute distance classes
    # For efficiency, use a different approach:
    # Start with the QR code and try to extend it using simulated annealing
    
    current_words = list(best_words)
    current_set = set(current_words)
    
    def can_add_fast(candidate, words_list):
        for w in words_list:
            if (candidate ^ w).bit_count() < MIN_DIST:
                return False
        return True
    
    # Phase 1: Greedy extension
    added_any = True
    while added_any and time.time() < deadline - 15.0:
        added_any = False
        for candidate in range(1 << N):
            if candidate in current_set:
                continue
            if can_add_fast(candidate, current_words):
                current_words.append(candidate)
                current_set.add(candidate)
                added_any = True
                if len(current_words) > best_size:
                    best_words = list(current_words)
                    best_size = len(current_words)
            if time.time() > deadline - 10.0:
                break
    
    # Phase 2: Bitset-based local search
    # Represent the set of selected vertices as a bitset
    # For each candidate, check if it's compatible with all selected
    
    if time.time() < deadline - 10.0:
        # Try to improve by swapping: remove one vertex, add another
        rng = random.Random(42)
        
        # Precompute a list of all vertices sorted by weight
        all_vertices = list(range(1 << N))
        all_vertices.sort(key=lambda x: (x.bit_count(), x))
        
        # Try to add vertices that are close to existing ones (bit flips)
        flip_candidates = set()
        for w in current_words:
            for r in range(1, 4):  # Hamming radius 1-3
                for bits in itertools.combinations(range(N), r):
                    flipped = w
                    for b in bits:
                        flipped ^= (1 << b)
                    flip_candidates.add(flipped)
        
        candidate_list = list(flip_candidates)
        rng.shuffle(candidate_list)
        
        for candidate in candidate_list:
            if candidate in current_set:
                continue
            if can_add_fast(candidate, current_words):
                current_words.append(candidate)
                current_set.add(candidate)
                if len(current_words) > best_size:
                    best_words = list(current_words)
                    best_size = len(current_words)
            if time.time() > deadline - 5.0:
                break
    
    # Phase 3: Multiple random restarts with greedy
    if time.time() < deadline - 15.0:
        rng2 = random.Random(12345)
        
        for restart in range(10):
            if time.time() > deadline - 8.0:
                break
            
            # Build a random seed code
            seed_words = []
            seed_set = set()
            attempts = 0
            while len(seed_words) < 15 and attempts < 100000:
                w = rng2.randint(0, (1 << N) - 1)
                if w not in seed_set:
                    valid = True
                    for sw in seed_words:
                        if (w ^ sw).bit_count() < MIN_DIST:
                            valid = False
                            break
                    if valid:
                        seed_words.append(w)
                        seed_set.add(w)
                attempts += 1
            
            if len(seed_words) < 8:
                continue
            
            # Greedy extend from this seed
            extended = list(seed_words)
            ext_set = set(extended)
            
            # Sort candidates by weight for better pruning
            for candidate in range(1 << N):
                if candidate in ext_set:
                    continue
                if can_add_fast(candidate, extended):
                    extended.append(candidate)
                    ext_set.add(candidate)
                if time.time() > deadline - 5.0:
                    break
            
            if len(extended) > best_size:
                best_words = extended
                best_size = len(extended)
    
    return best_words
