import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Map ternary strings to integers
    # w = sum(d_i * 3^(n-1-i))
    
    def to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    def from_idx(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # 2. Precompute neighbors
    # neighbors[idx] is a list of indices
    neighbors = [[] for _ in range(q**n)]
    for w in range(q**n):
        word = from_idx(w)
        # Center
        neighbors[w].append(w)
        # Distance 1
        for i in range(n):
            original = word[i]
            for d in range(q):
                if d == original: continue
                word[i] = d
                neighbors[w].append(to_idx(word))
            word[i] = original
            
    # 3. Initial Code: Direct Sum
    # Tetracode for n=4: [a, b, (a+b)%3, (a+2b)%3]
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    # Extend to length n
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_codewords.append(h + list(tail))
            
    # Convert to indices
    codewords = set(to_idx(c) for c in initial_codewords)
    
    # 4. Coverage Array
    cov = [0] * (q**n)
    for c in codewords:
        for nb in neighbors[c]:
            cov[nb] += 1
            
    # Helper: Is valid?
    # min(cov) > 0
    
    # Helper: Critical words (cov == 1)
    def count_critical():
        return sum(1 for x in cov if x == 1)
    
    best_size = len(codewords)
    best_codewords = list(codewords)
    
    # Optimization Loop
    while time.time() < deadline:
        # Phase 1: Greedy Deletion
        deleted = True
        while deleted:
            deleted = False
            # Shuffle for randomness in deletion order
            cw_list = list(codewords)
            random.shuffle(cw_list)
            for c in cw_list:
                if c not in codewords: continue # might have been removed
                # Check if removable
                removable = True
                for nb in neighbors[c]:
                    if cov[nb] == 1:
                        removable = False
                        break
                if removable:
                    # Remove
                    for nb in neighbors[c]:
                        cov[nb] -= 1
                    codewords.remove(c)
                    deleted = True
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                    
        # Phase 2: Local Search (Moves)
        # Try to move a codeword to reduce critical count
        # If we reduce critical count, we might enable a deletion later
        moved = False
        # Limit iterations per phase to avoid getting stuck in one long loop
        max_moves = 1000
        for _ in range(max_moves):
            if time.time() > deadline: break
            
            if len(codewords) == 0: break
            
            c = random.choice(list(codewords))
            # Generate candidate c_prime
            # Strategy: Pick a random word, or a word that helps reduce criticals
            # Let's try random first, it's simple and effective with many iterations
            c_prime = random.randint(0, q**n - 1)
            
            if c_prime in codewords:
                continue
                
            # Check validity of move c -> c_prime
            # Words that will lose coverage: N(c) \ N(c_prime)
            # These words must have cov > 1
            # Words that gain coverage: N(c_prime) \ N(c)
            # No constraint on them (they just get cov >= 1)
            
            # Efficient check:
            # 1. Identify words in N(c) that are NOT in N(c_prime)
            # 2. Check if any of them have cov == 1
            
            # Using sets for neighbors might be slower than lists for small N, but here N=15.
            # Let's use a temporary boolean array or set difference.
            # Since neighbors are small, set difference is fine.
            set_c = set(neighbors[c])
            set_cp = set(neighbors[c_prime])
            lost = set_c - set_cp
            
            valid_move = True
            for w in lost:
                if cov[w] == 1:
                    valid_move = False
                    break
            
            if not valid_move:
                continue
                
            # If valid, calculate change in critical count
            # Old critical count: sum(1 for x in cov if x == 1)
            # We can compute delta
            # Words in lost: cov decreases by 1.
            #   If cov was 2, becomes 1 (Critical increases by 1)
            #   If cov was 1, invalid (already checked)
            #   If cov > 2, becomes > 1 (Critical unchanged)
            # Words in gained = set_cp - set_c: cov increases by 1.
            #   If cov was 0, becomes 1 (Critical increases by 1) -- WAIT, if cov was 0, code was invalid?
            #   No, we maintain validity. So cov >= 1 always.
            #   If cov was 1, becomes 2 (Critical decreases by 1)
            #   If cov > 1, becomes > 2 (Critical unchanged)
            
            delta_critical = 0
            for w in lost:
                if cov[w] == 2:
                    delta_critical += 1
            for w in set_cp - set_c:
                if cov[w] == 1:
                    delta_critical -= 1
            
            # Accept if delta_critical < 0 (reduces criticals)
            # Or with some probability if delta_critical == 0 (random walk)
            if delta_critical < 0 or (delta_critical == 0 and random.random() < 0.5):
                # Perform move
                for w in lost:
                    cov[w] -= 1
                for w in set_cp - set_c:
                    cov[w] += 1
                codewords.remove(c)
                codewords.add(c_prime)
                moved = True
                
        # If no move was made or after some moves, loop back to deletion
        # The deletion phase is fast, so it's okay to run it frequently.
        
    # Convert back to lists of ints
    return [from_idx(c) for c in best_codewords]
