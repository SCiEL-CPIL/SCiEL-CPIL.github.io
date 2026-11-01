"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct(q, n, radius):
    # For the specific task: q=3, n=7, radius=1
    # We use simulated annealing starting from a greedy construction.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to direct sum for other parameters
        # For general case, use a simple greedy or direct sum if possible
        # Here we focus on the specific task q=3, n=7, r=1
        # Direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Fallback
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Specific algorithm for q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    
    # 1. Greedy Initialization
    # Start with an empty set of codewords.
    # At each step, pick a word that covers the maximum number of currently uncovered words.
    # To speed this up, we can use a heuristic.
    
    # Let's represent the state as a set of codewords.
    # We need to track which words are covered.
    
    # A word w is covered if there exists c in Code such that dist(w, c) <= 1.
    # dist(w, c) <= 1 means w == c or they differ in exactly one position.
    # So w is covered if w is in Code, or any neighbor of w is in Code.
    
    # Let's try a different greedy strategy:
    # Start with a random set of codewords, say size 200.
    # Then use local search to reduce the size.
    
    # Better: Start with a known good structure or a greedy packing.
    
    # Greedy Construction:
    # 1. Create a set of all possible words.
    # 2. While there are uncovered words:
    #    a. Pick an uncovered word w.
    #    b. Add w to the code.
    #    c. Mark w and all its neighbors as covered.
    # This produces a maximal code, but not necessarily minimal.
    
    # Let's implement a greedy that picks the word with maximum coverage.
    # This is expensive (O(3^n * 3^n)) if done naively.
    
    # Alternative: Start with a random set of size ~200, then anneal.
    
    # Let's use Simulated Annealing directly.
    # State: List of codewords.
    # Move: Remove a codeword, Add a codeword, or Replace a codeword.
    # Objective: Minimize |Code| subject to Coverage == 2187.
    
    # To make it efficient, we need to quickly check if a code covers the space.
    # We can maintain a count of how many times each word is covered.
    # count[w] = number of codewords c such that dist(w, c) <= 1.
    # A word is covered if count[w] > 0.
    # Total covered = sum(1 for w in all_words if count[w] > 0).
    
    # When we add a codeword c, we increment count[w] for all w in N(c) (c and its 6 neighbors).
    # When we remove a codeword c, we decrement count[w] for all w in N(c).
    # This is O(7) per move (1 + 6 neighbors).
    
    # Let's implement this.
    
    all_words = list(itertools.product(range(3), repeat=7))
    # Map word to index for fast lookup
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (including itself)
    # neighbors[w] = list of indices of words within distance 1 of w
    neighbors = []
    for w in all_words:
        w_list = list(w)
        idxs = [word_to_idx[w]]
        for i in range(7):
            orig = w_list[i]
            for v in range(3):
                if v != orig:
                    w_list[i] = v
                    idxs.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig
        neighbors.append(idxs)
    
    # Initialize with a greedy solution
    # Greedy: Start with empty code.
    # counts = [0] * 2187
    # covered = 0
    # code = []
    # 
    # To speed up greedy, we can just pick random words until covered, then try to remove.
    # Or use a simple greedy:
    # while covered < 2187:
    #    pick a random uncovered word w
    #    add w to code
    #    update counts and covered
    
    # Let's start with a random set of size 250, then anneal down.
    
    random.seed(42)
    
    # Greedy construction to get a valid starting point
    counts = [0] * 2187
    covered = 0
    code = []
    uncovered_indices = set(range(2187))
    
    # Simple greedy: pick random uncovered, add it.
    # This will likely end up with ~250-300 words.
    while uncovered_indices:
        w_idx = random.choice(list(uncovered_indices))
        code.append(all_words[w_idx])
        for nb_idx in neighbors[w_idx]:
            if counts[nb_idx] == 0:
                covered += 1
                uncovered_indices.remove(nb_idx)
            counts[nb_idx] += 1
    
    # Now we have a valid code.
    # Let's try to reduce its size using Simulated Annealing.
    
    # State: code (list of tuples), counts (list of ints)
    # We want to minimize len(code).
    # Moves:
    # 1. Remove a random codeword. If coverage is maintained, accept. Else, accept with prob exp(-delta/T).
    #    But we must maintain validity? No, SA allows invalid states but penalizes them.
    #    However, we want to end with a valid code.
    #    Better: Only allow moves that keep the code valid?
    #    If we only allow valid moves, we are doing Local Search.
    #    To escape local minima, we need to allow invalid states.
    #    Objective function: f(state) = len(code) + penalty * (2187 - covered)
    #    We want to minimize f.
    
    # Let's use a penalty approach.
    # f = len(code) + P * (2187 - covered)
    # If P is large, we prioritize coverage.
    
    # Start temperature T
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    current_code = code
    current_counts = counts
    current_covered = covered
    current_cost = len(current_code) + 10 * (2187 - current_covered)
    
    best_code = list(current_code)
    best_cost = current_cost
    
    start_time = time.time()
    time_limit = 115  # Leave some margin
    
    while time.time() - start_time < time_limit:
        # Generate a move
        # Option 1: Remove a codeword
        # Option 2: Add a codeword
        # Option 3: Replace a codeword
        
        move_type = random.random()
        
        new_code = list(current_code)
        new_counts = list(current_counts)
        new_covered = current_covered
        
        if move_type < 0.4:
            # Remove
            if not new_code:
                continue
            idx = random.randint(0, len(new_code) - 1)
            w = new_code[idx]
            w_idx = word_to_idx[w]
            new_code.pop(idx)
            for nb_idx in neighbors[w_idx]:
                new_counts[nb_idx] -= 1
                if new_counts[nb_idx] == 0:
                    new_covered -= 1
        elif move_type < 0.7:
            # Add
            # Pick a random word, preferably one that is uncovered or has low count
            # To bias towards useful additions, pick from uncovered or low count
            # For simplicity, pick random
            w = random.choice(all_words)
            w_idx = word_to_idx[w]
            new_code.append(w)
            for nb_idx in neighbors[w_idx]:
                if new_counts[nb_idx] == 0:
                    new_covered += 1
                new_counts[nb_idx] += 1
        else:
            # Replace
            if not new_code:
                continue
            idx = random.randint(0, len(new_code) - 1)
            old_w = new_code[idx]
            old_w_idx = word_to_idx[old_w]
            
            new_w = random.choice(all_words)
            new_w_idx = word_to_idx[new_w]
            
            if old_w_idx == new_w_idx:
                continue
                
            new_code[idx] = new_w
            
            # Update counts for old_w (decrement)
            for nb_idx in neighbors[old_w_idx]:
                new_counts[nb_idx] -= 1
                if new_counts[nb_idx] == 0:
                    new_covered -= 1
            
            # Update counts for new_w (increment)
            for nb_idx in neighbors[new_w_idx]:
                if new_counts[nb_idx] == 0:
                    new_covered += 1
                new_counts[nb_idx] += 1
        
        new_cost = len(new_code) + 10 * (2187 - new_covered)
        
        delta = new_cost - current_cost
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            current_code = new_code
            current_counts = new_counts
            current_covered = new_covered
            current_cost = new_cost
            
            if new_covered == 2187:
                if len(current_code) < len(best_code):
                    best_code = list(current_code)
                    best_cost = current_cost
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # If we have a valid code with 186 words, we can stop early?
        # No, try to find better.
        
    # Ensure the returned code is valid
    # If best_code is not valid, fall back to current if valid, else original greedy
    # Check validity of best_code
    # Since we tracked counts, we can verify best_code
    # But best_code was saved when it was valid (new_covered == 2187)
    
    # Return best_code
    return [list(w) for w in best_code]
# EVOLVE-BLOCK-END
