import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    random.seed(42) # Fixed seed for reproducibility in this context, but we will shuffle

    def is_valid_extension(current_products, new_tails):
        for p in new_tails:
            if p in current_products:
                return False
        return True

    def get_tails(x, current_seq):
        tails = [x]
        prod = x
        for a in reversed(current_seq):
            prod *= a
            tails.append(prod)
        return tails

    def dfs(last_picked, current_seq, current_products, candidates):
        nonlocal best_seq, best_len
        
        if time.time() > deadline:
            return

        current_len = len(current_seq)
        
        # Update best if current is better
        if current_len > best_len:
            best_len = current_len
            best_seq = current_seq[:]

        # Pruning: if we can't beat best_len with remaining numbers, stop
        # candidates are all > last_picked
        if current_len + len(candidates) <= best_len:
            return

        # Try adding next numbers from candidates
        # We iterate over a shuffled list of candidates to diversify search
        for idx, x in enumerate(candidates):
            # Calculate new tails
            tails = get_tails(x, current_seq)
            
            # Check for collisions
            if is_valid_extension(current_products, tails):
                # Add x
                current_seq.append(x)
                current_products.update(tails)
                
                # Recurse with remaining candidates (those > x)
                # Since candidates is a list of ints > last_picked, and we are picking x,
                # the next candidates must be > x.
                # We pass the slice of candidates that are > x.
                # Note: candidates is sorted in the initial call, but we shuffle it.
                # So we must filter.
                next_candidates = [c for c in candidates if c > x]
                
                dfs(x, current_seq, current_products, next_candidates)
                
                # Undo
                current_seq.pop()
                for p in tails:
                    current_products.remove(p)
                    
                if time.time() > deadline:
                    return

    # 1. Get a good initial lower bound using greedy
    # Greedy: pick smallest possible, then next smallest, etc.
    greedy_values = []
    greedy_products = set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        tails = get_tails(x, greedy_values)
        if not greedy_products.intersection(tails):
            greedy_values.append(x)
            greedy_products.update(tails)
            
    best_seq = greedy_values
    best_len = len(greedy_values)
    
    # 2. Run multiple randomized DFS searches
    # We will run DFS with different shuffles of the candidate pool
    # To be efficient, we limit the number of restarts based on time
    all_candidates = list(range(2, 129))
    
    # Run a few restarts
    for restart in range(100): # Cap restarts, but check time inside
        if time.time() > deadline - 5:
            break
            
        # Shuffle candidates to explore different branches
        shuffled_candidates = all_candidates[:]
        random.shuffle(shuffled_candidates)
        
        # Start DFS from empty sequence
        dfs(1, [], set(), shuffled_candidates)

    return best_seq
