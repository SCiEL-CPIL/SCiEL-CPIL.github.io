import sys

sys.setrecursionlimit(10000)

def construct():
    # Approach: Backtracking with Descending Order and Greedy Initial Bound
    
    best_len = 0
    best_seq = []
    
    # 1. Generate a good initial solution using a greedy heuristic
    # We try to include numbers from 128 down to 2 if they don't cause conflicts.
    # This tends to work well because large numbers produce large, distinct products.
    greedy_seq = []
    greedy_prods = set()
    greedy_sufs = []
    
    for x in range(128, 1, -1):
        new_tails = [x]
        for p in greedy_sufs:
            new_tails.append(x * p)
        
        # Check for collisions
        collision = False
        for nt in new_tails:
            if nt in greedy_prods:
                collision = True
                break
                
        if not collision:
            greedy_seq.append(x)
            greedy_prods.update(new_tails)
            greedy_sufs = [x] + [x * p for p in greedy_sufs]
            
    best_len = len(greedy_seq)
    best_seq = greedy_seq[:]
    
    # 2. Backtracking to find a potentially longer sequence
    # We iterate x from 128 down to 2.
    # At each step, we decide to include or exclude x.
    # Pruning: If current_length + remaining_numbers <= best_len, we cannot improve.
    
    def dfs(idx, seq, prods, sufs):
        nonlocal best_len, best_seq
        
        # Pruning: Max possible length from this state
        # Remaining numbers: idx, idx-1, ..., 2 (count: idx - 1)
        if len(seq) + (idx - 1) <= best_len:
            return
            
        if idx < 2:
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
            return
        
        x = idx
        
        # Option 1: Include x
        # Calculate new tails: x, x*sufs[0], x*sufs[1], ...
        new_tails = [x]
        for p in sufs:
            new_tails.append(x * p)
            
        # Check collision with existing products
        valid = True
        for nt in new_tails:
            if nt in prods:
                valid = False
                break
                
        if valid:
            # Create new state
            # To avoid copying the entire set every time (which is expensive), 
            # we can use a "undo" mechanism, but for simplicity and given the pruning, 
            # copying might be acceptable if the search space is small.
            # However, set.union is O(N). Let's try to optimize if needed.
            # For now, we copy.
            new_prods = prods.union(new_tails)
            new_sufs = [x] + [x * p for p in sufs]
            new_seq = seq + [x]
            
            dfs(x - 1, new_seq, new_prods, new_sufs)
            
        # Option 2: Exclude x
        # This is often the more "promising" path for length if including x blocks many others,
        # but since we prefer length, we tried Include first.
        # We must also explore Exclude.
        dfs(x - 1, seq, prods, sufs)
        
    dfs(128, [], set(), [])
    
    return best_seq
