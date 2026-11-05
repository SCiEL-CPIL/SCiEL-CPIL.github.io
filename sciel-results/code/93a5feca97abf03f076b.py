import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Pre-compute balls
    # Map tuple to index
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx
        
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n-1, -1, -1):
            t[i] = idx % q
            idx //= q
        return tuple(t)
        
    total_words = q ** n
    balls = [0] * total_words
    
    # Pre-compute balls as bitsets
    for idx in range(total_words):
        t = idx_to_tuple(idx)
        mask = 1 << idx
        for i in range(n):
            for val in range(q):
                if val != t[i]:
                    new_t = list(t)
                    new_t[i] = val
                    mask |= 1 << tuple_to_idx(tuple(new_t))
        balls[idx] = mask
        
    def greedy_cover(randomize_ties=False):
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            best_gain = -1
            best_candidates = []
            
            for c in range(total_words):
                # Count how many currently uncovered words are in ball[c]
                gain = (balls[c] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(c)
                    
            if not best_candidates:
                break # Should not happen if coverage is possible
                
            if randomize_ties and len(best_candidates) > 1:
                c = random.choice(best_candidates)
            else:
                c = best_candidates[0]
                
            code.append(c)
            uncovered &= ~balls[c]
            
        return code
        
    def prune_code(code):
        # Count coverage for each word
        cover_count = [0] * total_words
        for c in code:
            b = balls[c]
            # Iterate set bits in b
            while b:
                lsb = b & -b
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                b ^= lsb
                
        pruned_code = []
        # Try to remove each codeword
        # We iterate through the code. If we remove one, we update counts.
        # To make it efficient, we can just check if any word has count 1 and is covered by this codeword.
        # Actually, simpler: 
        # For each c in code:
        #   Check if there exists a word w in ball[c] such that cover_count[w] == 1.
        #   If yes, c is essential.
        #   If no, c is redundant. Remove it, decrement counts for ball[c].
        
        essential = [False] * len(code)
        
        for i, c in enumerate(code):
            b = balls[c]
            is_essential = False
            while b:
                lsb = b & -b
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    is_essential = True
                    break
                b ^= lsb
            essential[i] = is_essential
            
        # If a codeword is not essential, we can remove it.
        # But removing one might make another non-essential? No, removing a codeword only decreases counts.
        # So if a codeword was essential before removal, it remains essential or becomes more essential.
        # If it was non-essential, it stays non-essential (or becomes essential if counts drop to 1? No, counts drop, so if it was >1, it might become 1).
        # Wait. If cover_count[w] > 1 for all w in ball[c], then c is redundant.
        # If we remove c, cover_count[w] decreases by 1.
        # It is possible that after removing c, some other codeword d becomes redundant?
        # Yes. So we should iterate.
        
        current_code = list(code)
        current_cover_count = list(cover_count)
        
        changed = True
        while changed:
            changed = False
            for i in range(len(current_code)):
                c = current_code[i]
                b = balls[c]
                is_essential = False
                while b:
                    lsb = b & -b
                    idx = lsb.bit_length() - 1
                    if current_cover_count[idx] == 1:
                        is_essential = True
                        break
                    b ^= lsb
                    
                if not is_essential:
                    # Remove c
                    b = balls[c]
                    while b:
                        lsb = b & -b
                        idx = lsb.bit_length() - 1
                        current_cover_count[idx] -= 1
                        b ^= lsb
                    current_code.pop(i)
                    changed = True
                    break # Restart loop to be safe, or continue? Restart is safer.
                    
        return current_code

    best_code = None
    best_len = float('inf')
    
    # Run greedy with deterministic tie-breaking first
    code1 = greedy_cover(randomize_ties=False)
    pruned1 = prune_code(code1)
    if len(pruned1) < best_len:
        best_len = len(pruned1)
        best_code = pruned1
        
    # Run greedy with random tie-breaking multiple times
    runs = 0
    while time.time() < deadline:
        code_r = greedy_cover(randomize_ties=True)
        pruned_r = prune_code(code_r)
        if len(pruned_r) < best_len:
            best_len = len(pruned_r)
            best_code = pruned_r
        runs += 1
        
    # Convert indices to list of digits
    result = [list(idx_to_tuple(idx)) for idx in best_code]
    return result
