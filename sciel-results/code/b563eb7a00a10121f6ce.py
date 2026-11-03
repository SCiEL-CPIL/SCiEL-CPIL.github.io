import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n == 1:
        return "1"
    
    word = "1"
    for m in range(2, n + 1):
        if time.time() > deadline:
            # If we time out, we are in trouble for n=7.
            # But n=7 construction is fast.
            pass
            
        perms, seen = [], set()
        # Extract permutations of length m-1 from word
        # word is a string
        for i in range(len(word) - m + 2):
            window = word[i:i + m - 1]
            # Check if window contains all unique chars from 1..m-1
            # Since m is small, we can just check length of set
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
                
        new = ""
        num_perms = len(perms)
        
        for idx, p in enumerate(perms):
            block = p + str(m) + p
            
            # Find max overlap
            max_overlap = min(len(new), len(block))
            best_overlap = 0
            # Find all valid overlaps? Or just max?
            # For lookahead, we need top 3.
            candidates = []
            
            # We check k from max_overlap down to 0
            # But we stop after finding 3 valid ones?
            # Actually, overlaps are discrete.
            # We can just check max, max-1, max-2.
            
            # Optimization: 
            # The suffix of new must match prefix of block.
            
            found = 0
            for k in range(max_overlap, -1, -1):
                if new.endswith(block[:k]):
                    candidates.append(k)
                    found += 1
                    if found >= 3:
                        break
            
            if not candidates:
                # Should not happen, k=0 always works
                candidates = [0]
            
            best_k = candidates[0]
            
            if idx < num_perms - 1:
                next_block = perms[idx+1] + str(m) + perms[idx+1]
                
                best_score = -1
                for k in candidates:
                    # Simulate appending
                    # new_candidate = new + block[k:]
                    # We don't need to construct the full string to check next overlap?
                    # We need to check if new_candidate ends with next_block[:k_next]
                    # new_candidate ends with block[k:]
                    # So we need to check if (new + block[k:]) ends with next_block[:k_next]
                    
                    # This is complex. 
                    # Approximation:
                    # The suffix of the new word is determined by block[k:].
                    # If k is large, the suffix is mostly from block.
                    # If k is small, the suffix is from new.
                    
                    # Let's just construct the suffix part?
                    # Or just pick max overlap greedily?
                    # The 2-step lookahead might be too complex to implement correctly without full string construction.
                    
                    # Let's try a simpler heuristic:
                    # Pick max overlap.
                    pass
            
            # Fallback: Greedy max overlap
            # This is safe.
            new += block[best_k:]
            
        word = new
        
    return word
