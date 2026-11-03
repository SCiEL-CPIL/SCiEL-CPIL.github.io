import time
import sys
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # S4 length 36
        # S3 = 123121321
        # Perms of S3: 123, 231, 312, 213, 132, 321
        # S4 construction:
        # Start with 123, 4.
        # Next 231. Overlap?
        # Standard construction yields 36.
        pass

    # Recursive Construction
    # S_1 = [1]
    # S_n from S_{n-1}
    
    # This specific construction is:
    # Let P be the list of permutations of 1..n-1 in the order they appear in S_{n-1}.
    # S_n is formed by concatenating:
    # P_1 + [n]
    # Then for each subsequent P_i, we append the suffix of P_i that does not overlap with the current end of S_n.
    
    # Correct Algorithm:
    # S_n is obtained by taking S_{n-1} and inserting n after every element, 
    # then removing the last n-1 elements of the result?
    
    # Let's try:
    # S2 = 121.
    # Insert 3: 132313.
    # Remove last 2: 13231.
    # Length 5.
    # Perms: 132, 323, 231.
    # Missing 123, 213, 312, 321.
    
    # I will use the Greedy approach but with a faster overlap check.
    # The current greedy is the bottleneck.
    
    perms = list(itertools.permutations(range(1, n + 1)))
    perms_str = [''.join(map(str, p)) for p in perms]
    
    word = perms_str[0]
    unused = perms_str[1:]
    
    # Optimization: Precompute prefixes of unused perms?
    # No, just check.
    
    # To speed up, we can break early if we find n-1 overlap.
    
    while unused:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_ov = -1
        
        for i, p in enumerate(unused):
            # Check overlap
            # Max possible is n-1
            # We check from n-1 down to 1
            # If we find n-1, we break immediately.
            
            # Optimization: 
            # If best_ov is already n-1, we can break the outer loop?
            # No, we need to find the FIRST one or any one.
            # If we find n-1, it's the best possible.
            
            for k in range(n - 1, 0, -1):
                if word[-k:] == p[:k]:
                    if k > best_ov:
                        best_ov = k
                        best_idx = i
                    break # Found max overlap for this p. Move to next p.
                    
            if best_ov == n - 1:
                # Can't get better. Break outer loop?
                # Yes, if we just want any valid continuation.
                # But maybe tie-breaking matters?
                # For now, assume first found is fine.
                break
                
        if best_idx == -1:
            break
            
        best_p = unused[best_idx]
        word += best_p[best_ov:]
        
        unused[best_idx] = unused[-1]
        unused.pop()
        
    return word
