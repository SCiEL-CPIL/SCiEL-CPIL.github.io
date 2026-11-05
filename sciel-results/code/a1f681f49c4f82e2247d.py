import itertools
import time

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Start with S_1
    S = [1]
    
    for m in range(2, n + 1):
        # Extract the sequence of permutations of 1..m-1 from S
        # The windows of length m-1 in S that are permutations of 1..m-1
        # Note: S contains only symbols 1..m-1 at this stage.
        # All windows of length m-1 in S are permutations of 1..m-1?
        # Yes, by construction of superpermutations.
        
        perms_order = []
        seen = set()
        # Iterate through all windows of length m-1 in S
        for i in range(len(S) - (m - 1) + 1):
            window = tuple(S[i:i + m - 1])
            if window not in seen:
                seen.add(window)
                perms_order.append(window)
                
        new_S = []
        for P in perms_order:
            # Form block: P, m, P
            block = list(P) + [m] + list(P)
            
            # Find max overlap between new_S and block
            max_overlap = min(len(new_S), len(block) - 1)
            overlap = 0
            for k in range(max_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if new_S[-k:] == block[:k]:
                    overlap = k
                    break
            
            new_S.extend(block[overlap:])
        
        S = new_S
        
    return "".join(map(str, S))
