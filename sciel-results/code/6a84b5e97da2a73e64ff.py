import itertools
import random

def construct(n):
    if n < 7:
        return standard_construct(n)
    
    # For n=7, use TSP on permutations of 1..6
    symbols = list(range(1, 7))
    perms = list(itertools.permutations(symbols))
    num_perms = len(perms)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    overlap = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            # Max overlap is 6
            for k in range(6, -1, -1):
                if pi[-k:] == pj[:k]:
                    overlap[i][j] = k
                    break
    
    best_length = float('inf')
    best_path = None
    
    # Greedy with multiple restarts
    for _ in range(100):
        start = random.randint(0, num_perms - 1)
        visited = [False] * num_perms
        visited[start] = True
        path = [start]
        total_overlap = 0
        
        current = start
        for _ in range(num_perms - 1):
            best_next = -1
            best_ov = -1
            # Find unvisited with max overlap
            for j in range(num_perms):
                if not visited[j]:
                    ov = overlap[current][j]
                    if ov > best_ov:
                        best_ov = ov
                        best_next = j
            
            if best_next == -1:
                break # Should not happen
            
            visited[best_next] = True
            path.append(best_next)
            total_overlap += best_ov
            current = best_next
        
        # Calculate length
        # Len = 13 + 719 * 13 - total_overlap
        length = 13 + (num_perms - 1) * 13 - total_overlap
        
        if length < best_length:
            best_length = length
            best_path = path[:]
            
    # Construct string from best_path
    if best_path is None:
        return standard_construct(n)
        
    # Build the word
    word = []
    # First block
    p = perms[best_path[0]]
    block = list(p) + [7] + list(p)
    word.extend(block)
    
    for idx in range(1, len(best_path)):
        p = perms[best_path[idx]]
        block = list(p) + [7] + list(p)
        # Overlap with current end of word
        # The end of word is the suffix of the previous block, which is perms[best_path[idx-1]]
        # The start of block is perms[best_path[idx]]
        # We need to find the overlap between word[-k:] and block[:k]
        # We know the overlap is overlap[best_path[idx-1]][best_path[idx]]
        ov = overlap[best_path[idx-1]][best_path[idx]]
        # But wait, the previous block is p_prev 7 p_prev.
        # The suffix is p_prev.
        # The prefix of the new block is p_curr.
        # The overlap calculated in the matrix is between p_prev and p_curr.
        # Is it possible that the overlap extends into the 7?
        # p_prev has no 7. p_curr 7 p_curr has 7 at index 6.
        # If ov > 6, it would imply matching 7, which is impossible.
        # So ov is at most 6.
        # So we just append block[ov:]
        word.extend(block[ov:])
        
    return "".join(map(str, word))

def standard_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
