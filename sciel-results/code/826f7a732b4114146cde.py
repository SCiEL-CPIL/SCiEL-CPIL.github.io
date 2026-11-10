import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        
        # Extract permutations of length m-1 in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        num_perms = len(perms)
        
        if num_perms <= 1:
            # Only one permutation, just add the block
            p = perms[0]
            block = list(p) + [m] + list(p)
            # Overlap with current word end
            max_ov = min(len(word), len(block))
            ov = 0
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if word[len(word)-o:] == block[:o]:
                    ov = o
                    break
            word += block[ov:]
            continue
        
        # Build overlap matrix
        # overlap[p_idx][q_idx] = max k where perms[p_idx][-k:] == perms[q_idx][:k]
        # Cost of going p -> q is (m-1) - overlap
        perm_list = list(perms)
        perm_idx = {p: i for i, p in enumerate(perm_list)}
        
        # Precompute overlaps
        # For each pair (i,j), compute overlap between perm_list[i] and perm_list[j]
        # overlap = max k in 1..(m-1) such that perm_list[i][-k:] == perm_list[j][:k]
        
        # For efficiency, precompute for each permutation its suffixes
        # Actually, let's just compute directly for small m
        
        overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
        
        for i in range(num_perms):
            pi = perm_list[i]
            for j in range(num_perms):
                pj = perm_list[j]
                # Find max k where pi[-k:] == pj[:k]
                max_k = m - 1
                for k in range(max_k, 0, -1):
                    if pi[-k:] == pj[:k]:
                        overlap_matrix[i][j] = k
                        break
        
        # Now we need to find an ordering of all permutations starting from a fixed start
        # The start is determined by the last (m-1)-permutation in the current word
        # Actually, the first permutation in the new word should overlap with the end of the current word
        
        # Let's think: the current word ends with some (m-1)-permutation. 
        # Actually, the last (m-1) symbols of the current word form a permutation.
        # We want the first block to overlap maximally with the end of the word.
        
        # The end of the current word is word[-(m-1):]
        # We want to find the permutation p that maximizes overlap with this end
        # i.e., max k where word[-(m-1):][-k:] == p[:k], i.e., word[-k:] == p[:k]
        
        # Actually, the overlap between the current word end and the first block p+[m]+p:
        # The block starts with p, so we want max k where word[-k:] == p[:k]
        
        end_word = tuple(word[-(m-1):])
        
        # Find best starting permutation
        best_start = 0
        best_start_overlap = 0
        for i in range(num_perms):
            p = perm_list[i]
            # Max k where word[-k:] == p[:k]
            for k in range(m-1, 0, -1):
                if tuple(word[-k:]) == p[:k]:
                    if k > best_start_overlap:
                        best_start_overlap = k
                        best_start = i
                    break
        
        # Greedy construction starting from best_start
        # At each step, pick the next permutation that minimizes cost (maximizes overlap)
        
        visited = [False] * num_perms
        visited[best_start] = True
        current = best_start
        order = [current]
        
        # For efficiency with large num_perms, use a priority-based approach
        # But for m=7, num_perms=720, this is manageable
        
        for _ in range(num_perms - 1):
            # Find unvisited permutation with maximum overlap from current
            best_next = -1
            best_ov = -1
            for j in range(num_perms):
                if not visited[j]:
                    ov = overlap_matrix[current][j]
                    if ov > best_ov:
                        best_ov = ov
                        best_next = j
            
            if best_next == -1:
                break
            
            visited[best_next] = True
            order.append(best_next)
            current = best_next
        
        # Now construct the new word
        new_word = list(word)
        
        for idx in order:
            p = perm_list[idx]
            block = list(p) + [m] + list(p)
            
            # Find overlap with end of new_word
            max_ov = min(len(new_word), len(block))
            ov = 0
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new_word[len(new_word)-o:] == block[:o]:
                    ov = o
                    break
            
            new_word += block[ov:]
        
        word = new_word
    
    return "".join(map(str, word))
