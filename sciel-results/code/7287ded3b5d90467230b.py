import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    if n == 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline - 5.0:
            # Not enough time for complex search, use standard fast construction
            # Standard construction for remaining steps
            while m <= n:
                perms, seen = [], set()
                for i in range(len(word) - m + 2):
                    window = tuple(word[i:i + m - 1])
                    if len(set(window)) == m - 1 and window not in seen:
                        seen.add(window)
                        perms.append(window)
                new = []
                for p in perms:
                    block = list(p) + [m] + list(p)
                    # Find overlap
                    overlap = 0
                    max_ov = min(len(new), len(block))
                    for o in range(max_ov, -1, -1):
                        if new[len(new) - o:] == block[:o]:
                            overlap = o
                            break
                    new += block[overlap:]
                word = new
                m += 1
            return "".join(map(str, word))
            
        # Get permutations
        # For m <= 3, standard order is optimal/fast
        if m <= 3:
             perms, seen = [], set()
             for i in range(len(word) - m + 2):
                 window = tuple(word[i:i + m - 1])
                 if len(set(window)) == m - 1 and window not in seen:
                     seen.add(window)
                     perms.append(window)
             # Standard concatenation
             new = []
             for p in perms:
                 block = list(p) + [m] + list(p)
                 overlap = 0
                 max_ov = min(len(new), len(block))
                 for o in range(max_ov, -1, -1):
                     if new[len(new) - o:] == block[:o]:
                         overlap = o
                         break
                 new += block[overlap:]
             word = new
             continue
             
        # For m >= 4, use greedy reordering
        perms = list(itertools.permutations(range(1, m)))
        k_perms = len(perms)
        
        # Precompute prefixes?
        # perms are tuples.
        
        used = [False] * k_perms
        current_idx = 0
        used[0] = True
        current_block = list(perms[0]) + [m] + list(perms[0])
        
        for step in range(k_perms - 1):
            # Find best next
            best_idx = -1
            best_overlap = -1
            
            # Suffix of current_block
            # We only care about the last m-1 elements
            suffix = tuple(current_block[-(m-1):])
            
            for j in range(k_perms):
                if used[j]:
                    continue
                prefix = perms[j]
                # Max overlap
                # Check k from m-1 down to 1
                # Since perms are distinct, overlap < m-1 is likely, but can be m-1?
                # If suffix == prefix, overlap is m-1.
                # This implies P_i suffix == P_j prefix.
                
                # Optimization:
                # If we store perms as tuples, slicing is fast.
                for k in range(m-1, 0, -1):
                    if suffix[-k:] == prefix[:k]:
                        if k > best_overlap:
                            best_overlap = k
                            best_idx = j
                        break
            
            if best_idx == -1:
                # Fallback: any unused
                for j in range(k_perms):
                    if not used[j]:
                        best_idx = j
                        break
            
            used[best_idx] = True
            new_block = list(perms[best_idx]) + [m] + list(perms[best_idx])
            current_block += new_block[best_overlap:]
            
        word = current_block
        
    return "".join(map(str, word))
