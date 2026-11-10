import time
import random
import itertools
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    if n <= 1:
        return "".join(str(i) for i in range(1, n + 1))
    
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline - 10:
            break
            
        # Extract distinct (m-1) permutations from word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        
        if N <= 1:
            ordered_perms = perms
        else:
            perms_np = np.array(perms)
            
            # Compute overlap matrix
            # overlap_matrix[i][j] = max k such that suffix(perms[i], k) == prefix(perms[j], k)
            overlap_matrix = np.zeros((N, N), dtype=np.int8)
            
            for k in range(m - 1, 0, -1):
                prefix_map = {}
                for j in range(N):
                    pfx = tuple(perms_np[j][:k])
                    if pfx not in prefix_map:
                        prefix_map[pfx] = []
                    prefix_map[pfx].append(j)
                
                for i in range(N):
                    sfx = tuple(perms_np[i][-k:])
                    if sfx in prefix_map:
                        for j in prefix_map[sfx]:
                            if overlap_matrix[i][j] == 0:
                                overlap_matrix[i][j] = k
            
            # Search for best ordering
            # Use multiple greedy restarts with soft-max
            best_score = -1
            best_path = None
            
            search_deadline = deadline - 5
            restart_count = 0
            
            while time.time() < search_deadline:
                start = random.randint(0, N - 1)
                path = [start]
                visited = np.zeros(N, dtype=bool)
                visited[start] = True
                current = start
                score = 0.0
                
                for _ in range(N - 1):
                    overlaps = overlap_matrix[current].copy()
                    overlaps[visited] = -100
                    
                    # Soft-max
                    max_ov = np.max(overlaps)
                    if max_ov > -100:
                        # Temperature: start high, decay?
                        # Fixed temp for simplicity, or decay based on step?
                        temp = 0.5
                        exp_vals = np.exp((overlaps - max_ov) / temp)
                        total = np.sum(exp_vals)
                        if total > 0:
                            probs = exp_vals / total
                            best_next = np.random.choice(N, p=probs)
                        else:
                            best_next = np.argmax(overlaps)
                    else:
                        best_next = np.argmax(overlaps)
                        
                    visited[best_next] = True
                    path.append(best_next)
                    score += overlap_matrix[current, best_next]
                    current = best_next
                    
                if score > best_score:
                    best_score = score
                    best_path = path
                    
                restart_count += 1
                # If we have done enough restarts, maybe break early?
                # No, use all time.
                
            if best_path is not None:
                ordered_perms = [perms[i] for i in best_path]
            else:
                ordered_perms = perms
            
        # Build new word
        new = []
        for idx in range(len(ordered_perms)):
            p = ordered_perms[idx]
            block = list(p) + [m] + list(p)
            
            if idx == 0:
                new += block
            else:
                max_ov = min(len(new), len(block))
                ov = 0
                for k in range(max_ov, 0, -1):
                    if new[-k:] == block[:k]:
                        ov = k
                        break
                new += block[ov:]
        
        word = new
        
    return "".join(map(str, word))
