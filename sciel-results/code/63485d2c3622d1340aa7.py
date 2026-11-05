import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    if n <= 6:
        # Standard construction is optimal for n <= 6
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
    
    if n == 7:
        # Generate S_6 standard
        word = [1]
        for m in range(2, 7):
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
        
        # Extract perms of 1..6 in the order they appear in S_6
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms) # 720
        
        # Precompute overlap matrix
        # ov[i][j] = overlap(perms[i], perms[j])
        # Overlap k if perms[i][-k:] == perms[j][:k]
        # Max k is 5 (since i != j)
        ov = [[0]*N for _ in range(N)]
        for i in range(N):
            p = perms[i]
            for j in range(N):
                if i == j: continue
                q = perms[j]
                # Check k from 5 down to 1
                # p[-5:] == q[:5]
                if p[-5:] == q[:5]:
                    ov[i][j] = 5
                elif p[-4:] == q[:4]:
                    ov[i][j] = 4
                elif p[-3:] == q[:3]:
                    ov[i][j] = 3
                elif p[-2:] == q[:2]:
                    ov[i][j] = 2
                elif p[-1:] == q[:1]:
                    ov[i][j] = 1
        
        # Start with standard order (identity)
        path = list(range(N))
        
        # Calculate initial total overlap
        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                total += ov[path[i]][path[i+1]]
            return total
        
        current_overlap = calc_total_overlap(path)
        
        # Local Search: 2-opt
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            for i in range(N - 1):
                if time.time() > deadline - 10: break
                for j in range(i + 1, N):
                    # Reverse path[i+1..j]
                    # Old edges: (i, i+1) and (j, j+1) [if j < N-1]
                    # New edges: (i, j) and (i+1, j+1) [if j < N-1]
                    
                    if j == N - 1:
                        old_ov = ov[path[i]][path[i+1]]
                        new_ov = ov[path[i]][path[j]]
                        if new_ov > old_ov:
                            path[i+1:j+1] = reversed(path[i+1:j+1])
                            current_overlap += new_ov - old_ov
                            improved = True
                            break
                    else:
                        old_ov = ov[path[i]][path[i+1]] + ov[path[j]][path[j+1]]
                        new_ov = ov[path[i]][path[j]] + ov[path[i+1]][path[j+1]]
                        if new_ov > old_ov:
                            path[i+1:j+1] = reversed(path[i+1:j+1])
                            current_overlap += new_ov - old_ov
                            improved = True
                            break
                if improved:
                    break
        
        # Construct S_7 using the optimized path
        new_word = []
        for idx in path:
            p = perms[idx]
            block = list(p) + [7] + list(p)
            overlap = 0
            min_len = min(len(new_word), len(block))
            for o in range(min_len, -1, -1):
                if new_word[len(new_word)-o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
            
        return "".join(map(str, new_word))
    
    # Fallback for n > 7
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
