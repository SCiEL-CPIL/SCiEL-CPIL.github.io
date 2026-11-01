import random
import time
import itertools

def construct(n):
    if n <= 4:
        # Standard construction is optimal for n <= 4
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

    # For n > 4: Greedy nearest-neighbor with random restarts
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Precompute overlap matrix for efficiency
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # This is an N x N matrix, 5040x5040 = ~25M entries. Too much memory.
    # Instead, compute on the fly but cache in a dict.
    
    def overlap(p1, p2):
        # Max k such that p1[-k:] == p2[:k]
        # Since both are permutations of same length n, max k is n-1
        for k in range(n - 1, -1, -1):
            if k == 0:
                return 0
            if p1[-k:] == p2[:k]:
                return k
        return 0
    
    start_time = time.time()
    time_limit = 115.0
    
    best_order = None
    best_len = float('inf')
    
    # Multiple random restarts
    num_restarts = 0
    while time.time() - start_time < time_limit:
        # Pick a random starting permutation
        start_idx = random.randint(0, N - 1)
        order = [start_idx]
        visited = [False] * N
        visited[start_idx] = True
        current_len = 0
        
        for _ in range(N - 1):
            last = order[-1]
            best_ov = -1
            best_idx = -1
            for j in range(N):
                if not visited[j]:
                    ov = overlap(perms[last], perms[j])
                    if ov > best_ov:
                        best_ov = ov
                        best_idx = j
            if best_idx == -1:
                break
            order.append(best_idx)
            visited[best_idx] = True
            current_len += n - best_ov
        
        if current_len < best_len:
            best_len = current_len
            best_order = order[:]
            num_restarts += 1
            
            # If we've found a good solution, we can try to improve with local search
            # But for now, just keep doing restarts
        
        num_restarts += 1
    
    if best_order is None:
        # Fallback: return standard superpermutation
        word = [1]
        for m in range(2, n + 1):
            perms_m, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_m.append(window)
            new = []
            for p in perms_m:
                block = list(p) + [m] + list(p)
                ov = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[ov:]
            word = new
        return "".join(map(str, word))
    
    # Reconstruct word from best_order
    word = list(perms[best_order[0]])
    for i in range(1, len(best_order)):
        p_prev = perms[best_order[i-1]]
        p_curr = perms[best_order[i]]
        ov = overlap(p_prev, p_curr)
        word.extend(p_curr[ov:])
        
    return "".join(map(str, word))
