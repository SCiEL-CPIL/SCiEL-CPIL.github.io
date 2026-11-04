# EVOLVE-BLOCK-START
import time
import random
import math

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
        
        # Extract perms of 1..6
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        
        # Precompute overlap matrix
        # ov[i][j] = longest k such that perms[i][-k:] == perms[j][:k]
        # Max possible is 5 (distinct perms can't have overlap 6)
        ov = [[0] * N for _ in range(N)]
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    ov[i][j] = 6
                    continue
                pj = perms[j]
                for k in range(5, 0, -1):
                    if pi[-k:] == pj[:k]:
                        ov[i][j] = k
                        break
        
        def path_cost(path):
            """Total overlap along the path. We want to maximize this."""
            total = 0
            for idx in range(len(path) - 1):
                total += ov[path[idx]][path[idx+1]]
            return total
        
        def two_opt(path, ov, N, deadline):
            """Improve path via 2-opt moves. Returns improved path."""
            current_cost = path_cost(path)
            improved = True
            while improved and time.time() < deadline - 30:
                improved = False
                for i in range(len(path) - 2):
                    if time.time() > deadline - 30:
                        break
                    for j in range(i + 1, len(path) - 1):
                        # Old edges: (path[i], path[i+1]) and (path[j], path[j+1])
                        # New edges: (path[i], path[j]) and (path[i+1], path[j+1])
                        old_cost = ov[path[i]][path[i+1]]
                        if j < len(path) - 1:
                            old_cost += ov[path[j]][path[j+1]]
                        
                        new_cost = ov[path[i]][path[j]]
                        if j < len(path) - 1:
                            new_cost += ov[path[i+1]][path[j+1]]
                        
                        if new_cost > old_cost:
                            path[i+1:j+1] = path[i+1:j+1][::-1]
                            current_cost += new_cost - old_cost
                            improved = True
                            break
                    if improved:
                        break
            return path
        
        def nearest_neighbor(ov, N, deadline):
            """Construct a path using nearest neighbor heuristic."""
            unvisited = list(range(N))
            random.shuffle(unvisited)
            path = [unvisited.pop()]
            while unvisited:
                last = path[-1]
                best_idx = -1
                best_ov = -1
                for idx in unvisited:
                    if ov[last][idx] > best_ov:
                        best_ov = ov[last][idx]
                        best_idx = idx
                path.append(best_idx)
                unvisited.remove(best_idx)
            return path
        
        # Try multiple restarts
        best_path = None
        best_cost = -1
        
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 60:
                break
            
            path = nearest_neighbor(ov, N, deadline)
            path = two_opt(path, ov, N, deadline)
            cost = path_cost(path)
            
            if cost > best_cost:
                best_cost = cost
                best_path = path[:]
        
        if best_path is None:
            best_path = list(range(N))
        
        # Construct S_7 using the optimized path
        # For each perm p in path, the block is p + [7] + p
        # The overlap between consecutive blocks:
        # Block i ends with perms[path[i]] (the second copy)
        # Block i+1 starts with perms[path[i+1]] (the first copy)
        # So the overlap is ov[path[i]][path[i+1]]
        
        # Length = sum of (len(block_i) - overlap_i) for i=0..N-1
        # len(block_i) = 13 (6 + 1 + 6)
        # overlap_0 = 0 (no previous block)
        # overlap_i = ov[path[i-1]][path[i]] for i >= 1
        
        # Total length = 13 * N - sum(ov[path[i-1]][path[i]] for i in 1..N-1)
        # = 13 * 720 - best_cost
        # = 9360 - best_cost
        
        # But wait, we need to actually construct the word.
        # Let's build it efficiently.
        
        new_word = []
        for idx in range(len(best_path)):
            p = perms[best_path[idx]]
            block = list(p) + [7] + list(p)
            if not new_word:
                new_word = block[:]
            else:
                # Find overlap: suffix of new_word vs prefix of block
                # We know from the overlap matrix that the overlap should be
                # ov[best_path[idx-1]][best_path[idx]]
                # But let's verify by direct computation for safety
                overlap = 0
                min_len = min(len(new_word), len(block))
                for o in range(min_len, -1, -1):
                    if new_word[len(new_word)-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
        
        return "".join(map(str, new_word))
    
    # Fallback for n > 7 (should not happen per task, but good practice)
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
# EVOLVE-BLOCK-END
