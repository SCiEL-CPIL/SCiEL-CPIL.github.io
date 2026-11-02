import itertools
import time
import random
import math

def construct(n):
    if n < 7:
        # Use classic construction for small n
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
    
    # For n=7, try to optimize the order of 6-permutations
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Generate standard word to have a baseline
    # We can generate it quickly
    word_std = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word_std) - m + 2):
            window = tuple(word_std[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word_std = new
    std_len = len(word_std)
    best_word = word_std
    best_len = std_len
    
    # 2. Optimize for n=7
    # Permutations of 1..6
    perms_6 = list(itertools.permutations(range(1, 7)))
    N = len(perms_6)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms_6)}
    
    # Precompute overlap matrix
    # ov[i][j] = overlap(perms_6[i], perms_6[j])
    ov = [[0]*N for _ in range(N)]
    for i, p in enumerate(perms_6):
        for j, q in enumerate(perms_6):
            # Calculate overlap
            # Max overlap is 5
            for k in range(5, -1, -1):
                if p[6-k:] == q[:k]:
                    ov[i][j] = k
                    break
    
    def compute_path_length(path):
        """Compute the total length of the superpermutation for a given path."""
        total = 0
        for idx, p_idx in enumerate(path):
            if idx == 0:
                total += 13  # Full block length
            else:
                prev_p_idx = path[idx-1]
                ov_len = ov[prev_p_idx][p_idx]
                total += 13 - ov_len
        return total
    
    def greedy_path(start_idx=None):
        """Generate a greedy path starting from a given index."""
        if start_idx is None:
            start_idx = random.randint(0, N-1)
        path = [start_idx]
        remaining = set(range(N))
        remaining.remove(start_idx)
        current = start_idx
        while remaining:
            best_next = -1
            best_ov = -1
            for q in remaining:
                if ov[current][q] > best_ov:
                    best_ov = ov[current][q]
                    best_next = q
            path.append(best_next)
            remaining.remove(best_next)
            current = best_next
        return path
    
    def two_opt_improve(path, deadline):
        """Apply 2-opt improvement to the path until no more improvements or deadline."""
        total_weight = sum(ov[path[i]][path[i+1]] for i in range(N-1))
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(N-1):
                if time.time() > deadline - 10:
                    break
                for j in range(i+1, N-1):
                    # Reverse segment path[i+1] to path[j]
                    # Old edges: (path[i], path[i+1]) and (path[j], path[j+1])
                    # New edges: (path[i], path[j]) and (path[i+1], path[j+1])
                    old_w = ov[path[i]][path[i+1]] + ov[path[j]][path[j+1]]
                    new_w = ov[path[i]][path[j]] + ov[path[i+1]][path[j+1]]
                    if new_w > old_w:
                        total_weight += new_w - old_w
                        path[i+1:j+1] = path[i+1:j+1][::-1]
                        improved = True
        return path, total_weight
    
    def simulated_annealing(path, deadline):
        """Apply simulated annealing to the path."""
        current_path = path[:]
        current_weight = sum(ov[current_path[i]][current_path[i+1]] for i in range(N-1))
        best_path = current_path[:]
        best_weight = current_weight
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < deadline - 10 and T > min_T:
            i = random.randint(0, N-2)
            j = random.randint(0, N-2)
            if i > j:
                i, j = j, i
            
            # Try reversing segment [i+1, j]
            old_w = ov[current_path[i]][current_path[i+1]] + ov[current_path[j]][current_path[j+1]]
            new_w = ov[current_path[i]][current_path[j]] + ov[current_path[i+1]][current_path[j+1]]
            delta = new_w - old_w
            
            if delta > 0 or random.random() < math.exp(-delta / T):
                current_path[i+1:j+1] = current_path[i+1:j+1][::-1]
                current_weight += delta
                
                if current_weight < best_weight:
                    best_weight = current_weight
                    best_path = current_path[:]
            
            T *= cooling
        
        return best_path, best_weight
    
    # Try multiple strategies
    best_path = None
    best_weight = float('inf')
    
    # Strategy 1: Greedy from random starts with 2-opt
    for restart in range(20):
        if time.time() > deadline - 15:
            break
        path = greedy_path(random.randint(0, N-1))
        path, weight = two_opt_improve(path, deadline)
        if weight < best_weight:
            best_weight = weight
            best_path = path[:]
    
    # Strategy 2: Simulated annealing from best path found so far
    if best_path is not None and time.time() < deadline - 20:
        sa_path, sa_weight = simulated_annealing(best_path, deadline)
        if sa_weight < best_weight:
            best_weight = sa_weight
            best_path = sa_path[:]
    
    # Strategy 3: More 2-opt passes on the best path
    if best_path is not None and time.time() < deadline - 10:
        best_path, best_weight = two_opt_improve(best_path, deadline)
    
    # Construct word from optimized path
    if best_path is not None:
        word_opt = []
        for idx, p_idx in enumerate(best_path):
            p = perms_6[p_idx]
            block = list(p) + [7] + list(p)
            if idx == 0:
                word_opt = block
            else:
                prev_p_idx = best_path[idx-1]
                ov_len = ov[prev_p_idx][p_idx]
                word_opt += block[ov_len:]
        
        opt_len = len(word_opt)
        
        if opt_len < best_len:
            best_word = word_opt
            best_len = opt_len
    
    return "".join(map(str, best_word))
