import time
import random

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

def construct(n):
    if n <= 4:
        return standard_construct(n)
    
    # For n=5,6 standard is likely optimal or very close, and search is small.
    # For n=7, we do the search.
    if n <= 6:
        return standard_construct(n)

    # Generate W_{n-1}
    W_prev = standard_construct(n-1)
    
    # Extract windows of length n-1
    windows = []
    seen = set()
    for i in range(len(W_prev) - (n-1) + 1):
        w = tuple(W_prev[i:i+n-1])
        if len(set(w)) == n-1 and w not in seen:
            seen.add(w)
            windows.append(w)
            
    # Create blocks
    blocks = []
    for w in windows:
        b = list(w) + [n] + list(w)
        blocks.append(tuple(b))
        
    num_blocks = len(blocks)
    
    # Precompute Overlap Matrix M
    # M[i][j] = overlap(blocks[i], blocks[j])
    # Overlap is max k such that blocks[i][-k:] == blocks[j][:k]
    M = [[0]*num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            # Max overlap is len(bi)-1
            max_k = len(bi) - 1
            overlap = 0
            for k in range(max_k, -1, -1):
                if bi[-k:] == bj[:k]:
                    overlap = k
                    break
            M[i][j] = overlap

    # Greedy Search for initial path
    def greedy_search():
        used = [False] * num_blocks
        start_idx = random.randint(0, num_blocks - 1)
        current = start_idx
        used[start_idx] = True
        path = [start_idx]
        
        for _ in range(num_blocks - 1):
            best_next = -1
            best_score = -1
            candidates = []
            for j in range(num_blocks):
                if not used[j]:
                    score = M[current][j]
                    if score > best_score:
                        best_score = score
                        candidates = [j]
                    elif score == best_score:
                        candidates.append(j)
                        
            if not candidates:
                return path
                
            # Pick randomly from best candidates
            best_next = random.choice(candidates)
            used[best_next] = True
            path.append(best_next)
            current = best_next
            
        return path

    # 2-Opt Local Search
    def two_opt(path):
        current_sum = sum(M[path[i]][path[i+1]] for i in range(len(path)-1))
        improved = True
        start_time = time.time()
        deadline = start_time + 100 # 100s budget for search
        
        while improved and time.time() < deadline:
            improved = False
            # Precompute D and prefix sums for fast segment sum
            # D[k] = M[path[k+1]][path[k]] - M[path[k]][path[k+1]]
            D = [0] * (len(path) - 1)
            for k in range(len(path) - 1):
                D[k] = M[path[k+1]][path[k]] - M[path[k]][path[k+1]]
                
            pref_D = [0] * (len(path))
            for k in range(1, len(path)):
                pref_D[k] = pref_D[k-1] + D[k-1]
                
            for i in range(len(path) - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(path)):
                    # Calculate delta
                    # Change in boundary edges
                    left_edge_old = M[path[i-1]][path[i]] if i > 0 else 0
                    left_edge_new = M[path[i-1]][path[j]] if i > 0 else 0
                    
                    right_edge_old = M[path[j]][path[j+1]] if j < len(path) - 1 else 0
                    right_edge_new = M[path[i]][path[j+1]] if j < len(path) - 1 else 0
                    
                    # Change in internal edges
                    # Sum of D[k] for k in [i, j-1]
                    internal_change = pref_D[j] - pref_D[i]
                    
                    delta = (left_edge_new - left_edge_old) + (right_edge_new - right_edge_old) + internal_change
                    
                    if delta > 0:
                        # Accept move
                        current_sum += delta
                        # Reverse path[i:j+1]
                        path[i:j+1] = reversed(path[i:j+1])
                        improved = True
                        # Recalculate D and pref_D for next iteration? 
                        # Or just break and restart loop?
                        # Restarting the outer loop is simpler and ensures D is correct.
                        break
                if improved:
                    break
        return path

    # Run search
    best_path = greedy_search()
    best_sum = sum(M[best_path[i]][best_path[i+1]] for i in range(len(best_path)-1))
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin for construction and return
    
    while time.time() < deadline:
        # Try to improve best_path
        improved_path = two_opt(list(best_path))
        improved_sum = sum(M[improved_path[i]][improved_path[i+1]] for i in range(len(improved_path)-1))
        
        if improved_sum > best_sum:
            best_path = improved_path
            best_sum = improved_sum
        else:
            # Try a new greedy restart
            new_path = greedy_search()
            new_sum = sum(M[new_path[i]][new_path[i+1]] for i in range(len(new_path)-1))
            if new_sum > best_sum:
                best_path = new_path
                best_sum = new_sum
            # If we are stuck, we can just break or continue with more greedy restarts
            # Given the structure, greedy restarts might find better basins.
            # We'll just loop.

    # Construct final word
    word = list(blocks[best_path[0]])
    for i in range(1, len(best_path)):
        prev_block = blocks[best_path[i-1]]
        curr_block = blocks[best_path[i]]
        # Find overlap using M
        k = M[best_path[i-1]][best_path[i]]
        word.extend(curr_block[k:])
        
    return "".join(map(str, word))
