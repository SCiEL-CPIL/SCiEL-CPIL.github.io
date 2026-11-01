import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    word = [1]
    
    for k in range(2, n + 1):
        # Check time
        if time.time() > deadline - 5:
            # Fallback: use standard order
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                if not new:
                    new = block
                else:
                    overlap = get_overlap(new, block)
                    new += block[overlap:]
            word = new
            continue

        # Extract permutations
        perms, seen = [], set()
        for i in range(len(word) - k + 2):
            window = tuple(word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        
        # If N is small, we can be more exhaustive
        # If N is large, use heuristic
        
        # Compute overlap matrix
        # M[i][j] = overlap(perms[i] + [k] + perms[i], perms[j] + [k] + perms[j])
        # We can precompute blocks
        blocks = [list(p) + [k] + list(p) for p in perms]
        
        M = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i != j:
                    M[i][j] = get_overlap(blocks[i], blocks[j])
        
        # Solve TSP
        if N <= 10:
            order = tsp_exact(perms, M, blocks)
        else:
            order = tsp_heuristic(perms, M, blocks, deadline - time.time())
            
        # Construct new word
        new = []
        for idx in order:
            if not new:
                new = blocks[idx]
            else:
                ov = M[order[-1]][idx] # Wait, order is indices
                # Need to map index to index
                pass
        
        # Re-implement construction with indices
        new = []
        prev_idx = None
        for idx in order:
            if not new:
                new = blocks[idx]
            else:
                ov = M[prev_idx][idx]
                new += blocks[idx][ov:]
            prev_idx = idx
            
        word = new
        
    return "".join(map(str, word))

def get_overlap(a, b):
    max_l = min(len(a), len(b))
    # Start from max possible overlap
    # Overlap cannot be len(a) or len(b) if they are distinct blocks
    # Max overlap is min(len(a), len(b)) - 1
    for l in range(max_l - 1, -1, -1):
        if l == 0:
            return 0
        if a[-l:] == b[:l]:
            return l
    return 0

def tsp_exact(perms, M, blocks):
    N = len(perms)
    # DP over subsets
    # State: (mask, last)
    # Too slow for N > 10?
    # N=10, 2^10 * 10 = 10000 states. Fast.
    INF = float('inf')
    dp = [[INF]*N for _ in range(1<<N)]
    parent = [[-1]*N for _ in range(1<<N)]
    
    for i in range(N):
        dp[1<<i][i] = 0 # Cost to start at i is 0? 
        # We want to MAXIMIZE overlap.
        # So we minimize -overlap.
        
    # Actually, let's just do DFS/Backtracking for N <= 10
    best_order = []
    best_score = -1
    
    def dfs(current, visited, current_score):
        nonlocal best_order, best_score
        if len(current) == N:
            if current_score > best_score:
                best_score = current_score
                best_order = current[:]
            return
        
        last = current[-1]
        for next_node in range(N):
            if not (visited & (1<<next_node)):
                new_score = current_score + M[last][next_node]
                dfs(current + [next_node], visited | (1<<next_node), new_score)
                
    for i in range(N):
        dfs([i], 1<<i, 0)
        
    return best_order

def tsp_heuristic(perms, M, blocks, time_limit):
    N = len(perms)
    start_time = time.time()
    
    # Greedy
    order = [0]
    visited = {0}
    while len(order) < N:
        last = order[-1]
        best_next = -1
        best_ov = -1
        for j in range(N):
            if j not in visited:
                if M[last][j] > best_ov:
                    best_ov = M[last][j]
                    best_next = j
        order.append(best_next)
        visited.add(best_next)
        
    # 2-Opt
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(1, N-1):
            for j in range(i+1, N):
                # Check if reversing order[i:j+1] improves
                # This is complex for TSP with directed weights (overlap is not symmetric)
                # M[i][j] != M[j][i] generally.
                # So simple 2-opt (reverse segment) is not valid for asymmetric TSP.
                # We need Or-Opt or similar.
                # Let's just do simple swap of adjacent elements?
                pass
                
        # Simple Local Search: Swap adjacent
        for i in range(N-1):
            # Try swapping i and i+1
            if i == 0: continue # Start is fixed? No, we can rotate.
            # Actually, we can rotate the cycle.
            # Let's just try swapping any two elements i, j
            # And check if it improves total score.
            # Score = sum M[order[k]][order[k+1]]
            
            # Efficient check:
            # Remove i, j. Connect neighbors.
            # This is O(1) per swap if we maintain prefix sums?
            # For now, just calculate delta.
            
            pass
            
    return order
