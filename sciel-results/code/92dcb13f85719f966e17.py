import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Classic recursive construction
    def classic_construct(n):
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
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    if n != 7:
        word = classic_construct(n)
        return "".join(map(str, word))
    
    # Special optimization for n=7
    # Step 1: Build SP(6) and extract permutations of 1..6 in order
    word_6 = [1]
    for m in range(2, 7):
        perms, seen = [], set()
        for i in range(len(word_6) - m + 2):
            window = tuple(word_6[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
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
        word_6 = new
    
    # Extract unique 6-permutations in order of appearance
    perms_6, seen = [], set()
    for i in range(len(word_6) - 5):
        window = tuple(word_6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
    
    num_perms = len(perms_6)
    
    # Step 2: Precompute overlap matrix
    # M[i][j] = longest suffix of perms_6[i] that is a prefix of perms_6[j]
    M = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            # Max possible overlap is 6
            for k in range(6, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break
    
    def calc_overlap_sum(order):
        total = 0
        for i in range(len(order) - 1):
            total += M[order[i]][order[i + 1]]
        return total
    
    # Step 3: Bridge-and-tunnel heuristic
    # Build MST using overlap as weight (maximize overlap = minimize cost)
    # Then do DFS traversal to create initial ordering
    
    # Use Prim's algorithm for maximum spanning tree
    visited = [False] * num_perms
    mst_edges = []  # (u, v, weight)
    
    # Start from node 0
    visited[0] = True
    # Priority queue: (negative_weight, u, v) - we want max weight
    import heapq
    heap = []
    for v in range(1, num_perms):
        heapq.heappush(heap, (-M[0][v], 0, v))
    
    while len(mst_edges) < num_perms - 1:
        neg_w, u, v = heapq.heappop(heap)
        if visited[v]:
            continue
        visited[v] = True
        mst_edges.append((u, v, -neg_w))
        for w in range(num_perms):
            if not visited[w]:
                heapq.heappush(heap, (-M[v][w], v, w))
    
    # Build adjacency list for MST
    adj = [[] for _ in range(num_perms)]
    for u, v, w in mst_edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    
    # DFS traversal of MST to get initial ordering
    # This creates a Hamiltonian path that follows the tree structure
    def dfs_order(start, adj, n):
        order = []
        def _dfs(node, parent):
            order.append(node)
            # Sort neighbors by overlap (descending) for better transitions
            neighbors = sorted(adj[node], key=lambda x: -x[1])
            for nb, w in neighbors:
                if nb != parent:
                    _dfs(nb, node)
        _dfs(start, -1)
        return order
    
    mst_order = dfs_order(0, adj, num_perms)
    mst_score = calc_overlap_sum(mst_order)
    
    # Step 4: Local search starting from MST order
    random.seed(42)
    
    order = mst_order[:]
    current_score = mst_score
    best_order = order[:]
    best_score = current_score
    
    # Simulated annealing with multiple restarts if time permits
    T = 50.0  # initial temperature (lower than before since we start from better solution)
    cooling_rate = 0.9998
    
    # Track time for early stopping
    iterations = 0
    while time.time() < deadline - 10:  # Leave 10s buffer
        # Pick a random move: swap two elements
        i = random.randint(0, num_perms - 2)
        j = i + 1  # adjacent swap for efficiency
        
        new_order = order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        # Calculate delta in score
        old_score = 0
        new_score = 0
        
        # Affected edges: (i-1,i), (i,i+1), (i+1,i+2)
        if i > 0:
            old_score += M[order[i-1]][order[i]]
            new_score += M[order[i-1]][order[i+1]]
        old_score += M[order[i]][order[i+1]]
        new_score += M[order[i+1]][order[i]]
        if i + 2 < num_perms:
            old_score += M[order[i+1]][order[i+2]]
            new_score += M[order[i]][order[i+2]]
        
        delta = new_score - old_score
        
        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
            order = new_order
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_order = order[:]
        
        T *= cooling_rate
        if T < 0.01:
            # Restart with fresh temperature
            T = 20.0
            order = best_order[:]
            current_score = best_score
        
        iterations += 1
    
    # Step 5: Final polish - 2-opt style improvement
    # Try reversing segments to improve overlap
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(num_perms - 1):
            for j in range(i + 2, num_perms):
                # Try reversing segment [i, j]
                old_score = 0
                new_score = 0
                
                # Edges affected: (i-1,i), (i,j+1 or end), (j,j+1 or end)
                if i > 0:
                    old_score += M[best_order[i-1]][best_order[i]]
                    new_score += M[best_order[i-1]][best_order[j]]
                if j + 1 < num_perms:
                    old_score += M[best_order[j]][best_order[j+1]]
                    new_score += M[best_order[i]][best_order[j+1]]
                # Internal edges within [i,j] reverse direction
                for k in range(i, j):
                    old_score += M[best_order[k]][best_order[k+1]]
                    new_score += M[best_order[k+1]][best_order[k]]
                
                delta = new_score - old_score
                if delta > 0:
                    # Apply reversal
                    best_order[i:j+1] = best_order[i:j+1][::-1]
                    best_score += delta
                    improved = True
    
    # Step 6: Construct the final word from best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            # Find overlap with current word
            ov = 0
            max_ov = min(len(word), len(block))
            for k in range(max_ov, -1, -1):
                if word[-k:] == block[:k]:
                    ov = k
                    break
            word += block[ov:]
    
    return "".join(map(str, word))
