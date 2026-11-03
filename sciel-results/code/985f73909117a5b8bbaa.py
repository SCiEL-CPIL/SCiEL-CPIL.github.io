import numpy as np
import time
import random
import heapq
import sys

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        return [list(range(n)) for _ in range(1)]

    M = 3**n
    ball_size = 1 + 2 * n
    powers = [3**i for i in range(n)]

    # Precompute balls
    balls = [None] * M
    for i in range(M):
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        ball = [i]
        for pos in range(n):
            new_d = (digits[pos] + 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
            new_d = (digits[pos] - 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
        balls[i] = ball

    # Build adjacency for Flow Network
    # Nodes:
    # 0: Source
    # 1..M: Codewords
    # M+1..2M: Points
    # 2M+1: Sink
    N_nodes = 2 * M + 2
    SRC = 0
    SNK = 2 * M + 1
    C_START = 1
    P_START = 1 + M

    # Graph representation for Push-Relabel
    # graph[u] = list of (v, capacity, reverse_index)
    graph = [[] for _ in range(N_nodes)]
    
    def add_edge(u, v, cap):
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])

    # S -> C (cap 1)
    for c in range(M):
        add_edge(SRC, C_START + c, 1)
        
    # C -> P (cap 1) if c covers p
    # Note: For Max Matching, cap 1 is correct.
    # For Set Cover via Flow relaxation, we might want different caps, 
    # but Max Matching is a standard output of this graph.
    for c in range(M):
        for p in balls[c]:
            add_edge(C_START + c, P_START + p, 1)
            
    # P -> T (cap 1)
    for p in range(M):
        add_edge(P_START + p, SNK, 1)

    # Push-Relabel Implementation
    def push_relabel():
        height = [0] * N_nodes
        excess = [0] * N_nodes
        # Initialize preflow
        height[SRC] = N_nodes
        for u, v, rev in graph[SRC]:
            if v[1] > 0: # Actually v is list [v, cap, rev]
                pass # Just iterate properly
        
        # Correct Initialization
        for edge in graph[SRC]:
            v = edge[0]
            cap = edge[1]
            if cap > 0:
                excess[SRC] -= cap
                excess[v] += cap
                edge[1] -= cap
                graph[v][edge[2]][1] += cap
        
        active = [False] * N_nodes
        queue = []
        for v in range(1, N_nodes - 1):
            if excess[v] > 0:
                active[v] = True
                queue.append(v)
        
        def push(u, edge_idx):
            v = graph[u][edge_idx][0]
            cap = graph[u][edge_idx][1]
            if cap > 0 and height[u] == height[v] + 1:
                amt = min(excess[u], cap)
                excess[u] -= amt
                excess[v] += amt
                graph[u][edge_idx][1] -= amt
                graph[v][graph[u][edge_idx][2]][1] += amt
                if not active[v] and v != SRC and v != SNK and excess[v] > 0:
                    active[v] = True
                    queue.append(v)
        
        def relabel(u):
            min_h = float('inf')
            for edge in graph[u]:
                v = edge[0]
                cap = edge[1]
                if cap > 0:
                    min_h = min(min_h, height[v])
            if min_h < float('inf'):
                height[u] = min_h + 1
        
        while queue:
            u = queue.pop(0)
            active[u] = False
            old_height = height[u]
            
            # Discharge u
            while excess[u] > 0:
                pushed = False
                for i, edge in enumerate(graph[u]):
                    v = edge[0]
                    cap = edge[1]
                    if cap > 0 and height[u] == height[v] + 1:
                        push(u, i)
                        pushed = True
                        if excess[u] == 0:
                            break
                if not pushed:
                    relabel(u)
                    # If height changed, restart scan
                    if height[u] >= N_nodes:
                        break
            
            if excess[u] > 0 and not active[u] and u != SRC and u != SNK:
                active[u] = True
                queue.append(u)
        
        return excess, height

    # Run Push-Relabel to get Max Matching
    excess, height = push_relabel()
    
    # Extract matched codewords
    # A codeword c is matched if there is flow on S->c
    # Flow on S->c is 1 - graph[SRC][c-1][1]
    matched_codewords = set()
    for c in range(M):
        # Edge S -> C_START+c
        # Find index
        # graph[SRC] contains edges to C_START+c
        # We added them in order c=0..M-1
        # So graph[SRC][c] is the edge to C_START+c
        cap_remaining = graph[SRC][c][1]
        if cap_remaining == 0:
            matched_codewords.add(c)

    # Initialize code with matched codewords
    code = set(matched_codewords)
    
    # Calculate uncovered points
    covered = np.zeros(M, dtype=bool)
    for c in code:
        for p in balls[c]:
            covered[p] = True
            
    uncovered_points = [p for p in range(M) if not covered[p]]
    
    # Greedy Set Cover to cover remaining points
    # Calculate gains for all codewords not in code
    # Gain = number of uncovered points covered
    gains = np.zeros(M, dtype=np.int32)
    for c in range(M):
        if c not in code:
            g = 0
            for p in balls[c]:
                if not covered[p]:
                    g += 1
            gains[c] = g
            
    heap = [(-gains[c], c) for c in range(M) if gains[c] > 0]
    heapq.heapify(heap)
    
    while uncovered_points:
        valid_c = None
        while heap:
            neg_g, c = heapq.heappop(heap)
            if c in code:
                continue
            if -neg_g != gains[c]:
                continue
            if gains[c] <= 0:
                continue
            valid_c = c
            break
            
        if valid_c is None:
            break
            
        code.add(valid_c)
        
        # Update covered and gains
        for p in balls[valid_c]:
            if not covered[p]:
                covered[p] = True
                uncovered_points.remove(p) # This is O(N), need better way
                # For efficiency, we should use a list or set for uncovered
                # Let's switch uncovered to a set for removal
                # Actually, let's just recalculate or use a boolean array and count
                # For 6561 points, O(N) removal is okay for ~400 iterations
                for c_prime in range(M):
                    if c_prime not in code and c_prime != valid_c:
                        if p in balls[c_prime]:
                            gains[c_prime] -= 1
                            if gains[c_prime] > 0:
                                heapq.heappush(heap, (-gains[c_prime], c_prime))
                            
    # Optimization: Use a set for uncovered points to make removal O(1)
    # Let's restart the greedy part with better data structures if needed, 
    # but given the small size, the above might be slow due to O(N) in inner loop.
    # Let's refine the greedy part.
    
    # Re-implement Greedy with sets
    code = set(matched_codewords)
    covered = np.zeros(M, dtype=bool)
    for c in code:
        for p in balls[c]:
            covered[p] = True
            
    uncovered_set = set(p for p in range(M) if not covered[p])
    
    # Precompute coverers for faster gain update? 
    # coverers[p] = list of c covering p
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)
            
    gains = np.zeros(M, dtype=np.int32)
    for c in range(M):
        if c not in code:
            g = 0
            for p in balls[c]:
                if p in uncovered_set:
                    g += 1
            gains[c] = g
            
    heap = [(-gains[c], c) for c in range(M) if gains[c] > 0]
    heapq.heapify(heap)
    
    while uncovered_set:
        valid_c = None
        while heap:
            neg_g, c = heapq.heappop(heap)
            if c in code:
                continue
            if -neg_g != gains[c]:
                continue
            if gains[c] <= 0:
                continue
            valid_c = c
            break
            
        if valid_c is None:
            break
            
        code.add(valid_c)
        
        for p in balls[valid_c]:
            if p in uncovered_set:
                uncovered_set.remove(p)
                for c_prime in coverers[p]:
                    if c_prime not in code:
                        gains[c_prime] -= 1
                        if gains[c_prime] > 0:
                            heapq.heappush(heap, (-gains[c_prime], c_prime))

    # Local Search to improve
    start_time = time.time()
    time_limit = 115
    
    # Precompute count for redundancy check
    count = np.zeros(M, dtype=np.int32)
    for c in code:
        for p in balls[c]:
            count[p] += 1
            
    def is_redundant(c):
        for p in balls[c]:
            if count[p] == 1:
                return False
        return True
        
    def remove_c(c):
        code.remove(c)
        for p in balls[c]:
            count[p] -= 1
            
    def add_c(c):
        code.add(c)
        for p in balls[c]:
            count[p] += 1
            
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Remove redundant
        to_remove = []
        for c in list(code):
            if is_redundant(c):
                to_remove.append(c)
        if to_remove:
            for c in to_remove:
                remove_c(c)
            improved = True
            continue
            
        # 1-Swap
        code_list = list(code)
        random.shuffle(code_list)
        swapped = False
        for c in code_list:
            if (time.time() - start_time) >= time_limit:
                break
            unique_points = [p for p in balls[c] if count[p] == 1]
            if not unique_points:
                remove_c(c)
                improved = True
                swapped = True
                break
            candidates = set(coverers[unique_points[0]])
            for p in unique_points[1:]:
                candidates.intersection_update(coverers[p])
            candidates -= code
            if candidates:
                c_new = random.choice(list(candidates))
                remove_c(c)
                add_c(c_new)
                improved = True
                swapped = True
                break
        if swapped:
            continue
            
        # 2-Swap
        if (time.time() - start_time) < time_limit:
            if len(code) >= 2:
                c1, c2 = random.sample(list(code), 2)
                U1 = [p for p in balls[c1] if count[p] == 1]
                U2 = [p for p in balls[c2] if count[p] == 1]
                
                if U1 and U2:
                    all_unique = set(U1) | set(U2)
                    first_p = next(iter(all_unique))
                    cands = set(coverers[first_p])
                    for p in all_unique:
                        cands.intersection_update(coverers[p])
                    cands -= code
                    cands.discard(c1)
                    cands.discard(c2)
                    if cands:
                        c_new = random.choice(list(cands))
                        remove_c(c1)
                        remove_c(c2)
                        add_c(c_new)
                        improved = True
                        continue
                        
    return [list(map(int, [(c // (3**i)) % 3 for i in range(n)])) for c in code]
