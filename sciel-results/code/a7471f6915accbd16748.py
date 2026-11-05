import time
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 110.0
    
    # Base cases
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231423124312134213241321431214231412312"
    if n == 5:
        # Classic is optimal for n=5 (153)
        return _classic_construct(5)
    if n == 6:
        # Classic is optimal for n=6 (873)
        return _classic_construct(6)
    
    if n > 7:
        return _classic_construct(n)

    # For n=7, use the optimized TSP + DP approach
    symbols = list(range(1, n + 1))
    all_perms = list(permutations(symbols))
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Identify cycles (based on left shift)
    # shift(p) = (p[1], ..., p[n-1], p[0])
    visited = set()
    cycles = []
    perm_to_cycle = {}
    perm_to_pos_in_cycle = {}
    
    for p in all_perms:
        if p in visited:
            continue
        cycle = []
        curr = p
        while curr not in visited:
            visited.add(curr)
            cycle.append(curr)
            curr = (curr[1:] + curr[:1])
        cycles.append(cycle)
        
        for pos, c_perm in enumerate(cycle):
            perm_to_cycle[c_perm] = len(cycles) - 1
            perm_to_pos_in_cycle[c_perm] = pos

    K = len(cycles)
    
    # Precompute overlap costs
    # W[i][j] = min cost to go from any node in cycle i to any node in cycle j
    # Cost = n - overlap
    
    def get_overlap(p, q):
        # Max k such that p[n-k:] == q[:k]
        for k in range(n, -1, -1):
            if k == 0:
                return 0
            if p[n-k:] == q[:k]:
                return k
        return 0

    W = [[n] * K for _ in range(K)]
    
    for i in range(K):
        for j in range(K):
            if i == j:
                W[i][j] = 1 # Intra-cycle cost is 1
                continue
            min_cost = n
            for u in cycles[i]:
                for v in cycles[j]:
                    ov = get_overlap(u, v)
                    cost = n - ov
                    if cost < min_cost:
                        min_cost = cost
                        if min_cost == 2: # Can't get lower than 2 for distinct cycles
                            break
                if min_cost == 2:
                    break
            W[i][j] = min_cost
            
    if time.time() > deadline - 10:
        return _classic_construct(n)

    # Solve TSP on K nodes
    tour = [0]
    unvisited = set(range(1, K))
    
    while unvisited:
        curr = tour[-1]
        best_node = None
        best_dist = float('inf')
        for node in unvisited:
            if W[curr][node] < best_dist:
                best_dist = W[curr][node]
                best_node = node
        tour.append(best_node)
        unvisited.remove(best_node)
        
    # 2-Opt optimization
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for i in range(K - 1):
            for j in range(i + 1, K):
                if j == i + 1:
                    continue
                
                node_a = tour[i]
                node_b = tour[i + 1]
                node_c = tour[j]
                node_d = tour[(j + 1) % K]
                
                old_cost = W[node_a][node_b] + W[node_c][node_d]
                new_cost = W[node_a][node_d] + W[node_c][node_b]
                
                if new_cost < old_cost:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
    
    # DP to find optimal entry nodes
    # dp[i][u_idx] = min cost to reach cycle i and enter at u_idx (index in cycle)
    # prev_entry[i][u_idx] = the entry index in cycle i-1 that led to this
    
    # Note: The "cost" we are minimizing in DP is the sum of jump costs.
    # Total length = n + (K-1)*(n-1) + sum(jump_costs)
    # So minimizing sum(jump_costs) is equivalent.
    
    # Initialize DP for first cycle
    # We can enter the first cycle at any node.
    # The "previous exit" is virtual. We don't pay a jump cost for the first entry.
    # So dp[0][u] = 0 for all u in C_0.
    
    dp = [[0] * n for _ in range(K)]
    prev_entry = [[-1] * n for _ in range(K)]
    
    # For i > 0:
    # dp[i][u] = min_{v in C_{i-1}} ( dp[i-1][v] + n - overlap(prev(v), u) )
    # where prev(v) is the exit permutation of cycle i-1 if we entered at v.
    
    # Precompute exit permutations for each cycle and entry index
    # exit_perm[cycle_idx][entry_idx] = the permutation we are at after traversing the cycle
    # If we enter at u (index k), we visit u, s(u), ..., s^{n-1}(u).
    # The last one is s^{n-1}(u).
    
    exit_perms = [[None]*n for _ in range(K)]
    for i in range(K):
        cycle = cycles[i]
        for k in range(n):
            u = cycle[k]
            # s^{n-1}(u)
            # Since cycle is u, s(u), ..., s^{n-1}(u)
            # The element at index (k + n - 1) % n in the cycle list?
            # Wait.
            # cycle[0] = u
            # cycle[1] = s(u)
            # ...
            # cycle[n-1] = s^{n-1}(u)
            # So if we enter at cycle[k], the sequence is cycle[k], cycle[k+1], ..., cycle[k-1] (mod n).
            # The last one is cycle[k-1] (mod n).
            # Let's verify.
            # Enter at k.
            # Visit k.
            # Visit k+1.
            # ...
            # Visit k+n-1.
            # Index (k+n-1) % n = k-1 (mod n).
            # So exit is cycle[(k-1) % n].
            
            exit_perms[i][k] = cycle[(k - 1) % n]
            
    for i in range(1, K):
        curr_cycle = cycles[i]
        prev_cycle = cycles[i-1]
        
        for u_idx, u in enumerate(curr_cycle):
            min_cost = float('inf')
            best_v_idx = -1
            for v_idx, v in enumerate(prev_cycle):
                exit_v = exit_perms[i-1][v_idx]
                ov = get_overlap(exit_v, u)
                cost = n - ov
                total = dp[i-1][v_idx] + cost
                if total < min_cost:
                    min_cost = total
                    best_v_idx = v_idx
            dp[i][u_idx] = min_cost
            prev_entry[i][u_idx] = best_v_idx
            
    if time.time() > deadline - 10:
        return _classic_construct(n)

    # Find best entry for last cycle
    last_cycle_idx = K - 1
    best_last_entry = 0
    min_last_cost = dp[last_cycle_idx][0]
    for k in range(1, n):
        if dp[last_cycle_idx][k] < min_last_cost:
            min_last_cost = dp[last_cycle_idx][k]
            best_last_entry = k
            
    # Reconstruct entries
    entries = [0] * K
    entries[last_cycle_idx] = best_last_entry
    for i in range(K - 1, 0, -1):
        entries[i-1] = prev_entry[i][entries[i]]
        
    # Reconstruct word
    word = []
    current_perm = None
    
    for i, c_idx in enumerate(tour):
        cycle = cycles[c_idx]
        u_idx = entries[i]
        u = cycle[u_idx]
        
        if i == 0:
            word.extend(u)
            current_perm = u
        else:
            # Jump from current_perm (which is exit of prev cycle) to u
            ov = get_overlap(current_perm, u)
            to_append = u[n - ov:]
            word.extend(to_append)
            current_perm = u
            
        # Traverse rest of cycle
        for k in range(1, n):
            current_perm = (current_perm[1:] + current_perm[:1])
            word.append(current_perm[-1])
            
    return "".join(map(str, word))

def _classic_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        word = new_word
    return "".join(map(str, word))
