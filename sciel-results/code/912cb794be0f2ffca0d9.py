import time
import math
from itertools import permutations
import random

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    Uses a TSP heuristic on the cycle structure of permutations.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231423124312134213241321431214231412312"
    
    # For n >= 5, use the TSP heuristic on cycles
    symbols = list(range(1, n + 1))
    all_perms = list(permutations(symbols))
    perm_set = set(all_perms)
    
    # Identify cycles (based on left shift)
    # shift(p) = (p[1], p[2], ..., p[n-1], p[0])
    visited = set()
    cycles = []
    
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
        
    K = len(cycles)
    
    # Precompute overlap costs between cycles
    # W[i][j] = min cost to go from any node in cycle i to any node in cycle j
    # Cost = n - overlap
    # Also store the best u and v for reconstruction
    
    def get_overlap(p, q):
        # Max k such that p[n-k:] == q[:k]
        # p and q are tuples of length n
        max_k = n - 1 # Max possible overlap for distinct cycles is n-2, but we check up to n-1
        # Actually, if overlap is n-1, they are in the same cycle.
        for k in range(max_k, -1, -1):
            if k == 0:
                return 0
            if p[n-k:] == q[:k]:
                return k
        return 0

    W = [[0] * K for _ in range(K)]
    best_u = [[None] * K for _ in range(K)]
    best_v = [[None] * K for _ in range(K)]
    
    # Optimization: Precompute suffixes/prefixes?
    # For now, direct calculation.
    
    for i in range(K):
        for j in range(K):
            if i == j:
                W[i][j] = n - 1 # Internal cost
                best_u[i][j] = cycles[i][0]
                best_v[i][j] = cycles[i][0]
                continue
            
            min_cost = n
            best_ui = None
            best_vj = None
            
            for u in cycles[i]:
                # Quick check: if min_cost is 2, we can't do better (for distinct cycles)
                if min_cost == 2:
                    break
                for v in cycles[j]:
                    ov = get_overlap(u, v)
                    cost = n - ov
                    if cost < min_cost:
                        min_cost = cost
                        best_ui = u
                        best_vj = v
                        if min_cost == 2:
                            break
            
            W[i][j] = min_cost
            best_u[i][j] = best_ui
            best_v[i][j] = best_vj
            
    if time.time() > deadline - 20:
        # Fallback if matrix calculation is too slow
        return _classic_construct(n)

    # Solve TSP on K nodes with weights W
    # Use Nearest Neighbor + 2-Opt
    
    # Start at 0
    tour = [0]
    unvisited = set(range(1, K))
    
    while unvisited:
        curr = tour[-1]
        # Find nearest neighbor
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
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(K - 1):
            if time.time() > deadline - 5:
                break
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
                    
    # Reconstruct the superpermutation
    word = []
    
    # Pick starting node for first cycle
    c_idx = tour[0]
    u = best_u[c_idx][c_idx] # Just pick first
    word.extend(u)
    current_perm = u
    
    # Traverse first cycle
    # We entered at u. We must exit at the node that is best for the next jump?
    # No, we traverse the cycle. The exit node is determined by where we stop.
    # We want to exit at a node x such that the jump to the next cycle is minimal.
    # But we don't know the next cycle's entry node yet?
    # We do: tour[1] is the next cycle.
    # We want to minimize Cost(exit_node, entry_node_next).
    # We can choose exit_node to be any node in the cycle.
    # But we must visit ALL nodes in the cycle.
    # If we enter at u, we visit u, shift(u), ..., prev(u).
    # The exit node is prev(u).
    # So the exit node is fixed by the entry node!
    # Exit(u) = prev(u) in the cycle.
    
    for idx in range(len(tour)):
        c_idx = tour[idx]
        cycle = cycles[c_idx]
        
        if idx == 0:
            # Already added u
            # Traverse rest of cycle
            for k in range(1, n):
                current_perm = (current_perm[1:] + current_perm[:1])
                word.append(current_perm[-1])
            # Now current_perm is exit node
        else:
            # Jump to next cycle
            next_c_idx = tour[idx] # This is current cycle in loop, but we need to jump TO it
            # Wait, the loop structure is:
            # We are at end of cycle tour[idx-1].
            # We need to jump to tour[idx].
            
            # Determine entry node for current cycle tour[idx]
            # We want to minimize Cost(current_perm, entry_node)
            # We have precomputed best_u[prev_idx][idx] and best_v[prev_idx][idx]
            # But current_perm is the exit of the previous cycle.
            # The previous cycle was tour[idx-1].
            # The entry node chosen in best_v[tour[idx-1]][idx] was optimal for the exit of tour[idx-1].
            # But is current_perm actually that exit?
            # Yes, because we traversed the full cycle.
            
            prev_idx = tour[idx-1]
            entry_u = best_v[prev_idx][idx] # This is the 'v' from the pair (u in prev, v in curr)
            
            # Jump
            ov = get_overlap(current_perm, entry_u)
            to_append = entry_u[n - ov:]
            word.extend(to_append)
            current_perm = entry_u
            
            # Traverse this cycle
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
