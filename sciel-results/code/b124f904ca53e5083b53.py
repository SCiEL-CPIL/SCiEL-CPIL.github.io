import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict adjacency as bitmasks
    conflicts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # For efficiency, also store neighbor lists
    neighbor_lists = []
    for i in range(n):
        neighbors = []
        mask = conflicts[i]
        while mask:
            lsb = mask & (-mask)
            j = lsb.bit_length() - 1
            neighbors.append(j)
            mask ^= lsb
        neighbor_lists.append(neighbors)
    
    # Precompute degrees
    degrees = [len(neighbor_lists[i]) for i in range(n)]
    
    # Tighter upper bound: pack maximal independent sets greedily
    # Each packed set uses one "color", so the number of packs is an upper bound on MIS
    def color_bound_tight(vertices_mask):
        """Return an upper bound on the size of MIS in the induced subgraph of vertices_mask."""
        if vertices_mask == 0:
            return 0
        remaining = vertices_mask
        color_count = 0
        while remaining:
            color_count += 1
            # Greedily build a maximal independent set from remaining
            # Pick vertex with highest degree in remaining to reduce remaining faster
            best_v = -1
            best_deg = -1
            temp = remaining
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = bin(conflicts[v] & remaining).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            # Add best_v to current color class, remove it and its neighbors
            remaining &= ~(conflicts[best_v] | (1 << best_v))
        return color_count
    
    # Local search: try to add a vertex or swap to improve the independent set
    def local_search_improve(chosen_mask, available_mask):
        """Try to improve the independent set by adding vertices from available."""
        improved = True
        while improved:
            improved = False
            temp_avail = available_mask
            while temp_avail:
                lsb = temp_avail & (-temp_avail)
                v = lsb.bit_length() - 1
                # Check if v can be added: no conflict with chosen
                if (conflicts[v] & chosen_mask) == 0:
                    chosen_mask |= (1 << v)
                    available_mask &= ~(1 << v)
                    improved = True
                    # Restart scan since adding v may allow more
                    temp_avail = available_mask
                    break
                temp_avail ^= lsb
        return chosen_mask, available_mask
    
    # Branch and bound for Maximum Independent Set
    def branch_and_bound(available_mask, current_size, chosen_mask, best):
        """
        available_mask: bitmask of vertices still available to add
        current_size: size of current independent set
        chosen_mask: bitmask of vertices in current set
        best: mutable list [best_size, best_set_mask]
        """
        # Check time
        if time.time() > deadline:
            return
        
        # Prune: even if we add all remaining vertices, we can't beat best
        remaining_count = bin(available_mask).count('1')
        if current_size + remaining_count <= best[0]:
            return
        
        if available_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
                best[1] = chosen_mask
            return
        
        # Color-based upper bound
        ub = color_bound_tight(available_mask)
        if current_size + ub <= best[0]:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in the remaining subgraph
        best_v = -1
        best_deg = -1
        temp_mask = available_mask
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            v = lsb.bit_length() - 1
            deg_in_avail = bin(conflicts[v] & available_mask).count('1')
            if deg_in_avail > best_deg:
                best_deg = deg_in_avail
                best_v = v
            temp_mask ^= lsb
        
        if best_v == -1:
            return
        
        v_bit = 1 << best_v
        
        # Branch 1: include best_v
        new_available = available_mask & ~(v_bit | conflicts[best_v])
        if time.time() < deadline:
            branch_and_bound(new_available, current_size + 1, chosen_mask | v_bit, best)
        
        # Branch 2: exclude best_v
        new_available2 = available_mask & ~v_bit
        if time.time() < deadline:
            branch_and_bound(new_available2, current_size, chosen_mask, best)
    
    # Track best solution
    best_size = 0
    best_mask = 0  # bitmask of the best independent set
    
    # Initial lower bound from a greedy approach
    def greedy_mis():
        available = (1 << n) - 1
        chosen = 0
        while available:
            # Pick vertex with lowest degree in remaining subgraph
            best_v = -1
            best_deg = float('inf')
            temp = available
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = bin(conflicts[v] & available).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            chosen |= (1 << best_v)
            available &= ~(1 << best_v | conflicts[best_v])
        return chosen
    
    greedy_result = greedy_mis()
    greedy_size = bin(greedy_result).count('1')
    best_size = greedy_size
    best_mask = greedy_result
    
    # Now run branch and bound to try to improve
    all_mask = (1 << n) - 1
    
    # Run B&B
    best = [best_size, best_mask]
    branch_and_bound(all_mask, 0, 0, best)
    best_size = best[0]
    best_mask = best[1]
    
    # Final local search polish: try to add any vertex that doesn't conflict
    # Also try 1-swap moves
    if time.time() < deadline - 2:
        # First, try to add any vertex not in the set that doesn't conflict
        current_set = set()
        for i in range(n):
            if best_mask & (1 << i):
                current_set.add(i)
        
        # Try adding vertices
        for i in range(n):
            if i not in current_set:
                # Check if i conflicts with any vertex in current_set
                conflict = False
                for j in current_set:
                    if (conflicts[i] >> j) & 1:
                        conflict = True
                        break
                if not conflict:
                    current_set.add(i)
                    best_mask |= (1 << i)
        
        # Try 1-swap moves: remove one vertex, try to add one or more
        if time.time() < deadline - 1:
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                current_list = list(current_set)
                for rem in current_list:
                    if time.time() > deadline - 1:
                        break
                    # Remove rem
                    new_set = current_set - {rem}
                    # Try to add any vertex not in new_set that doesn't conflict
                    added = False
                    for add in range(n):
                        if add not in new_set:
                            conflict = False
                            for j in new_set:
                                if (conflicts[add] >> j) & 1:
                                    conflict = True
                                    break
                            if not conflict:
                                new_set.add(add)
                                added = True
                                break
                    if added and len(new_set) > len(current_set):
                        current_set = new_set
                        best_mask = 0
                        for i in current_set:
                            best_mask |= (1 << i)
                        improved = True
                        break
    
    # Extract the result
    result_nodes = []
    for i in range(n):
        if best_mask & (1 << i):
            result_nodes.append(valid_nodes[i])
    
    if not result_nodes:
        result_nodes = [valid_nodes[0]]
    
    return result_nodes
