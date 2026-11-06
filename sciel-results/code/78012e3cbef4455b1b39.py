import random
import time
import numpy as np

def construct():
    d = 7
    n = 2 ** d
    num_edges = n * d // 2  # 448
    
    # 1. Precompute edges and their indices
    # Edge index mapping: (u, v) -> idx
    # We'll just generate edges and assign indices 0..447
    edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    idx = 0
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                idx_to_edge[idx] = (x, y)
                idx += 1
                
    # 2. Precompute C4s
    # A C4 is defined by two distinct dimensions i, j and a base vertex x
    # such that bits i and j are 0 in x (or we just iterate all x and mask out duplicates?)
    # Standard way: Iterate all x in 0..127. For each pair of dims (i, j) with i < j:
    # The square is x, x^i, x^j, x^i^j.
    # To avoid duplicates, we can require that x has 0 in bit i and 0 in bit j.
    # Then the square is unique.
    c4s = []
    edge_to_c4s = [[] for _ in range(num_edges)]
    
    for x in range(n):
        for i in range(d):
            if x & (1 << i):
                continue
            for j in range(i + 1, d):
                if x & (1 << j):
                    continue
                
                # Vertices of the square
                v0 = x
                v1 = x ^ (1 << i)
                v2 = x ^ (1 << j)
                v3 = x ^ (1 << i) ^ (1 << j)
                
                # Edges of the square
                e1 = (min(v0, v1), max(v0, v1))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v3, v2), max(v3, v2))
                e4 = (min(v2, v0), max(v2, v0))
                
                c4_edges = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                c4s.append(c4_edges)
                
                for e_idx in c4_edges:
                    edge_to_c4s[e_idx].append(len(c4s) - 1)
    
    num_c4s = len(c4s)
    
    # Data structures for fast updates
    selected = np.zeros(num_edges, dtype=np.bool_)
    c4_counts = np.zeros(num_c4s, dtype=np.int8)
    
    def add_edge(e_idx):
        selected[e_idx] = True
        for c4_idx in edge_to_c4s[e_idx]:
            c4_counts[c4_idx] += 1
            
    def remove_edge(e_idx):
        selected[e_idx] = False
        for c4_idx in edge_to_c4s[e_idx]:
            c4_counts[c4_idx] -= 1
            
    def can_add(e_idx):
        # Check if adding e_idx would create a C4 with 4 edges
        # i.e., if any c4 containing e_idx already has 3 edges selected
        for c4_idx in edge_to_c4s[e_idx]:
            if c4_counts[c4_idx] >= 3:
                return False
        return True

    best_count = 0
    best_selected = selected.copy()
    
    deadline = time.time() + 110
    
    # Greedy initialization
    def greedy_init():
        nonlocal selected, c4_counts, best_count, best_selected
        selected[:] = False
        c4_counts[:] = 0
        count = 0
        order = list(range(num_edges))
        random.shuffle(order)
        for e in order:
            if can_add(e):
                add_edge(e)
                count += 1
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
        return count

    # Local Search
    def local_search(current_count):
        nonlocal selected, c4_counts, best_count, best_selected
        count = current_count
        no_improve_steps = 0
        max_no_improve = 5000
        last_best_update = time.time()
        
        while time.time() < deadline - 5: # Leave 5s for final cleanup
            # Try Add
            if random.random() < 0.5:
                e = random.randint(0, num_edges - 1)
                if not selected[e]:
                    if can_add(e):
                        add_edge(e)
                        count += 1
                        no_improve_steps = 0
                        if count > best_count:
                            best_count = count
                            best_selected = selected.copy()
                            last_best_update = time.time()
                    else:
                        no_improve_steps += 1
            else:
                # Try Swap: Remove e1, Add e2
                # Pick random selected e1
                # To pick a random selected edge efficiently, we can just pick random e1 and check if selected.
                # Or maintain a list of selected edges? 
                # Given 448 edges, picking random e1 and checking is O(1) on average if density is ~0.7.
                e1 = random.randint(0, num_edges - 1)
                if selected[e1]:
                    e2 = random.randint(0, num_edges - 1)
                    if not selected[e2]:
                        # Try swap
                        remove_edge(e1)
                        if can_add(e2):
                            add_edge(e2)
                            # Count stays same, but configuration changes
                            no_improve_steps += 1
                        else:
                            # Revert
                            add_edge(e1)
                            no_improve_steps += 1
                            
            # Occasional Kick: Remove some random edges and greedily re-add
            if no_improve_steps > max_no_improve or (time.time() - last_best_update > 5 and random.random() < 0.1):
                # Kick
                kick_size = random.randint(5, 20)
                to_remove = []
                attempts = 0
                while len(to_remove) < kick_size and attempts < 100:
                    e = random.randint(0, num_edges - 1)
                    if selected[e] and e not in to_remove:
                        to_remove.append(e)
                    attempts += 1
                
                for e in to_remove:
                    remove_edge(e)
                    count -= 1
                
                # Greedy re-add
                order = list(range(num_edges))
                random.shuffle(order)
                for e in order:
                    if not selected[e] and can_add(e):
                        add_edge(e)
                        count += 1
                        
                no_improve_steps = 0
                if count > best_count:
                    best_count = count
                    best_selected = selected.copy()
                    last_best_update = time.time()

    # Main Loop
    # Run multiple restarts
    while time.time() < deadline - 10:
        # Greedy init
        curr_count = greedy_init()
        
        # Local search
        local_search(curr_count)
        
        # If we are close to the end, just keep polishing the best found so far?
        # Or just let the loop continue. The local_search will exit when deadline - 5.
        # We need to ensure we don't start a new greedy init if time is low.
        if time.time() > deadline - 15:
            break

    # Final polish: Try to add any remaining edges to best_selected?
    # The local search already tries to add. But let's do one final greedy pass on the best solution.
    # Actually, best_selected is a snapshot. We should restore it and try to improve it one last time.
    selected[:] = best_selected
    c4_counts[:] = 0
    for i in range(num_edges):
        if selected[i]:
            add_edge(i) # This is O(N * 6) which is fine.
            
    # Final greedy pass
    order = list(range(num_edges))
    random.shuffle(order)
    for e in order:
        if not selected[e]:
            if can_add(e):
                add_edge(e)
                best_count += 1
                best_selected = selected.copy()

    # Convert back to list of [u, v]
    result = []
    for i in range(num_edges):
        if best_selected[i]:
            result.append(list(idx_to_edge[i]))
            
    return result
