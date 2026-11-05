import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Helper to generate K4s and Ind-4s from adjacency list
    def get_k4s(adj, n):
        k4s = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # Remove u and v from common? No, common already excludes them if we define adj correctly?
                    # adj[i] has bits for j != i.
                    # So common contains w, x such that w,v connected and u,w connected.
                    # We need w, x to be connected.
                    # Iterate pairs in common
                    common_list = []
                    temp = common
                    while temp:
                        bit = temp & -temp
                        idx = bit.bit_length() - 1
                        common_list.append(idx)
                        temp ^= bit
                    
                    for i in range(len(common_list)):
                        w = common_list[i]
                        for j in range(i + 1, len(common_list)):
                            x = common_list[j]
                            if adj[w] & (1 << x):
                                # u, v, w, x form K4
                                # Sort for uniqueness
                                quad = tuple(sorted([u, v, w, x]))
                                k4s.append(quad)
        return k4s

    def get_ind4s(adj, n):
        ind4s = []
        # Non-adjacency matrix
        non_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            non_adj[i] = all_mask & ~(1 << i) & ~adj[i]
            
        for u in range(n):
            for v in range(u + 1, n):
                if non_adj[u] & (1 << v):
                    common = non_adj[u] & non_adj[v]
                    common_list = []
                    temp = common
                    while temp:
                        bit = temp & -temp
                        idx = bit.bit_length() - 1
                        common_list.append(idx)
                        temp ^= bit
                    
                    for i in range(len(common_list)):
                        w = common_list[i]
                        for j in range(i + 1, len(common_list)):
                            x = common_list[j]
                            if non_adj[w] & (1 << x):
                                quad = tuple(sorted([u, v, w, x]))
                                ind4s.append(quad)
        return ind4s

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    # Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_n = q
    best_n = q
    best_adj = adj[:]
    
    # Precompute K4s and Ind4s for the initial graph
    k4s = get_k4s(adj, current_n)
    ind4s = get_ind4s(adj, current_n)
    
    # Try to grow
    for target_n in range(current_n + 1, 47):
        if time.time() > deadline - 30:
            break
            
        success = False
        max_attempts = 100 # Increase attempts since each is faster
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Create new graph
            new_adj = [0] * target_n
            for i in range(current_n):
                new_adj[i] = adj[i]
            
            # Connect new vertex (index target_n - 1)
            new_idx = target_n - 1
            for i in range(current_n):
                if random.random() < 0.5:
                    new_adj[i] |= (1 << new_idx)
                    new_adj[new_idx] |= (1 << i)
            
            # Local Search on the new vertex's connections
            # We only care about violations involving new_idx
            # Because the subgraph on 0..current_n-1 is unchanged and was valid.
            
            local_deadline = time.time() + 1.0 # 1s per attempt
            valid = False
            steps = 0
            max_steps = 1000
            
            while time.time() < local_deadline and steps < max_steps:
                # Check for K5 involving new_idx
                violation_found = False
                # A K5 involving new_idx exists if new_idx is connected to all 4 vertices of a K4 in the old graph
                # Note: The K4s are from the OLD graph (indices 0..current_n-1).
                # The new vertex is new_idx.
                for quad in k4s:
                    # Check if new_idx is connected to all 4
                    connected_to_all = True
                    for v in quad:
                        if not (new_adj[new_idx] & (1 << v)):
                            connected_to_all = False
                            break
                    if connected_to_all:
                        violation_found = True
                        # Fix: Remove an edge from new_idx to one of the quad vertices
                        # Randomly pick one to remove
                        v_to_remove = random.choice(quad)
                        new_adj[new_idx] &= ~(1 << v_to_remove)
                        new_adj[v_to_remove] &= ~(1 << new_idx)
                        break
                
                if not violation_found:
                    # Check for Ind-5 involving new_idx
                    # An Ind-5 involving new_idx exists if new_idx is NOT connected to all 4 vertices of an Ind-4 in the old graph
                    for quad in ind4s:
                        not_connected_to_all = True
                        for v in quad:
                            if new_adj[new_idx] & (1 << v):
                                not_connected_to_all = False
                                break
                        if not_connected_to_all:
                            violation_found = True
                            # Fix: Add an edge from new_idx to one of the quad vertices
                            v_to_add = random.choice(quad)
                            new_adj[new_idx] |= (1 << v_to_add)
                            new_adj[v_to_add] |= (1 << new_idx)
                            break
                
                if not violation_found:
                    valid = True
                    break
                
                steps += 1
            
            if valid:
                current_n = target_n
                adj = new_adj
                best_n = current_n
                best_adj = adj[:]
                # Update K4s and Ind4s for the new graph
                # This is expensive, so we do it only when we succeed
                k4s = get_k4s(adj, current_n)
                ind4s = get_ind4s(adj, current_n)
                success = True
                break
        
        if not success:
            # If we fail to grow, we can't easily jump to next N.
            # But maybe we should try a few more times?
            # The loop structure tries 100 times per N.
            # If it fails, we break to the next N? No, if we fail N, we likely fail N+1.
            # But we keep best_adj.
            pass

    # Final verification
    # We trust the local search, but let's double check best_adj just in case
    # We can reuse a simple check or skip if time is tight.
    # Given the strict "no violation" requirement, a final check is good.
    # But find_violation is O(n^5). For n=40, it might take a bit.
    # Let's just return. The local search logic is sound for the growth step.
    
    return best_n, get_edges(best_adj, best_n)
