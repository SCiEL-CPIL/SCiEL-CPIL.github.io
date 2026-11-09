import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    n = 16
    # Precompute all 8-vertex subsets
    subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets)
    
    # Precompute list of subsets containing each pair of vertices
    # pair_to_subsets[u][v] = list of indices
    pair_to_subsets = [[[] for _ in range(n)] for _ in range(n)]
    for i, S in enumerate(subsets):
        for u_idx in range(8):
            for v_idx in range(u_idx + 1, 8):
                u = S[u_idx]
                v = S[v_idx]
                pair_to_subsets[u][v].append(i)
                pair_to_subsets[v][u].append(i)

    def has_q3_mapping(S, adj):
        """
        Returns a list 'mapping' of length 8 where mapping[i] is the cube node
        assigned to S[i], or None if no Q3 found.
        """
        assigned = [-1] * 8
        
        def backtrack(idx):
            if idx == 8:
                return True
            
            v = S[idx]
            # Try all cube nodes 0-7
            for c in range(8):
                if c in assigned:
                    continue
                
                ok = True
                # Check against previously assigned vertices
                for j in range(idx):
                    vj = S[j]
                    # Check if edge(v, vj) exists in G
                    edge_exists = (adj[v] >> vj) & 1
                    # Check if edge(c, assigned[j]) exists in Cube
                    cube_edge_exists = ((c ^ assigned[j]) in (1, 2, 4))
                    
                    # If cube edge exists, graph edge MUST exist
                    if cube_edge_exists and not edge_exists:
                        ok = False
                        break
                
                if ok:
                    assigned[idx] = c
                    if backtrack(idx + 1):
                        return True
                    assigned[idx] = -1
            
            return False
        
        if backtrack(0):
            return assigned
        return None

    best_edges = []
    best_count = 0

    # Run multiple restarts
    while time.time() < deadline:
        # Init graph K16
        adj = [0] * n
        for i in range(n):
            mask = (1 << n) - 1
            mask &= ~(1 << i)
            adj[i] = mask
        
        # Edge counts for subsets
        edge_counts = [28] * num_subsets
        active = [True] * num_subsets
        
        removed_edges = []
        
        # Phase 1: Remove edges to destroy all Q3s
        while time.time() < deadline:
            found_q3 = False
            # Iterate through active subsets
            # To make it more effective, we can shuffle the active indices?
            # Or just iterate.
            for i in range(num_subsets):
                if not active[i]:
                    continue
                if edge_counts[i] < 12:
                    active[i] = False
                    continue
                
                S = subsets[i]
                mapping = has_q3_mapping(S, adj)
                
                if mapping:
                    # Found Q3
                    # Identify edges of this Q3
                    q3_edges = []
                    for a in range(8):
                        for b in range(a + 1, 8):
                            if (mapping[a] ^ mapping[b]) in (1, 2, 4):
                                q3_edges.append((S[a], S[b]))
                    
                    # Pick an edge to remove
                    # Heuristic: Pick the one with highest degree sum?
                    # Or random.
                    e = random.choice(q3_edges)
                    u, v = e
                    
                    # Remove edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    removed_edges.append(e)
                    
                    # Update counts for subsets containing u, v
                    for j in pair_to_subsets[u][v]:
                        edge_counts[j] -= 1
                        if edge_counts[j] < 12:
                            active[j] = False
                    
                    found_q3 = True
                    break
                else:
                    # No Q3 in this subset
                    active[i] = False
            
            if not found_q3:
                break
        
        # Phase 2: Add edges back
        # Shuffle removed edges
        random.shuffle(removed_edges)
        
        # We need to re-activate subsets?
        # No, we just check validity.
        # But we need to be careful: adding an edge might create a Q3 in a subset
        # that was previously marked inactive (safe).
        # So we must check ALL subsets containing the new edge.
        
        for e in removed_edges:
            if time.time() > deadline:
                break
            u, v = e
            
            # Check if adding (u,v) is safe
            safe = True
            # Only check subsets containing u and v
            for j in pair_to_subsets[u][v]:
                S = subsets[j]
                # Temporarily add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Check if this subset contains Q3
                # We can use has_q3_mapping, but it's slow.
                # We can optimize: if edge_counts[j] < 12 before adding, it was safe.
                # After adding, count is edge_counts[j] + 1.
                # If edge_counts[j] + 1 < 12, still safe.
                # So only check if edge_counts[j] >= 11.
                if edge_counts[j] >= 11:
                    if has_q3_mapping(S, adj):
                        safe = False
                        break
                
                # Revert edge for next check
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            
            if safe:
                # Keep edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                # Update edge_counts?
                # Not strictly necessary if we don't do more removals.
                # But if we do restarts, we re-init.
                # For this run, we are done with removal.
                # So we just build the final graph.
        
        # Count edges
        count = sum(bin(adj[i]).count("1") for i in range(n)) // 2
        
        if count > best_count:
            best_count = count
            # Build edge list
            best_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        best_edges.append([i, j])
        
        # If we have time, continue restarts
        if time.time() > deadline - 5:
            break

    return best_edges
