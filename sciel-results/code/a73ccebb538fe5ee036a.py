import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin
    
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Fast adjacency matrix
    adj = [[False] * n for _ in range(n)]
    deg = [0] * n
    edge_set = set()
    
    def add_edge(u, v):
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if adj[u][v]:
            return False
        adj[u][v] = adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
        edge_set.add((u, v))
        return True
    
    def remove_edge(u, v):
        if u > v:
            u, v = v, u
        if not adj[u][v]:
            return False
        adj[u][v] = adj[v][u] = False
        deg[u] -= 1
        deg[v] -= 1
        edge_set.discard((u, v))
        return True
    
    def get_violations():
        """Count violating pairs (s, t) where deg[s]==deg[t] and a 3-path exists."""
        # Group vertices by degree
        deg_groups = {}
        for i in range(n):
            deg_groups.setdefault(deg[i], []).append(i)
        
        violations = 0
        for d, vertices in deg_groups.items():
            if len(vertices) < 2:
                continue
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    s = vertices[i]
                    t = vertices[j]
                    # Check for path s-u-v-t where u != t, v != s, u != v, v != t
                    # s and t are distinct
                    found = False
                    for u in range(n):
                        if u == s or u == t:
                            continue
                        if not adj[s][u]:
                            continue
                        for v in range(n):
                            if v == s or v == t or v == u:
                                continue
                            if adj[u][v] and adj[v][t]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        violations += 1
        return violations
    
    # Start with K_{7,8} bipartite graph
    A = list(range(7))
    B = list(range(7, 15))
    for a in A:
        for b in B:
            add_edge(a, b)
    
    # Verify base
    assert get_violations() == 0, "Base graph should be valid"
    
    best_count = len(edge_set)
    best_edges_snapshot = set(edge_set)
    
    # Try to add intra-part edges
    intra_edges = [(i, j) for i in range(7) for j in range(i + 1, 7)] + \
                  [(i, j) for i in range(7, 15) for j in range(i + 1, 15)]
    
    for e in intra_edges:
        if time.time() > deadline:
            break
        u, v = e
        if not adj[u][v]:
            add_edge(u, v)
            if get_violations() == 0:
                best_count = len(edge_set)
                best_edges_snapshot = set(edge_set)
            else:
                remove_edge(u, v)
    
    # Local search: Try random swaps and additions
    # We maintain the invariant that the current graph is valid.
    # Move 1: Add edge. If valid, keep.
    # Move 2: Remove edge, then add different edge. If valid and count >= old count, keep.
    
    # To speed up, we can precompute which edges are "safe" to add? No, degrees change.
    
    # Optimization: Instead of full scan, try random edges.
    current_edges_list = list(edge_set)
    
    for _ in range(10000):
        if time.time() > deadline:
            break
        
        # Try adding a random edge
        if len(edge_set) < len(all_edges):
            missing = [e for e in all_edges if e not in edge_set]
            if missing:
                u, v = random.choice(missing)
                add_edge(u, v)
                if get_violations() == 0:
                    best_count = len(edge_set)
                    best_edges_snapshot = set(edge_set)
                else:
                    remove_edge(u, v)
        
        # Try swapping: remove one, add another
        if edge_set:
            rem_u, rem_v = random.choice(list(edge_set))
            remove_edge(rem_u, rem_v)
            
            missing = [e for e in all_edges if e not in edge_set]
            if missing:
                # Try a few candidates
                random.shuffle(missing)
                for add_u, add_v in missing[:5]:
                    add_edge(add_u, add_v)
                    if get_violations() == 0:
                        # Keep this state, and maybe try to add more immediately?
                        best_count = len(edge_set)
                        best_edges_snapshot = set(edge_set)
                        break
                    else:
                        remove_edge(add_u, add_v)
                
                # If we didn't add any, we still need to restore the removed edge or try others?
                # Wait, if we removed rem and tried adds and failed, we are left with graph missing rem.
                # We must restore rem to maintain validity (assuming it was valid before).
                # Actually, if we failed to add any edge such that the new graph is valid,
                # we should restore the original state.
                # But wait, did we add one? The loop breaks if added. If not added, we need to re-add rem.
                if not any((rem_u, rem_v) != (add_u, add_v) for add_u, add_v in []): # This logic is messy
                    pass
                
                # Let's re-do the swap logic more cleanly.
                pass

    # Let's rewrite the loop properly to be safe.
    # Reset to best snapshot
    # Actually, let's just restart from scratch with a better loop structure.
    
    # Clear current state
    for e in list(edge_set):
        remove_edge(*e)
    for e in best_edges_snapshot:
        add_edge(*e)
        
    best_count = len(edge_set)
    
    # Local Search
    for step in range(20000):
        if time.time() > deadline:
            break
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Add
            missing = [e for e in all_edges if e not in edge_set]
            if not missing:
                continue
            u, v = random.choice(missing)
            add_edge(u, v)
            if get_violations() == 0:
                if len(edge_set) > best_count:
                    best_count = len(edge_set)
                    best_edges_snapshot = set(edge_set)
            else:
                remove_edge(u, v)
                
        else:
            # Swap: Remove one, Add one
            if not edge_set:
                continue
            rem_u, rem_v = random.choice(list(edge_set))
            remove_edge(rem_u, rem_v)
            
            missing = [e for e in all_edges if e not in edge_set]
            if not missing:
                # Restore
                add_edge(rem_u, rem_v)
                continue
                
            # Try up to 3 random additions
            tried = 0
            success = False
            candidates = random.sample(missing, min(3, len(missing)))
            for add_u, add_v in candidates:
                add_edge(add_u, add_v)
                if get_violations() == 0:
                    # Valid state reached.
                    if len(edge_set) > best_count:
                        best_count = len(edge_set)
                        best_edges_snapshot = set(edge_set)
                    success = True
                    break
                else:
                    remove_edge(add_u, add_v)
            
            if not success:
                # Restore original edge
                add_edge(rem_u, rem_v)

    # Final cleanup: ensure we are at best_edges_snapshot
    for e in list(edge_set):
        remove_edge(*e)
    for e in best_edges_snapshot:
        add_edge(*e)
        
    # One more pass: greedy add of any remaining edge
    for e in all_edges:
        if time.time() > deadline:
            break
        if e not in edge_set:
            add_edge(*e)
            if get_violations() == 0:
                best_count = len(edge_set)
                best_edges_snapshot = set(edge_set)
            else:
                remove_edge(*e)

    # Restore best snapshot again
    for e in list(edge_set):
        remove_edge(*e)
    for e in best_edges_snapshot:
        add_edge(*e)

    if get_violations() != 0:
        # Fallback to known valid K_{7,8}
        for e in list(edge_set):
            remove_edge(*e)
        for a in A:
            for b in B:
                add_edge(a, b)
        best_edges_snapshot = set(edge_set)

    result = [[int(u), int(v)] for u, v in best_edges_snapshot]
    return result
