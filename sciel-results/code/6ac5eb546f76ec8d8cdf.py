import time
import random
from itertools import combinations
import numpy as np

def construct():
    deadline = time.time() + 285
    
    def verify_graph(adj, n, size=5):
        """Full verification that no K5 or I5 exists using efficient bit operations."""
        # Precompute neighbor lists as sets for faster intersection checks if needed, 
        # but bitmasks are already fast enough for n<=42.
        # We check all combinations of size 5.
        # Optimization: Iterate through all 5-subsets.
        # 42C5 = 850668.
        # For each, check 10 edges. 8.5M ops. ~1-2s in Python.
        
        # A faster check:
        # For each 5-subset, check if clique or independent.
        for subset in combinations(range(n), size):
            is_clique = True
            is_indep = True
            for i in range(size):
                for j in range(i+1, size):
                    u, v = subset[i], subset[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique or is_indep:
                return False
        return True

    def get_k4s_i4s(adj, n):
        """Get all K4s and I4s in the graph."""
        k4s = []
        i4s = []
        # Iterate all 4-subsets
        for subset in combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for i in range(4):
                for j in range(i+1, 4):
                    u, v = subset[i], subset[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                        if not is_indep:
                            break
                    if not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique:
                k4s.append(frozenset(subset))
            if is_indep:
                i4s.append(frozenset(subset))
        return k4s, i4s

    def try_add_vertex(adj, k4s, i4s, current_n, deadline):
        """
        Try to add a new vertex by finding a set S of neighbors such that:
        1. No K4 is contained in S (so no K5 formed with new vertex)
        2. No I4 is contained in V\S (so no I5 formed with new vertex)
        
        We use local search on the bitmask S.
        """
        best_S = None
        best_violations = float('inf')
        
        for restart in range(100):
            if time.time() > deadline - 10:
                break
            
            # Initialize S randomly
            S_mask = 0
            for v in range(current_n):
                if random.random() < 0.5:
                    S_mask |= (1 << v)
            
            S_set = set(v for v in range(current_n) if (S_mask >> v) & 1)
            
            # Local search to fix violations
            for it in range(2000):
                if time.time() > deadline - 10:
                    break
                
                # Check violations
                bad_k4 = None
                # Check if any K4 is a subset of S
                for k in k4s:
                    if k.issubset(S_set):
                        bad_k4 = k
                        break
                
                bad_i4 = None
                if bad_k4 is None:
                    # Check if any I4 is a subset of V\S
                    V_minus_S = set(range(current_n)) - S_set
                    for i in i4s:
                        if i.issubset(V_minus_S):
                            bad_i4 = i
                            break
                
                if bad_k4 is None and bad_i4 is None:
                    # Success! Return the set of neighbors
                    return S_set
                
                if bad_k4 is not None:
                    # Remove a random vertex from bad_k4 that is in S
                    candidates = list(bad_k4 & S_set)
                    if candidates:
                        v_to_remove = random.choice(candidates)
                        S_set.remove(v_to_remove)
                        S_mask &= ~(1 << v_to_remove)
                    else:
                        # Should not happen if bad_k4 is subset of S
                        break
                elif bad_i4 is not None:
                    # Add a random vertex from bad_i4 that is in V\S
                    candidates = list(bad_i4 - S_set)
                    if candidates:
                        v_to_add = random.choice(candidates)
                        S_set.add(v_to_add)
                        S_mask |= (1 << v_to_add)
                    else:
                        break
                
                # Track best state (optional, but good for robustness)
                # For simplicity, we just return on success or break after max iters
                # If we don't succeed, we discard this restart.
            
            # If we didn't find a solution in this restart, continue to next restart
            
        return None

    # Start from Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_n = q
    current_adj = adj
    
    # Verify the starting graph
    if not verify_graph(current_adj, current_n):
        # Fallback: return edges as is if verification fails (should not happen for Paley 37)
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Precompute K4s and I4s for the initial graph
    k4s, i4s = get_k4s_i4s(current_adj, current_n)
    
    # Try to add vertices one at a time up to 42
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline - 30:
            break
        
        # Try multiple times to add a vertex
        success = False
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            
            S = try_add_vertex(current_adj, k4s, i4s, current_n, deadline)
            if S is not None:
                # Add new vertex
                new_idx = current_n
                current_adj.append(0)
                for v in S:
                    current_adj[v] |= (1 << new_idx)
                    current_adj[new_idx] |= (1 << v)
                
                current_n += 1
                
                # Update K4s and I4s
                # Recompute is safer and fast enough for n<=42
                k4s, i4s = get_k4s_i4s(current_adj, current_n)
                
                success = True
                break
        
        if not success:
            break
    
    # Final verification
    if verify_graph(current_adj, current_n):
        edges = []
        for i in range(current_n):
            for j in range(i+1, current_n):
                if (current_adj[i] >> j) & 1:
                    edges.append((i, j))
        return current_n, edges
    
    # Fallback to Paley graph on 37
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges
