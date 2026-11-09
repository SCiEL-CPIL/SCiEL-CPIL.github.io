import sys
import time
import math
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    N = 255
    target = 255
    
    # 1. Filter candidates that can coexist with 255
    # A number x can be in the set with 255 only if gcd(x, 255) > 1
    candidates = [i for i in range(1, N + 1) if i != target and math.gcd(i, target) > 1]
    
    # Build adjacency list for the "Non-Coprime" graph (edges if gcd > 1)
    # We want a Max Clique in this graph among 'candidates'
    # Plus the target 255 itself
    
    # Optimization: Precompute gcds? Or just use math.gcd
    # N is small (126 nodes), O(N^2) is fine
    
    n = len(candidates)
    # Map index to value for quick access
    # adj[u] is a bitset or list of indices v such that gcd(candidates[u], candidates[v]) > 1
    adj = [0] * n
    for u in range(n):
        cu = candidates[u]
        for v in range(u + 1, n):
            cv = candidates[v]
            if math.gcd(cu, cv) > 1:
                # Set bit v in adj[u] and bit u in adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    # Bron-Kerbosch with pivot for Max Clique
    # We use a recursive approach with bitsets for speed
    
    best_clique = 0
    best_clique_set = set()
    
    def bk(R, P, X):
        """
        R: set of nodes in current clique (bitmask or set)
        P: set of candidates (nodes that can be added)
        X: set of excluded nodes (already processed)
        """
        nonlocal best_clique, best_clique_set
        nonlocal deadline
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            # Found a maximal clique
            if len(R) > best_clique:
                best_clique = len(R)
                best_clique_set = set(R)
            return
            
        if not P:
            # No more nodes to add, maximal clique
            if len(R) > best_clique:
                best_clique = len(R)
                best_clique_set = set(R)
            return

        # Select pivot u from P union X to minimize branching
        # u with max degree in P
        P_set = set(P)
        X_set = set(X)
        candidates_for_pivot = P_set.union(X_set)
        
        # Heuristic: pick pivot with max neighbors in P
        # For small n, iterating is fine
        best_pivot = None
        max_deg = -1
        for u in candidates_for_pivot:
            # Count neighbors in P
            # adj[u] is a bitmask
            # P is a set of ints
            # Convert P to bitmask for intersection?
            # Or just loop
            deg = 0
            for v in P_set:
                if (adj[u] >> v) & 1:
                    deg += 1
            if deg > max_deg:
                max_deg = deg
                best_pivot = u
                
        if best_pivot is None:
            return
            
        # Branch on P \ N(best_pivot)
        # Neighbors of best_pivot in P
        neighbors_in_P = set()
        for v in P_set:
            if (adj[best_pivot] >> v) & 1:
                neighbors_in_P.add(v)
                
        # Nodes to branch on: P - neighbors_in_P
        to_branch = P_set.difference(neighbors_in_P)
        
        for v in to_branch:
            # New R
            new_R = R.union({v})
            # New P: nodes in P connected to v
            new_P = set()
            for u in P_set:
                if u != v and ((adj[v] >> u) & 1):
                    new_P.add(u)
            # New X: nodes in X connected to v
            new_X = set()
            for u in X_set:
                if ((adj[v] >> u) & 1):
                    new_X.add(u)
            
            bk(new_R, new_P, new_X)
            
            # Move v from P to X
            P_set.discard(v)
            X_set.add(v)
            
            if time.time() > deadline:
                return

    # Initial call
    # R is empty
    # P is all candidates
    # X is empty
    initial_P = set(range(n))
    bk(set(), initial_P, set())
    
    # The found clique is in candidates.
    # Add 255 to it.
    result_clique_size = best_clique + 1
    
    # Construct the result list
    # If best_clique_set is not empty, map back to values
    # If timeout occurred before finding any, fallback to a known good set
    if best_clique_set:
        result_values = [candidates[i] for i in best_clique_set]
        result_values.append(target)
    else:
        # Fallback: Multiples of 3
        result_values = list(range(3, N + 1, 3))
        if target not in result_values:
             # 255 is a multiple of 3, so it should be there
             pass

    return result_values
# EVOLVE-BLOCK-END
