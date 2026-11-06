import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin
    
    # 1. Generate Paley graph on 37 vertices
    def paley_graph(n):
        if n < 2:
            return []
        def is_prime(x):
            if x < 2: return False
            if x == 2: return True
            if x % 2 == 0: return False
            for i in range(3, int(x**0.5) + 1, 2):
                if x % i == 0: return False
            return True
        if not is_prime(n) or n % 4 != 1:
            return []
        residues = set()
        for x in range(1, n):
            residues.add((x * x) % n)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in residues:
                    edges.append((i, j))
        return edges

    base_n = 37
    base_edges = paley_graph(base_n)
    
    # Helper to build adjacency and check validity efficiently for small N
    # For N=42, we use the specific 5-set counting method.
    
    N_TARGET = 42
    
    # State:
    # adj: list of sets
    # edge_set: set of (u,v) tuples
    # For SA, we need:
    # all_5sets: list of tuples of 5 vertices
    # set_edges: list of int (count of edges in each 5-set)
    # is_bad: list of bool (1 if set_edges is 0 or 10)
    # edge_to_sets: dict mapping (u,v) to list of indices in all_5sets
    
    # Precompute all 5-sets for N=42
    all_vertices = list(range(N_TARGET))
    all_5sets = list(combinations(all_vertices, 5))
    num_5sets = len(all_5sets)
    
    # Map edge to list of 5-set indices containing it
    # This is memory heavy but N=42 is small enough.
    # 850k sets, each has 10 edges. 8.5M entries.
    # Using a dictionary of lists might be slow to build.
    # Alternative: For a given edge (u,v), iterate all 5-sets? No, too slow.
    # We must precompute.
    
    # Optimization: Only build the structure when we reach N_TARGET.
    # For the greedy growth phase, we can use a simpler check.
    
    def count_violations(n, edges):
        """Count number of K5 and independent 5-sets in graph with n vertices."""
        if n < 5:
            return 0
        edge_set = set(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        violations = 0
        for combo in combinations(range(n), 5):
            # Check clique
            is_clique = True
            for i in range(5):
                for j in range(i+1, 5):
                    if combo[j] not in adj[combo[i]]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                violations += 1
                continue # Count as 1 violation type, or both? 
                # A set cannot be both clique and independent unless n<2.
                
            # Check independent
            is_independent = True
            for i in range(5):
                for j in range(i+1, 5):
                    if combo[j] in adj[combo[i]]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                violations += 1
        
        return violations

    # Phase 1: Greedy Growth from 37 to 42
    current_n = base_n
    current_edges = list(base_edges)
    
    for target_n in range(current_n + 1, N_TARGET + 1):
        best_edges = None
        best_violations = float('inf')
        
        # Try several random connection patterns for the new vertex
        # New vertex index is target_n - 1
        new_idx = target_n - 1
        num_trials = 200
        for _ in range(num_trials):
            # Randomly decide connections to existing vertices
            new_edge_list = []
            for i in range(current_n):
                if random.random() < 0.5:
                    new_edge_list.append((i, new_idx))
            
            candidate_edges = current_edges + new_edge_list
            v = count_violations(target_n, candidate_edges)
            
            if v < best_violations:
                best_violations = v
                best_edges = candidate_edges
                if v == 0:
                    break
        
        # If no improvement or high violations, keep the best found
        if best_edges is not None:
            current_edges = best_edges
            current_n = target_n
        else:
            # Fallback: just add vertex with random edges
            new_edge_list = []
            for i in range(current_n):
                if random.random() < 0.5:
                    new_edge_list.append((i, new_idx))
            current_edges = current_edges + new_edge_list
            current_n = target_n

    # Now we have a 42-vertex graph.
    # Phase 2: Simulated Annealing to remove violations
    
    # Build SA structures
    edge_set = set(current_edges)
    adj = [set() for _ in range(N_TARGET)]
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute 5-sets and their edge counts
    # To save memory and time, we can just store the 5-sets and recompute edge counts?
    # No, we need fast updates.
    
    # Let's build the edge_to_sets mapping.
    # This is the most expensive part.
    edge_to_sets = {}
    set_edges = [0] * num_5sets
    
    # Iterate all 5-sets
    for idx, combo in enumerate(all_5sets):
        # Count edges in this combo
        e_count = 0
        for i in range(5):
            for j in range(i+1, 5):
                u, v = combo[i], combo[j]
                if (u, v) in edge_set:
                    e_count += 1
                    # Register this 5-set index for this edge
                    key = (u, v)
                    if key not in edge_to_sets:
                        edge_to_sets[key] = []
                    edge_to_sets[key].append(idx)
        set_edges[idx] = e_count
        
    # Initial energy
    energy = sum(1 for e in set_edges if e == 0 or e == 10)
    
    # SA Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    max_steps = 500000
    
    steps = 0
    while T > min_T and steps < max_steps:
        if time.time() > deadline - 10:
            break
            
        # Pick random edge to flip
        # All possible edges: (u, v) with u < v
        u = random.randint(0, N_TARGET - 1)
        v = random.randint(0, N_TARGET - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        key = (u, v)
        
        # Calculate delta energy
        # Get list of 5-sets containing this edge
        affected_sets = edge_to_sets.get(key, [])
        
        delta_e = 0
        is_edge = key in edge_set
        
        for s_idx in affected_sets:
            old_count = set_edges[s_idx]
            new_count = old_count + (1 if is_edge else -1)
            
            old_bad = 1 if (old_count == 0 or old_count == 10) else 0
            new_bad = 1 if (new_count == 0 or new_count == 10) else 0
            
            delta_e += new_bad - old_bad
            set_edges[s_idx] = new_count
            
        # Acceptance
        if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
            # Apply flip
            if is_edge:
                edge_set.remove(key)
                adj[u].remove(v)
                adj[v].remove(u)
                current_edges.remove(key) # Note: current_edges is a list, removal is O(N). 
                # Better to maintain edge_set as source of truth and rebuild list at end.
            else:
                edge_set.add(key)
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append(key)
                
            energy += delta_e
        else:
            # Revert set_edges
            for s_idx in affected_sets:
                old_count = set_edges[s_idx]
                set_edges[s_idx] = old_count + (-1 if is_edge else 1)
                
        T *= cooling
        steps += 1
        
        if energy == 0:
            break

    # Final check and cleanup
    # Rebuild edge list from edge_set to ensure consistency
    final_edges = list(edge_set)
    
    # Verify validity
    valid, _, _ = check_valid(N_TARGET, final_edges)
    if not valid:
        # Fallback to Paley 37
        return base_n, base_edges
        
    return N_TARGET, final_edges

def check_valid(n, edges):
    """Check if graph has no K5 and no independent set of size 5."""
    edge_set = set(edges)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    for combo in combinations(range(n), 5):
        is_clique = True
        for i in range(5):
            for j in range(i + 1, 5):
                if combo[j] not in adj[combo[i]]:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            return (False, "clique", combo)
            
        is_independent = True
        for i in range(5):
            for j in range(i + 1, 5):
                if combo[j] in adj[combo[i]]:
                    is_independent = False
                    break
            if not is_independent:
                break
        if is_independent:
            return (False, "independent", combo)
            
    return (True, None, None)
