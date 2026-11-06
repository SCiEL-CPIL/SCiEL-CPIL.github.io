import random
import time
from itertools import combinations

def construct():
    # We will try to find a graph for N=42, 43, 44, 45, 46
    # Start with 42 as it's the known best and easier to find.
    # If we find 42, we can try to extend or search for 43 with remaining time.
    
    def solve_for_n(n, time_limit):
        # Precompute 4-subsets
        # For incremental update, we need to know which 4-subsets contain a given pair (u, v)
        # We can store for each pair (u, v) the list of 4-subsets containing it.
        # Number of pairs: n*(n-1)/2
        # For each pair, number of 4-subsets containing it: C(n-2, 2)
        
        subsets_by_pair = {}
        all_4_subsets = []
        
        # Generate all 4-subsets
        for subset in combinations(range(n), 4):
            all_4_subsets.append(subset)
            # Add to pair lists
            s = set(subset)
            # This is inefficient. Better to generate pairs from subset
            for u, v in combinations(subset, 2):
                if (u, v) not in subsets_by_pair:
                    subsets_by_pair[(u, v)] = []
                subsets_by_pair[(u, v)].append(subset)
        
        # Initialize graph
        # Use bitmasks for adjacency
        adj = [0] * n
        # Random init
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Helper to check if a 4-subset is a clique
        def is_clique_4(subset, adj):
            # subset is tuple of 4 ints
            v1, v2, v3, v4 = subset
            # Check edges
            if not (adj[v1] & (1 << v2)): return False
            if not (adj[v1] & (1 << v3)): return False
            if not (adj[v1] & (1 << v4)): return False
            if not (adj[v2] & (1 << v3)): return False
            if not (adj[v2] & (1 << v4)): return False
            if not (adj[v3] & (1 << v4)): return False
            return True
            
        def is_ind_4(subset, adj):
            # Check if NO edges
            v1, v2, v3, v4 = subset
            if (adj[v1] & (1 << v2)): return False
            if (adj[v1] & (1 << v3)): return False
            if (adj[v1] & (1 << v4)): return False
            if (adj[v2] & (1 << v3)): return False
            if (adj[v2] & (1 << v4)): return False
            if (adj[v3] & (1 << v4)): return False
            return True

        # Initial counts
        k4_count = 0
        ind4_count = 0
        for subset in all_4_subsets:
            if is_clique_4(subset, adj):
                k4_count += 1
            if is_ind_4(subset, adj):
                ind4_count += 1
                
        current_energy = k4_count + ind4_count
        best_energy = current_energy
        best_adj = adj[:]
        
        start_time = time.time()
        T = 1.0
        cooling = 0.999
        steps = 0
        
        while time.time() - start_time < time_limit:
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            # Calculate delta
            delta_k4 = 0
            delta_ind4 = 0
            
            # Get subsets containing (u, v)
            if (u, v) in subsets_by_pair:
                for subset in subsets_by_pair[(u, v)]:
                    # Check current state
                    was_clique = is_clique_4(subset, adj)
                    was_ind = is_ind_4(subset, adj)
                    
                    # Simulate flip
                    # Flip edge (u, v)
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
                    is_clique = is_clique_4(subset, adj)
                    is_ind = is_ind_4(subset, adj)
                    
                    # Undo flip
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
                    if was_clique and not is_clique: delta_k4 -= 1
                    if not was_clique and is_clique: delta_k4 += 1
                    if was_ind and not is_ind: delta_ind4 -= 1
                    if not was_ind and is_ind: delta_ind4 += 1
            
            delta_energy = delta_k4 + delta_ind4
            
            # Accept?
            if delta_energy < 0 or (T > 0 and random.random() < pow(2.71828, -delta_energy / T)):
                # Apply flip
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_energy += delta_energy
                k4_count += delta_k4
                ind4_count += delta_ind4
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = adj[:]
                    
                    # Check for K5/Ind5 periodically
                    if best_energy == 0:
                         # Found a graph with no K4 and no Ind4.
                         # This implies no K5 and no Ind5.
                         # Verify and return
                         if check_valid(best_adj, n):
                             return best_adj, n
                             
            T *= cooling
            steps += 1
            if T < 0.01:
                T = 0.01 # Keep some exploration? Or restart?
                # Let's just keep cooling to 0.1
                if T < 0.1: T = 0.1

        # End of time limit for this N
        # Check if best_adj is valid
        if check_valid(best_adj, n):
            return best_adj, n
        return None, 0

    def check_valid(adj, n):
        # Check for K5 and Ind 5
        # Iterate all 5-subsets
        for subset in combinations(range(n), 5):
            if is_clique_5(subset, adj):
                return False
            if is_ind_5(subset, adj):
                return False
        return True
        
    def is_clique_5(subset, adj):
        for i in range(5):
            for j in range(i+1, 5):
                u = subset[i]
                v = subset[j]
                if not (adj[u] & (1 << v)):
                    return False
        return True
        
    def is_ind_5(subset, adj):
        for i in range(5):
            for j in range(i+1, 5):
                u = subset[i]
                v = subset[j]
                if (adj[u] & (1 << v)):
                    return False
        return True

    # Try N from 46 down to 42
    # Allocate time
    # 46: 60s, 45: 60s, 44: 60s, 43: 60s, 42: 60s
    # If 42 fails, return the 37-vertex Paley graph
    
    for n in [46, 45, 44, 43, 42]:
        adj, size = solve_for_n(n, 50)
        if size > 0:
            # Convert adj to edge list
            edges = []
            for i in range(size):
                for j in range(i+1, size):
                    if adj[i] & (1 << j):
                        edges.append((i, j))
            return size, edges
            
    # Fallback: Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
