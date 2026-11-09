"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 255
    target = 255
    
    # All candidate numbers that share a factor with 255
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, target) > 1:
            candidates.append(x)
    
    n_cand = len(candidates)
    
    # Build adjacency matrix: adj[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    adj = [[False] * n_cand for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    # Find index of 255
    target_idx = candidates.index(target)
    
    # Maximum clique containing target_idx
    # Use greedy + local search with multiple restarts
    
    def greedy_clique(start_set_indices, adj, n):
        """Grow a clique greedily by adding compatible vertices."""
        clique = set(start_set_indices)
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if v in clique:
                    continue
                # Check if v is compatible with all in clique
                compatible = all(adj[v][u] for u in clique)
                if compatible:
                    clique.add(v)
                    changed = True
                    break
        return clique
    
    def local_search(clique, adj, n, target_idx, max_iters=1000):
        """Try to improve clique by swapping elements."""
        clique = set(clique)
        best_size = len(clique)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try adding one element
            for v in range(n):
                if v in clique:
                    continue
                compatible = all(adj[v][u] for u in clique)
                if compatible:
                    clique.add(v)
                    best_size += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add one or more
            for u in list(clique):
                if u == target_idx:
                    continue
                # Try removing u and see if we can add more
                new_clique = clique - {u}
                # Try to grow new_clique
                added = 0
                for v in range(n):
                    if v in new_clique:
                        continue
                    compatible = all(adj[v][w] for w in new_clique)
                    if compatible:
                        new_clique.add(v)
                        added += 1
                        if added >= 2:  # Net gain of at least 1
                            break
                
                if len(new_clique) > len(clique):
                    clique = new_clique
                    best_size = len(clique)
                    improved = True
                    break
        
            if not improved:
                break
        
        return clique
    
    # Strategy 1: Start with all multiples of 3
    mult3_indices = [i for i, x in enumerate(candidates) if x % 3 == 0]
    if target_idx not in mult3_indices:
        mult3_indices.append(target_idx)
    
    best_clique = set(mult3_indices)
    best_size = len(best_clique)
    
    # Grow greedily
    best_clique = greedy_clique(best_clique, adj, n_cand)
    best_size = len(best_clique)
    
    # Local search
    best_clique = local_search(best_clique, adj, n_cand, target_idx)
    best_size = len(best_clique)
    
    # Strategy 2: Start with all multiples of 5
    mult5_indices = [i for i, x in enumerate(candidates) if x % 5 == 0]
    if target_idx not in mult5_indices:
        mult5_indices.append(target_idx)
    
    clique5 = set(mult5_indices)
    clique5 = greedy_clique(clique5, adj, n_cand)
    clique5 = local_search(clique5, adj, n_cand, target_idx)
    if len(clique5) > best_size:
        best_clique = clique5
        best_size = len(clique5)
    
    # Strategy 3: Start with all multiples of 17
    mult17_indices = [i for i, x in enumerate(candidates) if x % 17 == 0]
    if target_idx not in mult17_indices:
        mult17_indices.append(target_idx)
    
    clique17 = set(mult17_indices)
    clique17 = greedy_clique(clique17, adj, n_cand)
    clique17 = local_search(clique17, adj, n_cand, target_idx)
    if len(clique17) > best_size:
        best_clique = clique17
        best_size = len(clique17)
    
    # Strategy 4: Random restarts - start with random subsets and grow
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Start with target and a few random compatible elements
        init_clique = {target_idx}
        
        # Add random elements that are compatible with current clique
        attempts = 0
        while len(init_clique) < 20 and attempts < 200:
            v = random.randint(0, n_cand - 1)
            if v in init_clique:
                attempts += 1
                continue
            compatible = all(adj[v][u] for u in init_clique)
            if compatible:
                init_clique.add(v)
            attempts += 1
        
        # Grow greedily
        init_clique = greedy_clique(init_clique, adj, n_cand)
        
        # Local search
        init_clique = local_search(init_clique, adj, n_cand, target_idx, max_iters=500)
        
        if len(init_clique) > best_size:
            best_clique = init_clique
            best_size = len(init_clique)
    
    # Convert back to values
    result = sorted([candidates[i] for i in best_clique])
    
    # Ensure 255 is in the set
    if target not in result:
        result.append(target)
        result = sorted(result)
    
    # Final verification
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to multiples of 3
        return list(range(3, 256, 3))
    
    return result
# EVOLVE-BLOCK-END
