import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Seed: Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges_37 = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Target: 43
    target_n = 43
    
    # Precompute combinations for 43 vertices
    comb_43 = list(combinations(range(target_n), 5))
    
    # Precompute for each 5-tuple, the 10 pair masks as a tuple of (u,v) pairs
    # This avoids recomputing them each time
    pair_indices_43 = []
    for c in comb_43:
        v0, v1, v2, v3, v4 = c
        pairs = [(v0,v1), (v0,v2), (v0,v3), (v0,v4), (v1,v2), (v1,v3), (v1,v4), (v2,v3), (v2,v4), (v3,v4)]
        pair_indices_43.append(pairs)
    
    def check_and_get_violation(adj):
        """Check if graph is (5,5)-free. Returns (valid, violation_5set)."""
        for idx, c in enumerate(comb_43):
            v0, v1, v2, v3, v4 = c
            pairs = pair_indices_43[idx]
            cnt = 0
            for u, v in pairs:
                if adj[u] & (1 << v):
                    cnt += 1
            if cnt == 10 or cnt == 0:
                return False, c
        return True, None
    
    def count_violations_fast(adj):
        """Count total violations (cliques + independent sets)."""
        total = 0
        for idx, c in enumerate(comb_43):
            v0, v1, v2, v3, v4 = c
            pairs = pair_indices_43[idx]
            cnt = 0
            for u, v in pairs:
                if adj[u] & (1 << v):
                    cnt += 1
            if cnt == 10 or cnt == 0:
                total += 1
        return total
    
    def init_paley_extended(n, densities=None):
        """Initialize with Paley(37) + random edges for new vertices."""
        if densities is None:
            densities = [0.5] * (n - 37)
        
        adj = [0] * n
        
        # Fill 0-36 with Paley
        for i in range(37):
            for j in range(i+1, 37):
                if (j-i) % 37 in residues:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        
        # Fill 37-(n-1) with random edges
        for i in range(37, n):
            p = densities[i - 37] if i - 37 < len(densities) else 0.5
            for j in range(n):
                if i == j:
                    continue
                if random.random() < p:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        
        return adj
    
    def local_search(adj, max_iters=500, temperature=1.0):
        """Run local search to reduce violations."""
        n = target_n
        current_viol, current_c = check_and_get_violation(adj)
        
        if current_viol:
            return adj, True
        
        best_adj = list(adj)
        best_score = count_violations_fast(adj)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Get a violation
            valid, viol = check_and_get_violation(adj)
            if valid:
                return adj, True
            
            # Pick a random edge from the violating 5-set to flip
            v0, v1, v2, v3, v4 = viol
            pairs = [(v0,v1), (v0,v2), (v0,v3), (v0,v4), 
                     (v1,v2), (v1,v3), (v1,v4), 
                     (v2,v3), (v2,v4), (v3,v4)]
            u, v = random.choice(pairs)
            
            # Try the flip
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_score = count_violations_fast(adj)
            
            if new_score <= best_score:
                best_score = new_score
                best_adj = list(adj)
            else:
                # Metropolis acceptance
                delta = new_score - best_score
                if random.random() < (1.0 / (1.0 + delta * temperature)):
                    best_score = new_score
                    best_adj = list(adj)
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            
            # Cool down temperature
            temperature *= 0.999
        
        # Check best
        valid, _ = check_and_get_violation(best_adj)
        if valid:
            return best_adj, True
        return best_adj, False
    
    # Try multiple strategies
    best_n = 37
    best_edges = edges_37
    
    # Strategy 1: Try to find 43-vertex solution
    found_43 = False
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Vary the density for new vertices
        base_density = 0.4 + random.random() * 0.3
        densities = [base_density + random.gauss(0, 0.1) for _ in range(6)]
        densities = [max(0.1, min(0.9, d)) for d in densities]
        
        adj = init_paley_extended(target_n, densities)
        
        # Run local search
        adj, valid = local_search(adj, max_iters=300, temperature=1.5)
        
        if valid:
            found_43 = True
            best_n = target_n
            best_edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if adj[i] & (1 << j):
                        best_edges.append((i, j))
            break
    
    if found_43:
        return best_n, best_edges
    
    # If we couldn't find 43, return the seed 37
    return 37, edges_37
