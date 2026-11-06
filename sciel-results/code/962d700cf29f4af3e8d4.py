"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 330 - 30  # 30s safety margin
    
    # Seed: Paley graph on 37 vertices (verified (5,5)-free)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_n = q
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    best_n = seed_n
    best_edges = seed_edges
    
    # Build adjacency as bitmasks for speed
    # adj[i] is a bitmask where bit j is set if edge (i,j) exists
    adj_masks = [0] * q
    for i, j in seed_edges:
        adj_masks[i] |= (1 << j)
        adj_masks[j] |= (1 << i)
    
    random.seed(42)
    
    def check_5_clique_free(n, adj_masks):
        """Check no 5-clique exists using bitmask operations."""
        for combo in itertools.combinations(range(n), 5):
            # Check all 10 pairs
            valid = True
            for idx_a in range(5):
                for idx_b in range(idx_a + 1, 5):
                    a, b = combo[idx_a], combo[idx_b]
                    if not (adj_masks[a] & (1 << b)):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return False
        return True
    
    def check_5_independent_free(n, adj_masks):
        """Check no 5-independent-set exists."""
        for combo in itertools.combinations(range(n), 5):
            # Check no edges among these 5
            valid = True
            for idx_a in range(5):
                for idx_b in range(idx_a + 1, 5):
                    a, b = combo[idx_a], combo[idx_b]
                    if adj_masks[a] & (1 << b):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return False
        return True
    
    def get_4_cliques_bitmask(n, adj_masks):
        """Get all 4-cliques using bitmask operations."""
        cliques = []
        for combo in itertools.combinations(range(n), 4):
            a, b, c, d = combo
            if (adj_masks[a] & (1 << b)) and (adj_masks[a] & (1 << c)) and \
               (adj_masks[a] & (1 << d)) and (adj_masks[b] & (1 << c)) and \
               (adj_masks[b] & (1 << d)) and (adj_masks[c] & (1 << d)):
                cliques.append(combo)
        return cliques
    
    def get_4_independent_bitmask(n, adj_masks):
        """Get all 4-independent-sets using bitmask operations."""
        indep = []
        for combo in itertools.combinations(range(n), 4):
            a, b, c, d = combo
            if not (adj_masks[a] & (1 << b)) and not (adj_masks[a] & (1 << c)) and \
               not (adj_masks[a] & (1 << d)) and not (adj_masks[b] & (1 << c)) and \
               not (adj_masks[b] & (1 << d)) and not (adj_masks[c] & (1 << d)):
                indep.append(combo)
        return indep
    
    current_n = q
    current_adj = adj_masks[:]
    
    four_cliques = get_4_cliques_bitmask(current_n, current_adj)
    four_indep = get_4_independent_bitmask(current_n, current_adj)
    
    # Try to add vertices with smarter random patterns
    max_add_attempts = 2000
    attempts = 0
    
    while time.time() < deadline and attempts < max_add_attempts:
        attempts += 1
        current_n = len(current_adj)
        
        # Try different connection densities
        density = random.uniform(0.3, 0.7)
        new_conn = [random.random() < density for _ in range(current_n)]
        
        # Check 4-cliques constraint
        valid = True
        for clique in four_cliques:
            if all(new_conn[v] for v in clique):
                valid = False
                break
        
        if valid:
            # Check 4-independent constraint
            for ind in four_indep:
                if not any(new_conn[v] for v in ind):
                    valid = False
                    break
        
        if valid:
            # Add vertex
            new_mask = 0
            for v in range(current_n):
                if new_conn[v]:
                    new_mask |= (1 << v)
                    current_adj[v] |= (1 << current_n)
            current_adj.append(new_mask)
            
            # Update cliques and independent sets
            four_cliques = get_4_cliques_bitmask(len(current_adj), current_adj)
            four_indep = get_4_independent_bitmask(len(current_adj), current_adj)
            
            # Update best
            if len(current_adj) > best_n:
                edges = []
                for i in range(len(current_adj)):
                    for j in range(i + 1, len(current_adj)):
                        if current_adj[i] & (1 << j):
                            edges.append((i, j))
                best_n = len(current_adj)
                best_edges = edges
    
    # Final verification
    n = best_n
    edges = best_edges
    
    if n > q:
        # Verify the improved graph
        verify_adj = [0] * n
        for i, j in edges:
            verify_adj[i] |= (1 << j)
            verify_adj[j] |= (1 << i)
        
        if not check_5_clique_free(n, verify_adj) or not check_5_independent_free(n, verify_adj):
            # Fallback to seed
            return seed_n, seed_edges
    
    return n, edges
# EVOLVE-BLOCK-END
