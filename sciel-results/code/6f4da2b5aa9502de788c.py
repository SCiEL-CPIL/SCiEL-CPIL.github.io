import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a, b) > 1 
    for all distinct pairs. Maximizes cardinality.
    
    Strategy:
    1. The set of all multiples of 3 in [1, 255] has size 85 and is a valid
       pairwise non-coprime set containing 255.
    2. We attempt to improve upon this by finding a larger clique in the
       graph of numbers in [1, 255] that share a factor with 255 (i.e., are
       divisible by 3, 5, or 17), where edges connect pairs with gcd > 1.
    3. We use Bron-Kerbosch with pivot selection, seeded with the known
       solution (multiples of 3) to ensure we never do worse than 85.
    4. If the clique search finds a larger set, we return it; otherwise,
       we return the multiples of 3.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    # Base solution: all multiples of 3 in [1, 255]
    base_solution = list(range(3, 256, 3))
    best_clique_size = len(base_solution)  # 85
    best_clique = set(base_solution)
    
    # Build candidate set: all numbers in [1, 255] divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Ensure 255 is in candidates
    if 255 not in candidates:
        candidates.append(255)
    
    n = len(candidates)
    
    # Build adjacency using sets for Bron-Kerbosch
    # adj[i] is a set of indices j != i such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Find index of 255
    idx_255 = candidates.index(255)
    
    # We need max clique containing idx_255
    # Use Bron-Kerbosch with pivot
    # Since we already have a clique of size 85, we can prune branches
    # that cannot exceed this size.
    
    best_clique_indices = set()
    # Initialize with the indices of the base solution
    for val in base_solution:
        best_clique_indices.add(candidates.index(val))
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique_size, best_clique_indices
        
        if time.time() > deadline:
            return
        
        # Pruning: if current clique size + |P| <= best_clique_size, no improvement possible
        if len(R) + len(P) <= best_clique_size:
            return
        
        # Update best if we have a larger clique
        if len(R) > best_clique_size:
            best_clique_size = len(R)
            best_clique_indices = set(R)
        
        if not P:
            return
        
        # Choose pivot u from P ∪ X to minimize |P \ N(u)|
        # i.e., maximize |P ∩ N(u)|
        pivot_set = P | X
        if not pivot_set:
            pivot_set = P
        
        best_u = -1
        best_count = -1
        for u in pivot_set:
            count = len(P & adj[u])
            if count > best_count:
                best_count = count
                best_u = u
        
        # Iterate over P \ N(best_u)
        candidates_iter = P - adj[best_u] if best_u != -1 else P
        
        for v in list(candidates_iter):
            if time.time() > deadline:
                return
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P = P - {v}
            X = X | {v}
    
    # Start with R = {255}, P = neighbors of 255, X = empty
    # But we can do better: start with R containing all multiples of 3 that are neighbors of 255
    # Actually, let's just start with R={255} and let the algorithm explore.
    # The pruning will handle the rest.
    
    R = {idx_255}
    P = adj[idx_255].copy()
    X = set()
    
    bron_kerbosch(R, P, X)
    
    # Convert best_clique_indices back to values
    if best_clique_indices:
        result = sorted([candidates[i] for i in best_clique_indices])
        # Verify 255 is in result
        if 255 in result:
            return result
    
    # Fallback to base solution
    return base_solution
# EVOLVE-BLOCK-END
