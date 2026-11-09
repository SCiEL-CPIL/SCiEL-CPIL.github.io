# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    # Generate all 4-subsets as integers
    # Bit i corresponds to element i+1
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    
    # Precompute popcounts? Not strictly necessary if using bit_count()
    
    # Build conflict graph adjacency bitsets
    # adj[i] is a bitmask of all j such that (i, j) is an edge
    adj = [0] * n_sets
    
    # Universe mask for 16 elements
    U = (1 << 16) - 1
    
    # Iterate over all pairs to build graph
    # This might be slow in pure Python if not optimized, but N=2184 is small enough
    # 2184 * 2183 / 2 ~ 2.4M pairs
    # Each pair check: bitwise AND, OR, NOT, bit_count.
    
    for i in range(n_sets):
        a = all_sets[i]
        for j in range(i + 1, n_sets):
            b = all_sets[j]
            
            # Check if there exists C such that A, B, C form a sunflower
            # Condition: A cap B = A cap C = B cap C = I
            # Let I = A cap B.
            # C must satisfy: C cap A = I and C cap B = I.
            # This implies I is subset of C, C is disjoint from (A \ I) and (B \ I).
            # Let Z be elements not in A union B.
            # C = I union W, where W is subset of Z, size 4 - |I|.
            # Such C exists if |Z| >= 4 - |I|.
            # And C must be distinct from A and B.
            # If 4 - |I| > 0, C contains elements from Z, A and B do not, so C != A, C != B.
            # If 4 - |I| == 0, then |I|=4, so A=B (since |A|=|B|=4). But i!=j so A!=B.
            # So for distinct A,B, |I| < 4, so 4-|I| > 0.
            
            inter = a & b
            inter_size = inter.bit_count()
            
            union = a | b
            # Z is complement of union in U
            z = U & ~union
            z_size = z.bit_count()
            
            k = 4 - inter_size
            
            if z_size >= k:
                # Edge exists
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Greedy Max Independent Set Heuristic
    best_size = 0
    best_mis_indices = []
    
    start_time = time.time()
    max_time = 55.0
    rng = random.Random(42)
    
    # Run multiple trials
    while time.time() - start_time < max_time:
        # Randomly shuffle the order of nodes to consider
        order = list(range(n_sets))
        rng.shuffle(order)
        
        available = (1 << n_sets) - 1  # Bitmask of available nodes
        mis_indices = []
        
        for u in order:
            if (available >> u) & 1:
                # Add u to MIS
                mis_indices.append(u)
                # Remove u and its neighbors from available
                available &= ~adj[u]
                available &= ~(1 << u)
                
        if len(mis_indices) > best_size:
            best_size = len(mis_indices)
            best_mis_indices = mis_indices[:]
            
    # Convert indices back to sets
    return [sorted([i+1 for i in range(16) if (all_sets[idx] >> i) & 1]) for idx in best_mis_indices]
# EVOLVE-BLOCK-END
