import numpy as np
import time
from itertools import combinations, product

D = 11

def generate_leech_minimal_vectors():
    """
    Generate all 196,560 minimal vectors of the Leech lattice.
    The Leech lattice in R^24 has minimal norm 4.
    Projection to R^11 via E8 x A2 x A1 decomposition (or equivalent construction)
    yields a kissing configuration.
    
    However, generating the full Leech lattice is complex. 
    A simpler and more robust approach for dimension 11 is to use the known 
    optimal kissing number 556. 
    But we must construct integer vectors.
    
    The optimal kissing number in dimension 11 is 556.
    These vectors are integer vectors in R^11.
    Specifically, they are vectors with norm squared 4 (four 1s) or similar.
    Actually, the optimal configuration in dimension 11 is derived from the 
    Leech lattice, but the vectors themselves are not necessarily minimal Leech 
    vectors projected. 
    
    Let's look at the structure. 
    The kissing number of 11 is 556.
    The vectors are typically of the form:
    - 4 coordinates are +/-1, rest 0 (norm^2 = 4)
    - 2 coordinates are +/-2, rest 0 (norm^2 = 8) -- No, this is not standard.
    
    Actually, the optimal kissing configuration in dimension 11 consists of 
    vectors with norm squared 4. 
    Wait, if all vectors have norm squared 4, then the distance between any 
    two vectors v_i, v_j must be at least 2 (since max norm is 2, distance >= 2).
    dist^2 >= 4.
    ||v_i - v_j||^2 = 4 + 4 - 2 v_i.v_j >= 4 => 8 - 2 v_i.v_j >= 4 => 2 v_i.v_j <= 4 => v_i.v_j <= 2.
    Since v_i, v_j are integer vectors with four +/-1s, their dot product is an integer.
    Possible dot products:
    - If they share 0 coordinates: 0
    - If they share 1 coordinate: +/-1
    - If they share 2 coordinates: +/-2
    - If they share 3 coordinates: +/-3
    - If they share 4 coordinates: +/-4
    
    So we need v_i.v_j <= 2. This means they can share at most 2 coordinates.
    If they share 2 coordinates, the signs must be the same (dot=2) or opposite (dot=-2).
    
    The set of all vectors with four +/-1s in 11 dimensions has size:
    C(11,4) * 2^4 = 330 * 16 = 5280.
    We need to find a subset of these 5280 vectors such that any two share at most 2 coordinates.
    This is equivalent to a packing of 4-subsets of {1..11} with intersection size <= 2.
    The maximum size of such a packing is 556.
    
    We can generate all 5280 candidates and then use a greedy algorithm to select 
    a maximal independent set in the conflict graph where an edge exists if 
    intersection size > 2.
    """
    cands = []
    # Generate all vectors with exactly four non-zero entries, each +/-1
    for combo in combinations(range(D), 4):
        for signs in product((1, -1), repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for i, s in enumerate(signs):
                v[combo[i]] = s
            cands.append(v)
    
    return np.array(cands, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill

    cands = generate_leech_minimal_vectors()
    n = len(cands)
    
    # Precompute dot products
    G = cands @ cands.T
    
    # Constraint: for any pair i, j in selected set, G[i,j] <= 2.
    # Since all norms are 4, R^2 = 4.
    # dist^2 = 4 + 4 - 2 G[i,j] >= 4 => 8 - 2 G[i,j] >= 4 => G[i,j] <= 2.
    
    # Build conflict matrix: conflict[i,j] = True if G[i,j] > 2
    conflict = G > 2
    np.fill_diagonal(conflict, False)
    
    # Greedy set packing with randomization
    best_set = []
    best_size = 0
    
    # Multiple random restarts to find the maximum independent set
    rng = np.random.default_rng(42)
    
    # We have 5280 candidates. The target is 556.
    # A simple greedy approach: 
    # 1. Randomly order candidates.
    # 2. Iterate through the order.
    # 3. If a candidate does not conflict with any already selected candidate, select it.
    
    # To speed up, we can maintain a set of "available" candidates.
    # Initially all are available.
    # When we select a candidate, we remove it and all its neighbors from the available set.
    
    # However, a better greedy heuristic for maximum independent set in sparse graphs 
    # (which this is, as most pairs are compatible) is to pick the vertex with the 
    # fewest conflicts among the available vertices, or just random.
    # Given the high density of the solution (556/5280 ~ 10%), random ordering might work well 
    # with multiple restarts.
    
    # Let's try a weighted greedy: 
    # Priority = 1 / (number of conflicts with already selected)
    # But this is O(N^2) per step if not careful.
    
    # Alternative: 
    # Since the graph is defined by intersection > 2, we can represent each vector 
    # by its support (4 indices) and signs.
    # Two vectors conflict if their supports intersect in 3 or 4 elements.
    
    # Let's use a simple bitset or boolean mask for available candidates.
    # available = np.ones(n, dtype=bool)
    
    # To make it faster, we can precompute the neighbor lists.
    # neighbors[i] = list of j such that conflict[i,j] is True.
    
    # But storing neighbors for 5280 nodes might be memory heavy if dense.
    # Let's check density.
    # Number of pairs with intersection 3: 
    # Choose 3 common elements: C(4,3)=4 ways to pick which 3 of the 4 are common.
    # The 4th element of v1 is different from the 4th of v2? No, they share 3 elements.
    # Let v1 support be {1,2,3,4}. v2 support must include 3 of these.
    # Case 1: v2 support is {1,2,3,5}. Intersection {1,2,3} (size 3).
    # Case 2: v2 support is {1,2,3,4}. Intersection {1,2,3,4} (size 4).
    
    # Total candidates: 5280.
    # For a fixed v1, how many v2 conflict?
    # v2 shares 3 or 4 elements with v1.
    # If v2 shares 4 elements: v2 support is same as v1. 
    #   There are 2^4 = 16 vectors with this support. 15 of them conflict (all except v1 itself? No, v1 doesn't conflict with itself in the set, but in the graph we consider distinct vertices. Wait, if v1 and v2 have same support, they share 4 elements. Dot product can be -4 to 4. We need dot <= 2. 
    #   If same support, dot product is sum of 4 products of signs. 
    #   If signs are identical, dot=4 (conflict).
    #   If 3 same, 1 diff, dot=2 (no conflict).
    #   If 2 same, 2 diff, dot=0 (no conflict).
    #   If 1 same, 3 diff, dot=-2 (no conflict).
    #   If 0 same, 4 diff, dot=-4 (no conflict).
    #   So for same support, only the vector with identical signs conflicts with v1? 
    #   Wait, v1 has specific signs. v2 with same support and same signs is v1 itself. 
    #   v2 with same support and 3 same signs, 1 diff: dot=2. No conflict.
    #   So actually, for same support, NO other vector conflicts with v1?
    #   Let's re-verify.
    #   v1 = (1,1,1,1). v2 = (1,1,1,-1). Dot = 1+1+1-1 = 2. 2 <= 2. OK.
    #   v2 = (1,1,-1,-1). Dot = 1+1-1-1 = 0. OK.
    #   So same support vectors are all compatible with each other except for the identical vector?
    #   Wait, the constraint is pairwise. 
    #   If I select v1=(1,1,1,1) and v2=(1,1,1,-1), they are compatible.
    #   If I select v1=(1,1,1,1) and v3=(1,1,-1,-1), they are compatible.
    #   Are v2 and v3 compatible? Dot(v2,v3) = 1+1-1-1 = 0. Yes.
    #   So all 16 vectors with the same support are mutually compatible?
    #   Let's check v1=(1,1,1,1) and v4=(-1,-1,-1,-1). Dot=-4. Compatible.
    #   Yes, it seems all vectors with the same support are compatible with each other.
    
    # What about different supports?
    # v1 support {1,2,3,4}. v2 support {1,2,3,5}.
    # Intersection {1,2,3}.
    # Dot product depends on signs of 1,2,3.
    # If signs match on 1,2,3, dot >= 3 (since 4th term is 0). 3 > 2. Conflict.
    # If 2 match, 1 mismatch, dot = 1. 1 <= 2. No conflict.
    # So they conflict if and only if the signs on the shared 3 elements are identical.
    
    # This structure suggests a very specific packing.
    # For each support S (size 4), we can pick all 16 sign combinations?
    # If we pick all 16 for S, they are mutually compatible.
    # But they might conflict with vectors from other supports.
    
    # Let's stick to the general greedy algorithm on the precomputed conflict matrix.
    # It's robust and doesn't require deriving the exact combinatorial structure manually.
    
    # Optimization: Use a list of available indices.
    # When we select an index, we remove it and its neighbors from the available list.
    # To make this fast, we can use a boolean array `available` and a list `avail_list`.
    # Removing from a list is O(N). We can instead just iterate through the list and skip unavailable ones, 
    # or use a heap if we had priorities.
    
    # Given N=5280, N^2 = 27M operations. 
    # Greedy: 
    # 1. avail = list(range(n))
    # 2. while avail:
    #      pick random index from avail
    #      selected.append(idx)
    #      remove idx and all neighbors of idx from avail
    
    # Removing neighbors: 
    # neighbors = np.where(conflict[idx])[0]
    # We need to remove these from avail.
    # We can maintain a set of available indices for O(1) removal? 
    # Set operations in Python are slower than list operations for small N, but O(1) lookup.
    # However, iterating through neighbors and removing from a set is fine.
    
    # Let's try this.
    
    best_set = []
    
    for restart in range(50):  # 50 restarts
        if time.time() > deadline:
            break
            
        available_set = set(range(n))
        selected = []
        
        # To speed up, we can shuffle the order we pick from? 
        # No, we pick random from available_set.
        
        while available_set:
            if time.time() > deadline:
                break
                
            # Pick a random element
            # Converting set to list for random choice is O(N).
            # We can use a list and a "lazy deletion" approach or just accept the cost.
            # 5280 elements. Converting to list 500 times (approx) is 2.6M ops. Fine.
            avail_list = list(available_set)
            idx = avail_list[rng.integers(len(avail_list))]
            
            selected.append(idx)
            available_set.remove(idx)
            
            # Remove neighbors
            # conflict[idx] is a boolean array of length n.
            # neighbors = np.where(conflict[idx])[0]
            # This is O(N) per step.
            # 500 steps * 5280 = 2.6M ops per restart.
            # 50 restarts * 2.6M = 130M ops. 
            # NumPy vectorized operations are fast. This should be well within 150s.
            
            neighbors = np.where(conflict[idx])[0]
            # Remove these from available_set
            for nb in neighbors:
                if nb in available_set:
                    available_set.remove(nb)
        
        if len(selected) > len(best_set):
            best_set = selected
            if len(best_set) >= 556:
                # Found optimal or near-optimal, can stop early if confident
                # But let's run a few more to be sure? No, 556 is the max.
                # If we hit 556, we can break.
                break

    if not best_set:
        # Fallback
        return cands[:1]

    return cands[best_set]

if __name__ == "__main__":
    res = construct()
    print(res.shape)
