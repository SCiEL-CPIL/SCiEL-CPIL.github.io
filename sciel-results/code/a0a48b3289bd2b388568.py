"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np

def construct():
    # We construct all integer vectors of weight 7 (exactly 7 non-zero entries, each +/- 1).
    # The norm squared is 7. Max norm R = sqrt(7).
    # For any two distinct such vectors u, v:
    # ||u - v||^2 = 7 + 7 - 2 u.v = 14 - 2 u.v
    # We need ||u - v|| >= sqrt(7) => 14 - 2 u.v >= 7 => 7 >= 2 u.v => u.v <= 3.5 => u.v <= 3.
    # The maximum dot product for distinct weight 7 vectors is 5 (if they share 5 indices with same sign and differ in 2? No).
    # If they share 6 indices: u.v = 6 - 2*1 = 4? 
    # Let u = e1+e2+e3+e4+e5+e6+e7.
    # Let v = e1+e2+e3+e4+e5+e6-e7. Dot = 5.
    # 5 <= 3.5 is FALSE.
    # So we CANNOT take all weight 7 vectors.
    
    # Let's re-evaluate the condition.
    # We need u.v <= floor(k_max / 2).
    # For k=7, floor(7/2) = 3.
    # Max dot product for distinct weight 7 vectors:
    # If they share 6 indices and 1 sign diff: Dot = 5.
    # 5 > 3. So pairs with dot product 5 or 6 are invalid.
    # We must prune the set.
    
    # This makes it a maximum independent set problem in a graph of 42k nodes.
    # Too slow for 150s? Maybe.
    
    # Let's check k=5.
    # R = sqrt(5).
    # Condition: 10 - 2 u.v >= 5 => 5 >= 2 u.v => u.v <= 2.5 => u.v <= 2.
    # Max dot product for distinct weight 5 vectors:
    # Share 4 indices, 1 sign diff: Dot = 3.
    # 3 > 2. Invalid.
    # Share 4 indices, same signs: u=v.
    # So we must prune pairs with dot product 3.
    
    # Let's check k=3.
    # R = sqrt(3).
    # Condition: 6 - 2 u.v >= 3 => 3 >= 2 u.v => u.v <= 1.5 => u.v <= 1.
    # Max dot product for distinct weight 3 vectors:
    # Share 2 indices, 1 sign diff: Dot = 1.
    # Share 2 indices, same signs: u=v (if 3rd index same) or different index?
    # If u=e1+e2+e3, v=e1+e2+e4. Dot = 2.
    # 2 > 1. Invalid.
    # So we must prune pairs with dot product 2.
    
    # It seems pruning is required for all k.
    # The seed (k=2) has max dot product 1.
    # Condition for k=2: u.v <= 1.
    # Max dot product for distinct weight 2 vectors is 1.
    # So NO pruning is needed for k=2.
    # This is why the seed works perfectly.
    
    # To beat the seed, we need a set with more vectors, even after pruning.
    # For k=3, we remove pairs with dot product 2.
    # Number of such pairs is manageable.
    
    # Let's try a greedy approach for k=3 or k=4.
    # Or simply return the seed (220) if pruning is too complex?
    # But 1320 is much larger. Even with 50% pruning, it's better.
    
    # Let's implement a greedy maximal independent set for weight 3 vectors.
    
    vectors = []
    # Generate all weight 3 vectors
    all_vecs = []
    for indices in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(indices, signs):
                v[idx] = s
            all_vecs.append(v)
            
    # Greedy Independent Set
    # We want to select a subset S such that u.v <= 1 for all u, v in S.
    # Note: u.v can be -3, -1, 1, 3.
    # u.v = 3 implies u=v.
    # u.v = 1 is allowed.
    # u.v = -1 is allowed.
    # u.v = -3 is allowed.
    # Wait, condition was u.v <= 1.
    # So u.v = 2 is the only forbidden value for distinct vectors?
    # Let's check if u.v can be 2.
    # u=e1+e2+e3, v=e1+e2+e4. Dot = 2. Yes.
    # So we just need to avoid pairs with dot product 2.
    
    selected = []
    used = set()
    
    # To speed up, we can just iterate and check against selected.
    # 1320 vectors is small.
    
    for v in all_vecs:
        valid = True
        for u in selected:
            dot = np.dot(u, v)
            if dot > 1:
                valid = False
                break
        if valid:
            selected.append(v)
            
    return np.array(selected, dtype=np.int64)

# EVOLVE-BLOCK-END
