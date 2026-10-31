import numpy as np
import time

D = 11

def construct():
    t0 = time.time()
    deadline = 120.0
    
    # Generate candidate vectors: all nonzero integer vectors with entries in {-1, 0, 1}
    # and weight (number of nonzeros) from 1 to some max
    # To keep it manageable, let's focus on weight 2, 3, 4 vectors
    # Weight 2: C(11,2) * 4 = 55 * 4 = 220
    # Weight 3: C(11,3) * 8 = 165 * 8 = 1320
    # Weight 4: C(11,4) * 16 = 330 * 16 = 5280
    # Total ~6820 candidates
    
    candidates = []
    for w in range(2, 5):  # weights 2, 3, 4
        from itertools import combinations, product
        for support in combinations(range(D), w):
            for signs in product([1, -1], repeat=w):
                v = [0] * D
                for idx, s in zip(support, signs):
                    v[idx] = s
                candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Compute dot product matrix
    dots = candidates @ candidates.T
    
    # For a set of vectors all with the same norm R, constraint: dots[i,j] <= R^2/2
    # But we can have mixed norms. The constraint is: ||v_i - v_j|| >= max_norm
    # i.e., ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j >= max_norm^2
    # This is complex with mixed norms. Let's stick with uniform norm for simplicity.
    # Use weight 2 vectors (norm^2 = 2): constraint dots <= 1
    # Or weight 3 (norm^2 = 3): constraint dots <= 1.5, so dots <= 1 for integers
    # Or weight 4 (norm^2 = 4): constraint dots <= 2
    
    # Let's try weight 4 vectors: norm^2 = 4, constraint dots[i,j] <= 2
    # This allows more flexibility
    
    # Actually, let me try a mixed approach: use all weight 2,3,4 and check the actual constraint
    # For a selected set S with max_norm_sq = max(||v||^2 for v in S):
    # For all i,j in S, i!=j: ||v_i - v_j||^2 >= max_norm_sq
    # i.e., norms_sq[i] + norms_sq[j] - 2*dots[i,j] >= max_norm_sq
    
    # This is complex. Let me use uniform weight for simplicity and efficiency.
    
    # Strategy: use weight 3 vectors (norm^2 = 3, constraint dots <= 1 for integer dots)
    # Actually for weight 3, dots can be -3,-2,-1,0,1,2,3. Constraint: dots <= 1.
    
    # Let me just do greedy on all candidates with the proper constraint.
    # Precompute compatibility: two vectors i,j are compatible if
    # norms_sq[i] + norms_sq[j] - 2*dots[i,j] >= max(norms_sq[i], norms_sq[j])
    # i.e., min(norms_sq[i], norms_sq[j]) >= 2*dots[i,j] - max(norms_sq[i], norms_sq[j])
    # This is getting complex. Let me simplify by using uniform norm.
    
    # Use weight 2 vectors only: 220 candidates, norm^2 = 2, constraint dots <= 1
    # The seed already does this and gets 220 - (forbidden pairs)
    # Forbidden: dots = 2 (same vector) - but we don't have duplicates
    # Actually dots=2 means same signs on same support. dots=1 means 3/4 agreement.
    # For weight 2: dots can be -2,-1,0,1,2. dots=2 only if identical.
    # So all 220 weight-2 vectors are mutually compatible! That's the seed's 220.
    
    # To do better, I need to add more vectors. Let me try adding weight 3 vectors.
    # A weight-3 vector has norm^2=3. With a weight-2 vector (norm^2=2):
    # Constraint: 2+3-2*dots >= 3 => dots <= 1
    # With another weight-3 vector: 3+3-2*dots >= 3 => dots <= 1.5 => dots <= 1
    
    # So if I use weight 2 and weight 3 vectors, the constraint between any two is dots <= 1.
    # Let me generate weight 2 and weight 3 vectors and greedily select.
    
    candidates2 = []
    from itertools import combinations, product
    for w in [2, 3]:
        for support in combinations(range(D), w):
            for signs in product([1, -1], repeat=w):
                v = [0] * D
                for idx, s in zip(support, signs):
                    v[idx] = s
                candidates2.append(v)
    
    candidates2 = np.array(candidates2, dtype=np.int64)
    n_cand2 = len(candidates2)
    dots2 = candidates2 @ candidates2.T
    
    # Constraint: for all i != j, dots2[i,j] <= 1
    # (since min norm^2 = 2, max norm^2 = 3, and 2+3-2*1=3 >= 3, 2+2-2*1=2>=2, 3+3-2*1=4>=3)
    
    # Greedy: repeatedly pick the vector with the most compatible unselected neighbors
    def greedy_select(dots, n, max_dot=1, time_limit=None):
        selected = []
        remaining = set(range(n))
        # Precompute compatibility
        compat = (dots <= max_dot).astype(np.uint8)
        np.fill_diagonal(compat, 0)
        
        while remaining:
            if time_limit and time.time() > time_limit:
                break
            # Pick the vector with the most compatible neighbors in remaining
            best = None
            best_count = -1
            for v in remaining:
                count = sum(compat[v][u] for u in remaining if u != v)
                if count > best_count:
                    best_count = count
                    best = v
            selected.append(best)
            remaining.discard(best)
            # Remove incompatible vectors
            to_remove = set()
            for u in remaining:
                if not compat[best][u]:
                    to_remove.add(u)
            remaining -= to_remove
        return selected
    
    # This greedy is O(n^3) which is too slow for 1540 candidates.
    # Let me use a faster approach: sort by norm, then greedily add.
    
    # Better approach: use the fact that for weight 2 vectors, all are compatible.
    # Start with all 220 weight-2 vectors, then try to add weight-3 vectors.
    
    w2_indices = list(range(220))  # first 220 are weight 2
    w3_indices = list(range(220, n_cand2))  # next 1320 are weight 3
    
    # All weight-2 vectors are mutually compatible (dots <= 1 always for distinct weight-2)
    # Check: two distinct weight-2 vectors can have dots = 2 only if identical.
    # So yes, all 220 weight-2 are compatible with each other.
    
    # For a weight-3 vector v3, it's compatible with weight-2 vector v2 if dots[v3,v2] <= 1
    # And with other weight-3 vectors v3' if dots[v3,v3'] <= 1
    
    # Greedily add weight-3 vectors that are compatible with all selected weight-2 vectors
    # and with previously added weight-3 vectors.
    
    selected = set(w2_indices)
    
    # For each weight-3 vector, check if it's compatible with all selected
    # This is O(1320 * 220) = ~290k checks, fast.
    
    added_w3 = []
    for v3 in w3_indices:
        if time.time() > deadline - 5:
            break
        # Check compatibility with all selected
        compatible = True
        for v_sel in selected:
            if dots2[v3, v_sel] > 1:
                compatible = False
                break
        if compatible:
            selected.add(v3)
            added_w3.append(v3)
    
    result = candidates2[list(selected)]
    return result
