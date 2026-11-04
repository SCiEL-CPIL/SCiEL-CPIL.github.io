import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = 145  # 5s margin before 150s limit
    
    # We work with 4-sparse vectors in R^11.
    # Two such vectors conflict (distance < max_norm) iff they share 3 or 4 non-zero positions with the same signs.
    # For 4-sparse vectors, sharing 4 positions with same signs means they are identical.
    # Sharing 3 positions with same signs means they differ in exactly one position (one has a sign, the other doesn't, or different signs? No, both are 4-sparse).
    # Let v and w be two 4-sparse vectors.
    # dist^2 = ||v-w||^2.
    # If they share k positions with same sign, (4-k) positions with opposite sign (impossible if both sparse at same support? No, supports can differ).
    # Let S_v and S_w be the supports.
    # Let I = S_v ∩ S_w.
    # For i in I, v_i and w_i are ±1.
    # If v_i == w_i, contribution to dist^2 is 0.
    # If v_i != w_i, contribution is 4.
    # For i in S_v \ S_w, v_i^2 = 1, w_i=0, contribution 1.
    # For i in S_w \ S_v, w_i^2 = 1, v_i=0, contribution 1.
    # dist^2 = 4 * (number of sign flips in I) + |S_v \ S_w| + |S_w \ S_v|.
    # We require dist^2 >= 4 (since max norm squared is 4).
    # Actually, the task says "minimum pairwise distance is at least the maximum vector norm".
    # Norm of each vector is 2. So dist >= 2, i.e., dist^2 >= 4.
    
    # Let m = |I|.
    # Let f = number of sign flips in I.
    # |S_v \ S_w| = 4 - m.
    # |S_w \ S_v| = 4 - m.
    # dist^2 = 4f + 2(4-m) = 4f + 8 - 2m.
    # We need 4f + 8 - 2m >= 4 => 4f + 4 >= 2m => 2f + 2 >= m.
    
    # Case m=4: S_v = S_w. Then 2f + 2 >= 4 => 2f >= 2 => f >= 1.
    #   If f=0 (identical), dist=0. Conflict.
    #   If f>=1, dist^2 = 4f + 0 >= 4. OK.
    #   So two vectors with same support are OK unless they are identical.
    #   Wait, if f=1, dist^2 = 4. OK.
    #   If f=2, dist^2 = 8. OK.
    #   So on the same support, we can pick at most 2 vectors?
    #   There are 16 sign combinations on a support.
    #   v and w conflict iff f=0 (identical).
    #   So we can pick ALL 16 vectors on a support?
    #   Let's check: v=(1,1,1,1), w=(1,1,1,-1). f=1. dist^2=4. OK.
    #   v=(1,1,1,1), w=(1,1,-1,-1). f=2. dist^2=8. OK.
    #   v=(1,1,1,1), w=(-1,-1,-1,-1). f=4. dist^2=16. OK.
    #   So YES, on a single support, we can include all 16 sign variants.
    
    # Case m=3: |S_v \ S_w|=1, |S_w \ S_v|=1.
    #   2f + 2 >= 3 => 2f >= 1 => f >= 1.
    #   If f=0 (same signs on the 3 shared positions), dist^2 = 0 + 2 = 2 < 4. CONFLICT.
    #   If f>=1, dist^2 >= 4 + 2 = 6. OK.
    #   So if two supports intersect in 3, they conflict ONLY IF the signs on the intersection are identical.
    
    # Case m=2: 2f + 2 >= 2 => always true.
    #   dist^2 = 4f + 4 >= 4. OK.
    #   So supports intersecting in 2 never conflict.
    
    # Case m=1: 2f + 2 >= 1 => always true.
    #   dist^2 = 4f + 6 >= 6. OK.
    
    # Case m=0: always true.
    #   dist^2 = 8. OK.
    
    # Conclusion:
    # The only conflicts are:
    # 1. Identical vectors (same support, same signs). Trivial, we pick distinct vectors.
    # 2. Two vectors with supports intersecting in exactly 3, where the signs on the intersection are identical.
    
    # This means:
    # We can pick all 16 sign variants for any set of supports, PROVIDED that for any pair of supports with intersection size 3, we do NOT pick a pair of vectors whose signs match on the intersection.
    # Wait, this is a global constraint.
    # Let S be a set of supports.
    # For each support s in S, we want to pick a subset of its 16 sign vectors.
    # Ideally we pick all 16.
    # If we pick all 16 for s1 and all 16 for s2 (|s1 ∩ s2|=3), is there a conflict?
    # Yes. Pick v in s1 with signs (1,1,1) on intersection, and w in s2 with signs (1,1,1) on intersection.
    # Then f=0 on intersection. dist^2=2. Conflict.
    # So we CANNOT pick all 16 for both s1 and s2.
    
    # This looks like a complex constraint satisfaction problem.
    # However, note the structure.
    # The conflict graph between (support, sign) pairs is dense.
    
    # Alternative perspective:
    # This is equivalent to finding a large code in the Johnson graph?
    # No, the "signs" add a layer.
    
    # Let's simplify.
    # If we restrict ourselves to picking AT MOST 1 vector per support, the problem reduces to:
    # Find a large set of 4-subsets of {1..11} such that no two subsets intersect in 3?
    # No, if we pick 1 vector per support, we can choose the signs to avoid conflicts.
    # If s1 and s2 intersect in 3, we just need to choose signs such that they differ on at least one of the 3 intersection points.
    # This is always possible unless the constraints from other supports force a specific sign pattern.
    
    # Let's try a different structural approach.
    # Use the "Type D" construction mentioned in inspirations?
    # "16 per support" suggests that maybe we CAN get close to 16 per support on average?
    # Or maybe the inspiration meant something else.
    
    # Let's look at the "kissing number in dimension 11".
    # Known kissing number for d=11 is 504? No, that's not right.
    # Kissing numbers:
    # d=1: 2
    # d=2: 6
    # d=3: 12
    # d=4: 24
    # d=5: 40
    # d=6: 72
    # d=7: 126
    # d=8: 240
    # d=9: 460 (lower bound)
    # d=10: 900 (lower bound)
    # d=11: ?
    
    # The seed program found 330.
    # The inspiration says 2.5455 score. Seed is 1.0.
    # If seed score is 330, then 2.5455 * 330 = 840.
    # So we need to find ~840 vectors.
    
    # If we can pick ~24 vectors per support on average, and we have ~35 supports, that's 840.
    # Total supports = C(11,4) = 330.
    # If we pick a subset of supports of size K, and for each support we pick N vectors, total = K*N.
    # But supports interact.
    
    # Let's try a greedy approach with randomization and local search.
    # Represent the state as a list of vectors.
    # Start with a random valid configuration.
    # Try to add more vectors.
    # Try to swap vectors to allow more additions.
    
    # To speed up conflict checking, we can precompute the "signature" of each vector.
    # A vector is defined by (support, signs).
    # Conflict with another vector (support2, signs2) iff:
    # 1. support == support2 and signs == signs2 (identical)
    # 2. |support ∩ support2| == 3 and signs on intersection are identical.
    
    # We can maintain a set of "forbidden" (support, signs) tuples.
    # When we add a vector v with support S and signs sig:
    # It forbids:
    # - (S, sig) itself.
    # - For any support T with |S ∩ T| == 3:
    #   Let I = S ∩ T.
    #   Let sig_I = sig restricted to I.
    #   Then any vector (T, sig_T) where sig_T restricted to I == sig_I is forbidden.
    #   How many such sig_T?
    #   T has 4 positions. 3 are in I, 1 is in T \ S.
    #   The signs on I are fixed. The sign on T \ S can be ±1.
    #   So 2 forbidden vectors per such T.
    
    # Number of T with |S ∩ T| == 3:
    # S has 4 elements. T must contain 3 of them and 1 new element.
    # Choose 3 from S: C(4,3)=4 ways.
    # Choose 1 from remaining 7: 7 ways.
    # Total 28 such T.
    # So adding one vector forbids 1 + 28*2 = 57 vectors.
    
    # Total possible vectors: 330 * 16 = 5280.
    # If we pick K vectors, we forbid ~57K vectors.
    # 57K <= 5280 => K <= 92.
    # This is a very loose upper bound. It suggests we can't pick many if we use this "forbidden" logic naively because the forbidden sets overlap heavily.
    
    # Actually, the forbidden sets are not disjoint.
    # This approach is too pessimistic.
    
    # Let's go back to the "1 vector per support" idea.
    # If we pick 1 vector per support, we have 330 vectors.
    # Can we pick more?
    # If we pick 2 vectors per support, we have 660.
    # Conflict condition:
    # If s1 and s2 intersect in 3, we cannot have v1 in s1 and v2 in s2 such that they match on the intersection.
    # If we pick a random pair of vectors for each support, what is the probability of conflict?
    # For a fixed pair of supports (s1, s2) with intersection 3:
    # There are 16*16 = 256 pairs of vectors.
    # How many conflict?
    # We need signs on intersection to match.
    # Intersection has 3 positions. Signs must match.
    # The 4th position in s1 can be anything (2 choices).
    # The 4th position in s2 can be anything (2 choices).
    # So 2*2 = 4 pairs of "signs on non-intersection" are free.
    # Wait, the signs on the intersection are fixed to match.
    # There are 2^3 = 8 possible sign patterns on the intersection.
    # For each pattern, there are 2*2=4 pairs of vectors (one from each support) that have this pattern on the intersection.
    # Total conflicting pairs = 8 * 4 = 32.
    # So 32 out of 256 pairs conflict. Probability = 32/256 = 1/8.
    
    # If we pick 2 vectors per support, and we have ~35 supports involved in a "cluster", the number of conflicting pairs might be high.
    # But we can use local search to resolve conflicts.
    
    # Strategy:
    # 1. Start with 1 vector per support (randomly chosen). 330 vectors.
    # 2. Try to add a second vector to each support.
    # 3. Use simulated annealing or local search to remove conflicts.
    
    # Implementation:
    # State: dict mapping support (tuple of 4 indices) to list of sign tuples.
    # Initially, each support has 1 random sign tuple.
    # We want to maximize total vectors.
    # Objective: maximize sum of len(signs[s]) for all s.
    # Constraint: no conflicts.
    
    # Conflict check:
    # For each pair of supports (s1, s2) with |s1 ∩ s2| == 3:
    #   For each v1 in signs[s1] and v2 in signs[s2]:
    #     Check if signs match on intersection.
    
    # This is O(N^2) per step if done naively.
    # We can optimize by maintaining a count of conflicts.
    
    # Let's try a simpler greedy first:
    # Sort supports by degree in the "intersection-3" graph.
    # Process in order.
    # For each support, try to add as many vectors as possible.
    
    # Actually, let's just use a random greedy with restarts.
    # 1. Generate all 5280 vectors.
    # 2. Shuffle them.
    # 3. Iterate through shuffled vectors.
    # 4. If a vector doesn't conflict with any already selected vector, select it.
    # 5. Repeat with different shuffles, keep best.
    
    # To make it fast, we need an efficient conflict check.
    # We can precompute for each vector a list of vectors it conflicts with.
    # But 5280 * 5280 is 27M pairs. Too many to precompute fully?
    # 27M is actually manageable in C, but in Python it's slow.
    
    # Let's use the "forbidden" idea but dynamically.
    # Maintain a set `selected`.
    # Maintain a set `forbidden` which contains vectors that conflict with any selected vector.
    # When we select v:
    #   Add v to `selected`.
    #   Compute all vectors that conflict with v.
    #   Add them to `forbidden`.
    #   If any of them are in `selected`, we have a conflict (shouldn't happen if we only select from non-forbidden).
    
    # Wait, if we only select from `all_vectors - forbidden`, then we never select a conflicting vector.
    # This is a valid greedy independent set algorithm.
    # The quality depends on the order.
    
    # Let's implement this.
    
    all_vectors = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(indices, signs):
                v[idx] = s
            all_vectors.append((indices, signs, v))
            
    # Precompute conflict lists?
    # For each vector i, find all j that conflict with i.
    # Conflict if:
    # 1. i == j
    # 2. |S_i ∩ S_j| == 3 and signs match on intersection.
    
    # Let's build a graph.
    # Nodes: 0..5279.
    # Edges: conflicts.
    
    # We can build the adjacency list.
    # For each i:
    #   S_i, sig_i
    #   For each j > i:
    #     Check conflict.
    
    # This is O(N^2) = 27M checks.
    # Each check is fast (tuple intersection).
    # 27M * ~100ns = 2.7s. Acceptable.
    
    # But we don't need the full graph for greedy.
    # We just need to know, for a candidate vector, if it conflicts with ANY selected vector.
    
    # Let's stick to the dynamic forbidden set.
    # `forbidden` is a set of vector indices.
    # `selected` is a list of vector indices.
    
    # When we select v_i:
    #   For each j in all_vectors:
    #     if j conflicts with i:
    #       forbidden.add(j)
    
    # This is O(N) per selection.
    # If we select K vectors, total O(K*N).
    # K ~ 500, N ~ 5280. 500 * 5280 = 2.6M operations. Very fast.
    
    # So the bottleneck is the conflict check itself.
    # We can precompute the conflict list for each vector.
    # `conflicts[i]` = list of j that conflict with i.
    
    # Let's do that.
    
    N = len(all_vectors)
    conflicts = [[] for _ in range(N)]
    
    for i in range(N):
        S_i, sig_i, _ = all_vectors[i]
        for j in range(i+1, N):
            S_j, sig_j, _ = all_vectors[j]
            # Check intersection size
            inter = set(S_i).intersection(S_j)
            if len(inter) == 3:
                # Check signs
                # Map intersection indices to positions in S_i and S_j
                # This is a bit slow.
                # Let's precompute sign tuples as dicts? No.
                # Just compare.
                match = True
                for idx in inter:
                    pos_i = S_i.index(idx)
                    pos_j = S_j.index(idx)
                    if sig_i[pos_i] != sig_j[pos_j]:
                        match = False
                        break
                if match:
                    conflicts[i].append(j)
                    conflicts[j].append(i)
            elif len(inter) == 4:
                # Same support
                if sig_i == sig_j:
                    conflicts[i].append(j)
                    conflicts[j].append(i)
                # If same support but different signs, no conflict (dist^2 >= 4)
                # Wait, earlier I said f>=1 is OK.
                # If sig_i != sig_j, then f >= 1. So no conflict.
    
    # Now run greedy with multiple random orders.
    best_selected = []
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        order = list(range(N))
        random.shuffle(order)
        
        selected = []
        forbidden = set()
        
        for i in order:
            if i in forbidden:
                continue
            # Check if i conflicts with any selected?
            # If we maintain `forbidden` correctly, i is not forbidden, so it doesn't conflict with any selected.
            # Wait, `forbidden` contains vectors that conflict with selected ones.
            # So if i is not in `forbidden`, it is safe to add.
            selected.append(i)
            for j in conflicts[i]:
                forbidden.add(j)
        
        if len(selected) > len(best_selected):
            best_selected = selected
            
    # Convert best_selected to array
    if not best_selected:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    result = np.array([all_vectors[i][2] for i in best_selected], dtype=np.int64)
    return result

construct()
