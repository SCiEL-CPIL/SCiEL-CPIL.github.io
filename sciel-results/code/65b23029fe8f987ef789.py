import numpy as np
import random
import time

N = 64
GRID_SIZE = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin

    # Precompute coordinates
    xs, ys = np.meshgrid(np.arange(N), np.arange(N))
    # Flatten to 1D arrays of coordinates
    # Each index i in [0, GRID_SIZE) corresponds to a point
    # We map index i to (x, y)
    x_coords = np.repeat(np.arange(N), N)
    y_coords = np.tile(np.arange(N), N)
    
    # Precompute squared distances between all pairs? 
    # That's GRID_SIZE x GRID_SIZE = 4096x4096 = 16M entries. 
    # 16M * 8 bytes = 128MB. Acceptable.
    # dist2[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
    
    # To save memory and time, we can compute on the fly or use a smaller subset?
    # No, vectorized update needs fast distance checks.
    # Let's compute the full distance matrix.
    # dx = x_coords[:, None] - x_coords[None, :]
    # dy = y_coords[:, None] - y_coords[None, :]
    # dist2 = dx*dx + dy*dy
    
    # This takes ~0.5s and 128MB.
    dx = x_coords[:, None] - x_coords[None, :]
    dy = y_coords[:, None] - y_coords[None, :]
    dist2 = dx*dx + dy*dy

    # We want to find a set of points S such that for all distinct u, v, w in S:
    # dist2[u,v] != dist2[u,w] and dist2[u,v] != dist2[v,w] and dist2[u,w] != dist2[v,w]
    # Actually, it's simpler:
    # For any pair (u,v) in S, there must be NO w in S such that:
    # 1. dist2[u,w] == dist2[v,w] (w on perp bisector)
    # 2. dist2[u,w] == dist2[u,v] (w on circle u)
    # 3. dist2[v,w] == dist2[u,v] (w on circle v)
    
    # Let's define a "conflict" for a potential new point w with respect to a set S.
    # w is invalid if there exist u,v in S such that the above holds.
    
    # Efficient Strategy: Simulated Annealing on a binary vector `active` of length GRID_SIZE.
    # Objective: Minimize total_conflicts, where total_conflicts is the number of isosceles triples in S.
    # If total_conflicts == 0, the set is valid.
    # We want to maximize |S| subject to total_conflicts == 0.
    
    # To make it a single optimization:
    # Score = -|S| * 1000 + total_conflicts
    # We want to minimize this.
    
    # How to calculate total_conflicts efficiently?
    # A triple (u,v,w) is isosceles if:
    # d(u,v) == d(u,w) OR d(u,v) == d(v,w) OR d(u,w) == d(v,w)
    # Note: This counts each isosceles triangle multiple times?
    # An isosceles triangle has exactly 2 equal sides (unless equilateral, which has 3).
    # If equilateral, it is counted 3 times in the sum of "pairs with equal distance to a third".
    # Let's define C(S) = sum over all distinct u,v,w in S of [d(u,v)==d(u,w)]
    # This is not quite right.
    
    # Alternative:
    # For each pair (u,v) in S, let K(u,v) be the number of w in S (w!=u,v) such that
    # d(u,w) == d(v,w) OR d(u,w) == d(u,v) OR d(v,w) == d(u,v).
    # If K(u,v) > 0, then the pair (u,v) is "bad".
    # A set is valid iff for all pairs (u,v) in S, K(u,v) == 0.
    # Total Bad Pairs = sum_{u<v in S} [K(u,v) > 0]
    
    # This is still hard to vectorize dynamically.
    
    # Let's try a different structural approach: **Greedy with Vectorized Pruning**.
    # We maintain a list of active points.
    # We maintain a "forbidden" mask for all grid points.
    # `forbidden[i]` is True if point i cannot be added to the current set.
    # When we add a point p:
    #   For every existing point q in the set:
    #     Find all r such that (p,q,r) is isosceles.
    #     Mark r as forbidden.
    #     Also, does p become forbidden? No, we just added it.
    #     But wait, if we add p, it might conflict with a *future* point?
    #     The constraint is symmetric.
    #     If we have set S and we want to add p, p is valid if p is not "forbidden" by S.
    #     How to update forbidden mask when adding p?
    #     For each q in S:
    #        The set of points r that form isosceles with p and q is:
    #        1. r on circle(p, |pq|)
    #        2. r on circle(q, |pq|)
    #        3. r on perp bisector of pq
    #     These r's become forbidden?
    #     NO. They are forbidden ONLY IF they are added.
    #     Actually, if r is in S, then (p,q,r) is isosceles, so p is INVALID.
    #     So, p is valid iff p is not in the union of "isosceles partners" of any pair (q1, q2) in S?
    #     No, p is valid iff for all q in S, p is not an isosceles partner of q with respect to ANY other point in S?
    #     This is getting complicated.
    
    # Let's stick to the **Vectorized Local Search** with a fast validity check.
    # We will use a **Bitset** representation of the active set.
    # But Python integers can be used as bitsets.
    
    # Let's try the **Conflict Graph** idea from the prompt, but implemented correctly.
    # We cannot solve Max Independent Set on a hypergraph easily.
    # However, we can use **Simulated Annealing** to minimize the number of isosceles triples.
    
    # Let's optimize the validity check for a single move (add/remove/swap).
    # Current set S (list of indices).
    # To check if index `new_idx` can be added:
    #   For each pair (u, v) in S:
    #      Check if (u, v, new_idx) is isosceles.
    #   This is O(|S|^2).
    
    # To speed this up, we can precompute for each pair (u,v) the list of "bad" points.
    # bad_points[u,v] = set of w such that (u,v,w) is isosceles.
    # Then new_idx is valid iff new_idx is not in bad_points[u,v] for any u,v in S.
    # This is still O(|S|^2) set lookups.
    
    # Let's try **Greedy with Random Restart** and **Vectorized Batch Check**.
    # We will generate many candidate sets and check them in parallel? No, too much memory.
    
    # **New Strategy: Vectorized Simulated Annealing**
    # State: `active` boolean array of size GRID_SIZE.
    # We maintain `conflict_count` array of size GRID_SIZE.
    # `conflict_count[i]` = number of isosceles triples involving i.
    # If we add point i, the new conflict count for j is:
    #   `conflict_count[j]` + (1 if (i, j, k) is isosceles for some k in S)
    # This is hard to update incrementally without iterating over k.
    
    # Let's go with a **High-Performance Greedy** using NumPy for the "is it valid?" check.
    # We will try to add points one by one.
    # For a candidate point `c`, we check validity against current set `S`.
    # `S` is a list of indices.
    # We extract coordinates of S: `S_coords` (k x 2).
    # `c_coords` (2,).
    # We compute distances from c to all points in S: `d_c_s` (k,).
    # We compute distances between all pairs in S: `d_s_s` (k, k).
    # We check if any `d_c_s[i] == d_c_s[j]` (i!=j) -> isosceles with c as apex.
    # We check if any `d_c_s[i] == d_s_s[i,j]` -> isosceles with base on S?
    #   Condition: d(c, s_i) == d(s_i, s_j).
    #   This means s_j is on circle(c, d(c,s_i)).
    #   We can check this by: for each i, is there a j such that d_s_s[i,j] == d_c_s[i]?
    
    # This check is O(k^2).
    # If k=20, k^2=400.
    # If we do 10000 candidates, 4M ops. NumPy handles this easily.
    
    # Let's implement a **Greedy with Backtracking** or **Local Search** using this vectorized check.
    
    def is_valid_vectorized(S_indices, c_idx, dist2_matrix):
        if len(S_indices) < 2:
            return True
        S = np.array(S_indices)
        c = c_idx
        # Distances from c to all in S
        d_cs = dist2_matrix[c, S]
        # Distances between all in S
        d_ss = dist2_matrix[np.ix_(S, S)]
        
        # Check 1: d(c, s_i) == d(c, s_j) for i != j
        # This is equivalent to: for any value v in d_cs, if it appears more than once, it's a conflict?
        # NO. If d(c, s_i) == d(c, s_j), then (c, s_i, s_j) is isosceles with c as apex.
        # So if any distance from c to S is repeated, it's invalid.
        # Note: d_cs contains distances to distinct points.
        unique_d, counts = np.unique(d_cs, return_counts=True)
        if np.any(counts > 1):
            return False
            
        # Check 2: d(c, s_i) == d(s_i, s_j)
        # For each i, we look at d_cs[i]. We check if this value exists in d_ss[i, :] (excluding j=i? no, j!=i)
        # Actually, d_ss[i, i] = 0. d_cs[i] > 0.
        # So we just check if d_cs[i] is in d_ss[i, :].
        # We can do this for all i at once?
        # d_ss is k x k. d_cs is k.
        # We want to know if for any i, d_ss[i, :] contains d_cs[i].
        # This is a row-wise membership test.
        # We can create a boolean mask: (d_ss == d_cs[:, None])
        # mask[i, j] is True if d_ss[i, j] == d_cs[i].
        # We need to exclude j=i?
        # If j=i, d_ss[i,i]=0. d_cs[i] > 0. So no match.
        # So we just check if any True in mask.
        mask = (d_ss == d_cs[:, None])
        if np.any(mask):
            return False
            
        # Check 3: d(c, s_j) == d(s_i, s_j)
        # This is symmetric to Check 2, just swapping i and j.
        # d_cs[j] == d_ss[i, j].
        # This is covered by the same mask logic if we consider the transpose?
        # mask2 = (d_ss == d_cs[None, :])
        # mask2[i, j] is True if d_ss[i, j] == d_cs[j].
        # If Check 2 passes, does Check 3 necessarily pass?
        # Check 2: d(c, s_i) == d(s_i, s_j).
        # Check 3: d(c, s_j) == d(s_i, s_j).
        # These are different conditions.
        # So we need to check both.
        mask2 = (d_ss == d_cs[None, :])
        if np.any(mask2):
            return False
            
        return True

    # Greedy Construction
    # Start with empty set.
    # Candidates: all points.
    # To make it faster, we can pre-shuffle candidates.
    
    def run_greedy(seed_offset=0):
        random.seed(seed_offset)
        S = []
        candidates = list(range(GRID_SIZE))
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            if is_valid_vectorized(S, c, dist2):
                S.append(c)
        return S

    # Local Search: Swap
    def run_local_search(S, time_limit=10.0):
        S = S[:]
        start = time.time()
        in_set = set(S)
        
        while time.time() < start + time_limit:
            if len(S) == 0: break
            i = random.randint(0, len(S) - 1)
            p_remove = S[i]
            
            # Try to find a replacement
            # Sample random candidates
            improved = False
            for _ in range(20):
                c = random.randint(0, GRID_SIZE - 1)
                if c in in_set:
                    continue
                # Check validity of c against S \ {p_remove}
                temp_S = S[:i] + S[i+1:]
                if is_valid_vectorized(temp_S, c, dist2):
                    S[i] = c
                    in_set.remove(p_remove)
                    in_set.add(c)
                    improved = True
                    break
            
            # Occasionally try to add
            if random.random() < 0.1:
                for _ in range(20):
                    c = random.randint(0, GRID_SIZE - 1)
                    if c in in_set:
                        continue
                    if is_valid_vectorized(S, c, dist2):
                        S.append(c)
                        in_set.add(c)
                        break
                        
        return S

    best_S = []
    
    # Multiple restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 30: # Leave time for local search
            break
            
        # Greedy phase
        S = run_greedy(seed_offset=r)
        
        # Local search phase
        S = run_local_search(S, time_limit=20.0)
        
        if len(S) > len(best_S):
            best_S = S
            
    return [(int(x_coords[i]), int(y_coords[i])) for i in best_S]

construct()
