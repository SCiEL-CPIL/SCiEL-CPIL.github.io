import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580.0  # Safety margin before 630s kill

    # ---------- Base: All weight-2 vectors (220 points, known valid) ----------
    # This serves as the fallback and initial lower bound.
    base_list = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    base_list.append(p)
    base_points = np.array(base_list, dtype=np.int64)
    best_points = base_points.copy()
    best_k = base_points.shape[0]  # 220

    # ---------- Candidate Generation ----------
    # We expand the pool to include vectors of weight 3, 4, and 5.
    # Weight w means w non-zero entries, each +/-1. Norm^2 = w.
    # w=2: 220 points (base)
    # w=3: C(11,3) * 2^3 = 165 * 8 = 1320 points
    # w=4: C(11,4) * 2^4 = 330 * 16 = 5280 points
    # w=5: C(11,5) * 2^5 = 462 * 32 = 14784 points
    # Total N = 220 + 1320 + 5280 + 14784 = 21604 points.
    # This is large for a dense matrix (21604^2 ~ 464M entries), but manageable in blocks or with sparse representation.
    # However, for B&B, we need fast neighbor queries. Bitsets are ideal.
    # Python integers can be used as arbitrary-precision bitsets.
    
    def generate_candidates(max_weight):
        cands = []
        for w in range(2, max_weight + 1):  # Start from 2 to include base
            for combo in combinations(range(d), w):
                for mask in range(1 << w):
                    v = np.zeros(d, dtype=np.int64)
                    for idx, pos in enumerate(combo):
                        v[pos] = 1 if (mask >> idx) & 1 else -1
                    cands.append(v)
        return np.array(cands, dtype=np.int64)

    # Generate pool up to weight 5
    cands = generate_candidates(5)
    n = cands.shape[0]
    norms = (cands * cands).sum(axis=1).astype(np.int64)

    # ---------- Conflict Graph Construction ----------
    # Two points i, j conflict if dist^2(i, j) < max(norms[i], norms[j]).
    # dist^2(i, j) = norms[i] + norms[j] - 2 * dot(i, j).
    # Condition for conflict: norms[i] + norms[j] - 2*dot(i, j) < max(norms[i], norms[j]).
    
    # We will build adjacency lists as Python integers (bitsets) for fast operations in B&B.
    # adj[i] is a bitmask where bit j is 1 if i and j conflict.
    
    adj = [0] * n
    time_start = time.time()
    
    # Vectorized dot product calculation in blocks to manage memory and speed
    # Block size 1000
    block_size = 1000
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = cands[start:end]
        # Compute dot products of block with all cands
        dots = block @ cands.T  # shape (end-start, n)
        norms_block = norms[start:end]
        
        # dist_sq matrix for this block
        # dist_sq[i, j] = norms_block[i] + norms[j] - 2 * dots[i, j]
        dist_sq = norms_block[:, None] + norms[None, :] - 2 * dots
        
        # required distance squared
        required = np.maximum(norms_block[:, None], norms[None, :])
        
        # Conflict mask
        conflict_mask = dist_sq < required
        
        # Convert to bitsets
        # For each row i in the block, we need to set bits in adj[start+i]
        # We can use a loop here for the bitset conversion as numpy doesn't have direct
        # "row to int bitmask" for arbitrary size, but we can use a helper.
        
        for i in range(end - start):
            row_mask = conflict_mask[i]
            # Convert boolean array to integer bitmask
            # This is the bottleneck. We need an efficient way.
            # Using np.packbits? No, we need Python int.
            # A simple loop over True indices is O(k) where k is degree.
            # Given the sparsity of conflicts (high dimensionality), k should be small.
            # Let's verify sparsity.
            # For random vectors in {-1,0,1}^11, dot products are small.
            # dist^2 is large. Conflicts are rare.
            
            # Find indices where conflict is True
            # Exclude self (i == j) which is always conflict (dist 0 < norm)
            # But in our logic, we want the independent set, so self-conflict is fine.
            # We just need the neighbors.
            
            # Efficient conversion:
            # If n is ~21k, we can't easily use a single numpy op to int.
            # Let's iterate over True values.
            true_indices = np.where(row_mask)[0]
            # Remove self if present
            # self_index = start + i
            # if self_index in true_indices: remove it
            # But we can just build the int.
            
            bit_int = 0
            # To speed up, we can process in chunks or just loop.
            # Since conflicts are sparse, this loop is fast.
            for idx in true_indices:
                if idx != (start + i): # Skip self
                    bit_int |= (1 << idx)
            
            adj[start + i] = bit_int

        if time.time() > deadline - 100:
            break

    if time.time() > deadline - 100:
        return np.array(best_points, dtype=np.int64)

    # ---------- Branch and Bound for Maximum Independent Set ----------
    # We want to find a set of vertices S such that no two vertices in S are adjacent in the Conflict Graph.
    # This is equivalent to a Clique in the Compatibility Graph, or MIS in Conflict Graph.
    
    # State:
    # P: Set of candidate vertices (bitmask)
    # S: Current independent set (list of vertices)
    
    # Reductions:
    # 1. If P is empty, update best.
    # 2. If |P| + |S| <= best_k, prune.
    # 3. Domination: If v in P is dominated by u in P (N(u) \subseteq N(v) \cup {u}), 
    #    then v can be removed from P without loss of optimality (in terms of finding a max set, 
    #    though we must be careful with weights, but here unweighted).
    #    Actually, standard domination: If N[v] \subseteq N[u], then v is dominated by u.
    #    For MIS, if u dominates v, we can remove v from P.
    
    # Let's implement a simple recursive B&B with bitsets.
    
    # Precompute neighborhoods as bitsets for fast intersection.
    # N[v] = adj[v] | (1 << v)
    closed_neigh = [adj[i] | (1 << i) for i in range(n)]
    
    # Initial P is all vertices
    # We need to be careful with recursion depth. Python recursion limit is low.
    # We can use an iterative approach or increase recursion limit.
    # Given n=21k, depth could be large. But B&B prunes heavily.
    # Let's use a stack-based approach or just increase recursion limit and hope for the best.
    # Better: Use a greedy coloring bound on the remaining graph P to prune.
    # Coloring the conflict graph P gives an upper bound on the size of the MIS in P.
    
    # Greedy Coloring for Upper Bound:
    # We need a fast heuristic coloring.
    # Since the graph is sparse, we can use a simple greedy coloring.
    
    def greedy_color_bound(P_mask):
        # P_mask is a bitmask of remaining vertices
        # We want to color the subgraph induced by P_mask.
        # Number of colors is an upper bound on clique size in compat graph?
        # No, we are solving MIS in Conflict Graph.
        # MIS in G is equivalent to Clique in G_complement.
        # Coloring G gives lower bound on Clique in G_complement? No.
        # Let's stick to MIS directly.
        # Upper bound for MIS: 
        # A simple bound is the number of vertices.
        # A tighter bound: Greedy coloring of the Conflict Graph.
        # Wait, Coloring the Conflict Graph gives a partition into independent sets in Conflict Graph?
        # No, Coloring G partitions V into independent sets in G.
        # So the size of a color class is an independent set in G.
        # The minimum number of colors (chromatic number) is a lower bound on the size of the *maximum* independent set? No.
        # Actually, if we color G, each color class is an independent set in G.
        # So the size of the *largest* color class is a lower bound on the MIS size.
        # This is not an upper bound for pruning.
        
        # We need an upper bound on the MIS size in the subgraph P.
        # A common upper bound is the clique cover number of the complement? Too complex.
        # Simpler: The maximum degree in P + 1 is an upper bound on the clique size in P_complement?
        # Let's use a simple bound: The number of vertices in P is a trivial upper bound.
        # A better one: If we can find a vertex v in P with high degree in P, 
        # then MIS(P) <= 1 + MIS(P \ N[v]).
        # This is the branching step.
        
        # For pruning, we can use the bound: |MIS(P)| <= |P|.
        # To improve, we can use a greedy packing bound or similar.
        # Given time constraints, let's rely on the branching and the initial lower bound from greedy.
        
        return np.bit_count(P_mask)

    # Initialize best solution with a greedy MIS
    # Greedy: Pick vertex with max degree in Conflict Graph? No, pick vertex with min degree in Conflict Graph (max compatibility).
    # Or just random.
    
    # Let's do a quick greedy to get a good initial lower bound.
    # Order vertices by degree in Conflict Graph (ascending).
    degrees = [np.bit_count(adj[i]) for i in range(n)]
    order = np.argsort(degrees)
    
    current_set = []
    P_mask = (1 << n) - 1  # All vertices
    
    for v in order:
        if time.time() > deadline - 50:
            break
        # Check if v is in P
        if not (P_mask & (1 << v)):
            continue
        # Check if v is independent of current_set
        # v is independent of current_set if adj[v] & mask(current_set) == 0
        # mask(current_set) is not a single int if current_set is a list.
        # Let's maintain P_mask and remove neighbors.
        
        # If v is in P, it is not adjacent to any vertex in current_set 
        # (because we removed neighbors when we added to current_set).
        # So we can add v.
        current_set.append(v)
        # Remove v and its neighbors from P
        P_mask &= ~(adj[v] | (1 << v))
        
    best_k = len(current_set)
    best_set = current_set.copy()
    
    # B&B
    # We will use a recursive function, but to avoid recursion limits, 
    # we'll implement a simple depth-first search with a stack.
    # State: (P_mask, current_set_size, current_set_list)
    # Storing current_set_list in the stack is memory heavy.
    # Instead, we can just track the set of vertices added in the path.
    # Or, we can use the property that we only need the size for the bound, 
    # but we need the actual set to return the best solution.
    # Let's store the list in the stack.
    
    stack = [(P_mask, 0, [])]
    # Note: The initial P_mask here should be the full set, and we restart the search from scratch?
    # No, we want to improve on the greedy solution.
    # We can start the B&B from the state after the greedy selection?
    # No, B&B needs to explore alternatives.
    # Let's start from the root: P = All, S = [].
    
    # Re-initialize stack for full search
    P_full = (1 << n) - 1
    stack = [(P_full, 0, [])]
    
    # To make it faster, we can use the greedy solution as the initial best.
    # And we can prune branches where |S| + |P| <= best_k.
    
    # We also need to handle the case where n is too large for 1<<n to be efficient in Python?
    # Python ints are arbitrary precision, so 1<<21604 is fine.
    
    # Let's run the B&B
    # We will use a simple DFS.
    # To avoid deep recursion, we use the stack.
    
    # Optimization: If time is running out, stop.
    
    while stack:
        if time.time() > deadline - 10:
            break
            
        P_mask, size, S_list = stack.pop()
        
        # Prune: If we can't beat best_k
        if size + np.bit_count(P_mask) <= best_k:
            continue
            
        if P_mask == 0:
            if size > best_k:
                best_k = size
                best_set = S_list.copy()
            continue
            
        # Branching:
        # Pick a vertex v from P_mask.
        # Heuristic: Pick vertex with max degree in P_mask? Or min?
        # For MIS, picking a vertex with high degree in the conflict graph 
        # means its exclusion removes many candidates, which might be good for pruning?
        # Or picking a vertex with low degree might lead to a larger set?
        # Standard: Pick vertex with max degree in the remaining graph.
        
        # Find vertices in P_mask
        # We need to pick one.
        # Let's pick the one with the smallest index for determinism, or use a heuristic.
        # Heuristic: Max degree in P_mask.
        
        # Calculating degrees in P_mask for all vertices is expensive.
        # Let's just pick the lowest set bit.
        # v_idx = (P_mask & -P_mask).bit_length() - 1
        
        # Better heuristic: Pick a vertex that is "central".
        # For now, let's use lowest index.
        v_idx = (P_mask & -P_mask).bit_length() - 1
        
        # Two branches:
        # 1. Include v_idx
        # 2. Exclude v_idx
        
        # Branch 1: Include v_idx
        # New P = P_mask \ N[v_idx]
        P_include = P_mask & ~(adj[v_idx] | (1 << v_idx))
        # New S = S_list + [v_idx]
        # We push this to stack.
        # To keep stack small, we can push the larger branch last (LIFO).
        
        # Branch 2: Exclude v_idx
        # New P = P_mask \ {v_idx}
        P_exclude = P_mask & ~(1 << v_idx)
        
        # Push Exclude first so Include is processed next?
        # Or push Include first so Exclude is processed next?
        # Usually, we want to find good solutions early.
        # Including a vertex increases size, so it's more likely to beat best_k quickly.
        # So process Include first.
        # Stack is LIFO, so push Exclude, then Include.
        
        stack.append((P_exclude, size, S_list))
        stack.append((P_include, size + 1, S_list + [v_idx]))
        
        # Limit stack size to prevent memory explosion
        if len(stack) > 10000:
            break

    # Convert best_set to points
    if best_set:
        final_points = cands[best_set]
        # Verify validity (safety check)
        # The B&B logic ensures independence in the conflict graph, which ensures validity.
        # But let's double check the max norm condition just in case.
        # The conflict graph was built based on the exact condition.
        
        if final_points.shape[0] > best_points.shape[0]:
            best_points = final_points
            best_k = final_points.shape[0]

    # Final safety
    if best_points is None or best_points.shape[0] < 2:
        best_points = base_points
        best_k = base_points.shape[0]

    return np.array(best_points, dtype=np.int64)
