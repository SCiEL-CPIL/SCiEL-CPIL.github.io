"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Generate Candidate Pool
    # We include vectors of norms 1, 2, 3, 4.
    # Norm 1: e_i, -e_i (22 vectors)
    # Norm 2: +/-e_i +/-e_j (220 vectors)
    # Norm 3: +/-e_i +/-e_j +/-e_k (1980 vectors)
    # Norm 4: 
    #   - four 1/-1 (5280 vectors)
    #   - one 2 (22 vectors)
    #   - two 2s (220 vectors)
    # Total ~ 7744 candidates.
    
    candidates = []
    
    # Helper to add vector
    def add_vec(coords):
        v = np.zeros(D, dtype=np.int64)
        for idx, val in coords.items():
            v[idx] = val
        candidates.append(v)

    # Norm 1
    for i in range(D):
        add_vec({i: 1})
        add_vec({i: -1})
        
    # Norm 2
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                add_vec({i: si, j: sj})
                
    # Norm 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    add_vec({i: si, j: sj, k: sk})
                    
    # Norm 4
    # Type 1: four 1/-1
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        add_vec({i: si, j: sj, k: sk, l: sl})
                        
    # Type 2: one 2
    for i in range(D):
        for si in [1, -1]:
            add_vec({i: si * 2})
            
    # Type 3: two 2s
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                add_vec({i: si * 2, j: sj * 2})

    C = np.array(candidates)
    n_cand = len(C)
    norms_sq = np.sum(C * C, axis=1)

    # 2. Build Incompatibility Graph
    # Two vectors u, v are incompatible if ||u-v||^2 < max(||u||^2, ||v||^2)
    # => ||u||^2 + ||v||^2 - 2 u.v < max(||u||^2, ||v||^2)
    # => 2 u.v > min(||u||^2, ||v||^2)
    
    # We will build the adjacency list of the conflict graph.
    # conflict[u][v] = True if they cannot coexist.
    # Since n_cand is ~7700, a full matrix is 7700x7700 ~ 60MB. 
    # But we can be smarter. We only need to store neighbors for the solver.
    
    # Let's compute the conflict matrix in blocks to save memory and time, 
    # or just store it as a list of sets/lists.
    # Given the deadline, a full matrix calculation might take a few seconds.
    # 7744^2 * 11 ops ~ 650 million ops. NumPy handles this fast.
    
    # Compute all pairwise dots
    # This creates a (n, n) matrix.
    # To save memory, we can compute block by block, but 60MB is fine.
    dots = C @ C.T
    
    # Compute limits: min(norms_sq[i], norms_sq[j]) for all pairs
    # norms_sq is (n,). We need outer min.
    # norms_sq[:, None] and norms_sq[None, :]
    min_norms = np.minimum(norms_sq[:, None], norms_sq[None, :])
    
    # Conflict condition: 2 * dots > min_norms
    conflict_matrix = 2 * dots > min_norms
    
    # Diagonal is always True (self-conflict), which is fine for MIS logic if we handle it, 
    # but usually we consider distinct nodes.
    
    # Convert to list of neighbors for the solver
    # We use a list of lists of indices.
    # To make it faster for the solver, we can use numpy arrays for neighbors.
    # However, Python lists of ints are often faster for the specific recursion patterns 
    # if we don't vectorize the solver itself (which is hard).
    
    # Let's optimize the solver to use bitsets or efficient numpy operations if possible,
    # but standard Python recursion with pruning is the "structural" change.
    
    # For better performance, we'll keep the conflict matrix as a boolean numpy array 
    # and access it directly, or convert to a list of sets for O(1) lookup?
    # Actually, for the "Tomita" or similar MIS algorithms, we need to iterate neighbors.
    # A list of lists is good.
    
    neighbors = []
    for i in range(n_cand):
        # Find j > i such that conflict[i, j] is True
        # We can use np.where(conflict_matrix[i, i+1:])
        idxs = np.where(conflict_matrix[i, i+1:])[0] + i + 1
        neighbors.append(idxs)
        
    # Also need reverse neighbors? Or just use the matrix?
    # The standard MIS algorithm (branch and bound) usually works on a subgraph.
    # We need to efficiently find the independent set in the subgraph induced by a set of vertices.
    
    # Let's implement a simple but effective Branch and Bound for MIS.
    # Algorithm:
    # 1. Order vertices by degree (descending) to improve pruning.
    # 2. Recursively try to include a vertex, or exclude it.
    # 3. Bound: If current_size + remaining_vertices <= best_size, prune.
    
    # To make this fast enough for 7700 nodes, we need a strong heuristic.
    # A common approach is to use a greedy coloring bound (like in Max Clique solvers).
    # MIS is Max Clique in the complement graph.
    # The complement graph has edges where vectors ARE compatible.
    # Let's work with the Compatibility Graph (Complement of Conflict).
    
    # Compatibility Graph:
    # comp_matrix[i, j] = not conflict_matrix[i, j] and i != j
    # We want Max Clique in this graph.
    
    # Let's build the adjacency list for the Compatibility Graph.
    # Note: The graph is dense? 
    # Most random vectors in R^11 with small norms are compatible?
    # ||u-v||^2 >= max(||u||^2, ||v||^2) is a fairly loose constraint for small norms.
    # E.g. norm 1 vs norm 1: 2 - 2 dot >= 1 => dot <= 0.5 => dot <= 0 (integer).
    # So orthogonal or anti-parallel.
    # Many pairs are compatible.
    
    # Let's build the compatibility adjacency list.
    # comp_neighbors[i] = list of j such that compatible(i, j)
    
    comp_neighbors = []
    for i in range(n_cand):
        # Compatible if NOT (2 * dots[i, j] > min_norms[i, j])
        # And i != j
        # We can compute this from conflict_matrix
        # conflict_matrix is True if incompatible.
        # So compatible is ~conflict_matrix.
        comp_mask = ~conflict_matrix[i]
        comp_mask[i] = False # No self loop
        idxs = np.where(comp_mask)[0]
        comp_neighbors.append(idxs)
        
    # Heuristic: Order vertices by degree in Compatibility Graph (descending).
    # High degree = many compatible partners = good candidate to start.
    degrees = np.array([len(n) for n in comp_neighbors])
    order = np.argsort(-degrees)
    
    # Map original indices to new order indices?
    # Or just permute the graph?
    # Permuting a 7700x7700 graph is expensive.
    # Instead, we can keep the original indices but visit them in the `order`.
    # The MIS algorithm usually operates on a set of available vertices.
    
    # Let's use a recursive function that takes a set of available vertices (as a numpy array or list).
    # To be fast, we should use a bitset representation if n < 64*64? No, 7700 is too big for single int.
    # Python's arbitrary precision ints can be used as bitsets.
    # 7700 bits is ~1000 bytes. Operations are fast in Python.
    
    # Let's try the Bitset approach.
    # It is often faster than list manipulation for dense graphs.
    
    # Precompute bitsets for neighbors?
    # neighbor_bitset[i] = int with bits set for all j in comp_neighbors[i]
    
    neighbor_bitsets = [0] * n_cand
    for i in range(n_cand):
        bits = 0
        for j in comp_neighbors[i]:
            bits |= (1 << j)
        neighbor_bitsets[i] = bits
        
    # This might be slow to build if we do it naively.
    # 7700 * (avg degree) operations.
    # Avg degree is likely high (maybe 5000+).
    # 7700 * 5000 = 38 million operations. Might take a few seconds in Python.
    # Let's optimize the bitset creation using numpy?
    # Or just stick to lists if bitset creation is too slow.
    
    # Alternative: Use a standard MIS solver on the list representation with good pruning.
    # The "Tomita" algorithm for Max Clique is very effective.
    # We adapt it for MIS on the compatibility graph.
    
    # Let's use a simpler, robust B&B with coloring bound.
    # Coloring the compatibility graph gives an upper bound on the clique size.
    # Greedy coloring is O(V+E).
    
    # Given the time constraints, let's try a simpler greedy + local improvement first, 
    # but the prompt asks for a STRUCTURAL change. 
    # The current program is SA. 
    # A "Graph MIS" solver is a structural change.
    
    # If the graph is too big for exact MIS in 150s, we might time out.
    # 7700 nodes is large for exact MIS.
    # However, the graph structure is regular-ish.
    
    # Let's try a different structural approach: 
    # **Integer Linear Programming (ILP) relaxation or heuristic**? No, scipy doesn't have ILP.
    
    # Let's try **Greedy Set Packing with Conflict Graph Pruning**.
    # This is similar to the current one but uses the pre-computed graph to make decisions faster.
    # The current program recalculates dots. This one uses the pre-computed `conflict_matrix`.
    # This is a significant speedup, allowing for MORE search (e.g., more SA restarts or a deeper search).
    
    # Strategy:
    # 1. Precompute conflict matrix (done).
    # 2. Use a fast greedy algorithm that uses the matrix for O(1) conflict checks (vectorized).
    # 3. Use Simulated Annealing or Local Search with the precomputed matrix.
    #    The cost of a move (swap) can be evaluated faster.
    
    # Actually, the most robust "different" strategy for this size is a **Greedy Algorithm with Backtracking** 
    # or **Beam Search**.
    
    # Let's implement a **Beam Search** on the candidate set.
    # State: Set of selected vectors (represented by indices).
    # Expansion: Add a compatible vector.
    # Pruning: Keep top K beams based on size (and maybe potential).
    
    # But Beam Search on a set is hard to represent.
    
    # Let's go with **Max Clique in Compatibility Graph** using a simple Branch and Bound 
    # with a greedy coloring bound.
    # If it's too slow, we fall back to the best found so far.
    
    # We will implement a simplified version of the Tomita algorithm for Max Clique.
    # It works on a graph G. We want Max Clique in G_compat.
    
    # To handle the size, we might need to reduce the candidate pool or use a heuristic.
    # Let's try to run the exact solver with a time limit.
    
    best_clique = []
    
    # We'll use a recursive function.
    # P: set of vertices available to extend the clique.
    # X: set of vertices already processed (for coloring bound).
    # C: current clique.
    
    # To make it fast, we represent P and X as numpy arrays of indices? 
    # Or lists?
    # Lists are easier for recursion.
    
    # Let's optimize the compatibility check using the precomputed matrix.
    
    # Greedy Initialization
    def greedy_max_clique(order):
        clique = []
        remaining = list(order)
        while remaining:
            # Pick vertex with max degree in remaining subgraph?
            # Or just pick the first one in the ordered list?
            # Tomita uses a specific ordering.
            # Let's just pick the first one in the provided order (which is by global degree).
            v = remaining[0]
            clique.append(v)
            # Filter remaining to only those compatible with v
            new_remaining = []
            for u in remaining[1:]:
                if not conflict_matrix[v, u]: # compatible
                    new_remaining.append(u)
            remaining = new_remaining
        return clique
    
    # This greedy might be weak. 
    # Let's use a better heuristic: 
    # Sort vertices by degree.
    # Greedy: 
    #   P = all vertices
    #   while P:
    #      v = vertex in P with max degree in P
    #      add v to clique
    #      P = P intersect N(v)
    
    # Let's implement this.
    
    def greedy_heuristic():
        P = set(range(n_cand))
        clique = []
        while P:
            # Find v in P with max degree in P
            # Degree in P: count of neighbors in P
            # This is O(|P|^2) if done naively.
            # We can approximate with global degree or just pick first.
            # For speed, let's just pick the vertex with highest global degree in P.
            # Since we have `degrees`, we can sort P by degrees.
            # But P changes.
            
            # Optimization: Use the precomputed `order` (sorted by global degree).
            # Pick the first v in `order` that is in P.
            v = None
            for u in order:
                if u in P:
                    v = u
                    break
            if v is None:
                break
                
            clique.append(v)
            # Update P: intersect with N(v)
            N_v = set(comp_neighbors[v])
            P = P.intersection(N_v)
            
        return clique
        
    init_clique = greedy_heuristic()
    best_size = len(init_clique)
    best_clique = init_clique
    
    # Now, let's try to improve using Local Search (Swaps).
    # This is similar to the current program but uses the precomputed matrix for speed.
    # The speedup allows for more iterations.
    
    current_clique = set(init_clique)
    current_size = len(current_clique)
    
    # Precompute a list of candidates not in the clique?
    # Or just iterate over all candidates.
    
    # We need to check if a candidate v is compatible with all in current_clique.
    # Using conflict_matrix, this is: 
    # conflicts = conflict_matrix[v, list(current_clique)]
    # if not any(conflicts): compatible
    
    # Let's run a loop of random swaps/adds.
    
    rng = np.random.default_rng(42)
    
    # To speed up, we can maintain a list of "blocked" candidates?
    # Or just check on the fly.
    
    # Since we have the matrix, checking compatibility of v against K vectors is O(K) vector ops.
    # Very fast.
    
    # Let's do 100,000 iterations of SA/Local Search.
    
    T = 1.0
    alpha = 0.9999
    
    # We need to be able to remove a vector from the set.
    # Set removal is O(1).
    
    # To pick a random conflicting vector, we need to know which ones conflict.
    # conflicts = conflict_matrix[v, list(current_clique)]
    # conflict_indices = np.where(conflicts)[0]
    
    # This requires converting set to list every time. 
    # Let's keep current_clique as a list and a set for O(1) lookup/removal?
    # Or just a list, and use a boolean array `in_clique` of size n_cand.
    
    in_clique = np.zeros(n_cand, dtype=bool)
    for v in current_clique:
        in_clique[v] = True
        
    current_list = list(current_clique)
    
    for step in range(200000):
        if time.time() > deadline:
            break
            
        # Pick random candidate
        v_new = rng.integers(0, n_cand)
        
        if in_clique[v_new]:
            continue
            
        # Check compatibility
        # We need to check v_new against all v in current_list
        # dots = C[v_new] @ C[current_list].T
        # But we have conflict_matrix.
        # conflicts = conflict_matrix[v_new, current_list]
        
        # To avoid list conversion overhead, we can use the boolean mask?
        # No, we need the specific indices.
        
        # Let's use the matrix row.
        row = conflict_matrix[v_new]
        # We need to check if any of the selected vectors conflict.
        # selected_mask = in_clique
        # conflicts = row & selected_mask
        # if not np.any(conflicts):
        
        # This is O(N) to check the whole row. N=7700.
        # 200,000 * 7700 = 1.5 billion ops. Too slow in Python.
        
        # We need a faster check.
        # Only check against the current clique members.
        # current_list length is ~500.
        # So we extract row[current_list].
        # This is a fancy indexing operation. O(K).
        
        if not current_list:
            in_clique[v_new] = True
            current_list.append(v_new)
            current_size += 1
            continue
            
        # Extract conflicts with current clique
        # Using numpy fancy indexing
        # This creates a view or copy.
        conflicts = row[current_list]
        
        if not np.any(conflicts):
            # Add
            in_clique[v_new] = True
            current_list.append(v_new)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_clique = current_list.copy()
        else:
            # Try swap
            # Find a conflicting vector to remove
            # conflict_indices_in_list = np.where(conflicts)[0]
            # idx_to_remove = conflict_indices_in_list[rng.integers(0, len(conflict_indices_in_list))]
            # v_old = current_list[idx_to_remove]
            
            # To make it faster, we can just pick a random index from the current list 
            # and check if it conflicts?
            # Or pick the first conflict?
            
            # Let's pick a random conflict.
            # We need the indices of conflicts in the `current_list`.
            # The `conflicts` array is aligned with `current_list`.
            
            conflict_idxs = np.where(conflicts)[0]
            if len(conflict_idxs) > 0:
                idx_old = conflict_idxs[rng.integers(0, len(conflict_idxs))]
                v_old = current_list[idx_old]
                
                # Remove v_old
                in_clique[v_old] = False
                # Remove from list (O(K))
                current_list.pop(idx_old)
                
                # Check if v_new is compatible with the rest
                # We already know it's compatible with all except v_old (and v_old was the only conflict? No, there might be others)
                # Wait, `conflicts` showed ALL conflicts.
                # If we remove v_old, we still need to check against the others that were conflicting.
                # Actually, if there were multiple conflicts, removing one doesn't make v_new compatible.
                # So we should only try swap if there is EXACTLY ONE conflict?
                # Or if we remove ALL conflicts? (Too complex)
                
                # Let's only swap if there is exactly 1 conflict.
                if len(conflict_idxs) == 1:
                    # It is compatible with the rest
                    in_clique[v_new] = True
                    current_list.append(v_new)
                    # Size stays same
                else:
                    # Reject, restore v_old
                    in_clique[v_old] = True
                    current_list.append(v_old)
                    # Note: order in list changed, but that's fine for set logic.
                    # However, `conflicts` array indices were based on the OLD list.
                    # We popped idx_old.
                    # We appended v_old.
                    # The list is now different.
                    # This is fine for the next iteration.

    return np.array(best_clique, dtype=np.int64)

# EVOLVE-BLOCK-END
