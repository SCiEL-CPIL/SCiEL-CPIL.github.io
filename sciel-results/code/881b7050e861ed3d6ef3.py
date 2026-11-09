"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 255
    # Numbers are 2 to 256
    # Index i corresponds to number i + 2
    
    # 1. Precompute Compatibility Graph (Complement of Conflict Graph)
    # We want to find a clique in this graph.
    # compat_mask[i] is a bitmask where bit j is 1 if i and j are compatible (lcm > 256).
    compat_mask = [0] * N
    weights = [0.0] * N
    
    for i in range(N):
        a = i + 2
        weights[i] = 1.0 / a
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 2
            g = gcd(a, b)
            # lcm(a, b) = a * b // g
            # Compatible if lcm > 256
            if a * b // g > 256:
                mask |= (1 << j)
        compat_mask[i] = mask

    # Pre-sort indices by weight descending (i.e., number ascending)
    # This helps in finding good solutions early and ordering vertices for coloring
    sorted_indices = list(range(N))
    sorted_indices.sort(key=lambda x: -weights[x])
    
    # 2. Optimized Bron-Kerbosch with Pivot and Greedy Coloring for Maximum Weight Clique
    # We will use a recursive function that takes a set of candidates P (bitmask)
    # and a current clique R (list of indices, though we mostly track weight and size for bounds)
    # Actually, for the bound, we just need the set P.
    
    best_weight = 0.0
    best_clique = [] # List of original indices

    # Helper to get list of indices from bitmask, sorted by weight descending
    # Optimization: Instead of sorting every time, we can iterate through sorted_indices 
    # and check if bit is set. Since N=255, this is fast.
    def get_sorted_vertices(P_mask):
        verts = []
        for idx in sorted_indices:
            if (P_mask >> idx) & 1:
                verts.append(idx)
        return verts

    def greedy_color_bound(P_mask):
        """
        Returns an upper bound on the maximum weight of a clique in the subgraph induced by P_mask.
        Uses greedy coloring on the graph induced by P_mask.
        Vertices are processed in order of descending weight.
        Color classes are independent sets in G (i.e., non-adjacent in G, meaning compatible in conflict graph? 
        No, in G, edge = compatible. Independent set in G = set of pairwise INcompatible vertices.
        A clique in G can contain at most 1 vertex from any independent set.
        So we partition P into independent sets.
        Bound = sum over each independent set of the max weight in that set.
        """
        if P_mask == 0:
            return 0.0
            
        verts = get_sorted_vertices(P_mask)
        
        # colors[v] is the color index
        # color_classes[c] is a list of vertices
        color_classes = []
        
        for v in verts:
            assigned_color = -1
            # Try to assign v to an existing color class
            # v can join color class c if v is NOT compatible with any vertex in that class
            # (i.e., v is incompatible with all vertices in the class, so {v} U class is still an independent set in G)
            for c, class_verts in enumerate(color_classes):
                can_join = True
                for u in class_verts:
                    if (compat_mask[v] >> u) & 1:
                        can_join = False
                        break
                if can_join:
                    assigned_color = c
                    break
            
            if assigned_color == -1:
                assigned_color = len(color_classes)
                color_classes.append([])
            
            color_classes[assigned_color].append(v)
            
        # Calculate bound
        bound = 0.0
        for c_verts in color_classes:
            if c_verts:
                max_w = max(weights[v] for v in c_verts)
                bound += max_w
        return bound

    def bron_kerbosch(P, X, R_weight, R_list):
        nonlocal best_weight, best_clique, deadline
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            if R_weight > best_weight:
                best_weight = R_weight
                best_clique = list(R_list)
            return
            
        # Bound check
        # Upper bound on weight of any clique in P
        b = greedy_color_bound(P)
        if R_weight + b <= best_weight:
            return
            
        # Pivot selection
        # Choose a pivot u from P union X that maximizes |N(u) intersect P|
        PX = P | X
        if PX == 0:
            return
            
        pivot = -1
        max_count = -1
        # Iterate through bits in PX
        # Optimization: We can iterate through sorted_indices for PX too? 
        # Or just simple bit iteration.
        m = PX
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            # Count neighbors of u in P
            neighbors_in_P = compat_mask[u] & P
            # Count bits
            count = bin(neighbors_in_P).count('1')
            if count > max_count:
                max_count = count
                pivot = u
            m ^= lsb
            
        # The set of vertices to recurse on is P \ N(pivot)
        # If P \ N(pivot) is empty, we just recurse on P?
        # Standard BK: Iterate over vertices in P \ N(pivot).
        # If that set is empty, it means P is a subset of N(pivot).
        # In that case, any maximal clique in P must contain a neighbor of pivot?
        # Actually, if P \ N(pivot) is empty, then for any v in P, v is adjacent to pivot.
        # This doesn't mean we can skip.
        # However, the standard algorithm iterates over P \ N(pivot).
        # If P \ N(pivot) is empty, the loop runs 0 times?
        # No, if P \ N(pivot) is empty, then P is a clique? No.
        # Let's check the logic.
        # Maximal cliques are subsets of R U {v} U C where v in P \ N(pivot) and C is a clique in P \ N(pivot) \ N(v)?
        # Wait, the theorem states that the set of maximal cliques is the union of:
        # 1. R U {v} U C, where v in P \ N(pivot) and C is a maximal clique in (P \ N(pivot)) \ N(v) ? No.
        # C is a maximal clique in (P \ N(pivot)) intersect N(v)?
        # Let's stick to the standard implementation which is robust.
        
        candidates_mask = P & ~compat_mask[pivot] if pivot != -1 else P
        
        if candidates_mask == 0:
            # If no candidates, we must still check if P itself forms a clique?
            # No, if P \ N(pivot) is empty, it means every node in P is connected to pivot.
            # This doesn't imply P is a clique.
            # However, if candidates_mask is 0, the standard BK loop terminates.
            # Is it possible we miss a clique?
            # If P \ N(pivot) is empty, then P is a subset of N(pivot).
            # This implies pivot is connected to all nodes in P.
            # But we are not adding pivot to R.
            # The standard BK algorithm is correct. If P \ N(pivot) is empty, 
            # it means there are no vertices in P that are NOT connected to pivot.
            # In this case, we don't need to recurse on any v in P \ N(pivot) because the set is empty.
            # But do we miss cliques that don't include pivot?
            # No, because any maximal clique in P must contain at least one vertex from P \ N(pivot) 
            # UNLESS P is empty? 
            # Wait, the theorem says: The set of maximal cliques of G is the union of 
            # R U {v} U C for all v in P \ N(pivot) ...
            # If P \ N(pivot) is empty, then there are no such cliques?
            # That would imply no maximal clique exists in P?
            # That can't be right.
            # Let's re-verify.
            # If P \ N(pivot) is empty, then P is a subset of N(pivot).
            # This means pivot is adjacent to all vertices in P.
            # Consider a maximal clique K in G[P].
            # If K does not contain pivot (which it can't, pivot is in X or P but not in R yet? 
            # Wait, pivot is in P or X. If pivot is in P, it's a candidate.
            # The algorithm iterates v in P \ N(pivot).
            # If P \ N(pivot) is empty, the loop doesn't run.
            # This suggests we might miss cliques.
            # Actually, if P \ N(pivot) is empty, then P is a clique?
            # No.
            # Let's look at a counterexample.
            # G: 1-2, 2-3. P={1,2,3}. Pivot=2. N(2)={1,3}. P \ N(2) = {}
            # Maximal cliques: {1,2}, {2,3}.
            # If we don't iterate, we find nothing.
            # So, if candidates_mask == 0, we should iterate over P?
            # Yes, standard implementations often fall back to P if P \ N(pivot) is empty.
            candidates_mask = P

        # Get sorted list of candidates
        cand_list = get_sorted_vertices(candidates_mask)
        
        for v in cand_list:
            # Check if v is still in P (it might have been removed? No, we are iterating a static list)
            # But we need to ensure v is in the current P.
            if not (P & (1 << v)):
                continue
                
            # Recurse
            new_P = P & compat_mask[v]
            new_X = X & compat_mask[v]
            
            R_list.append(v)
            bron_kerbosch(new_P, new_X, R_weight + weights[v], R_list)
            R_list.pop()
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)
            
            if time.time() > deadline:
                return

    # Strategy: Core-Forced Search
    # 1. Try finding best clique with no forced core (standard)
    # 2. Try forcing specific small numbers (high weight) to be in the clique
    
    # Identify top candidates for cores
    # Small numbers have high weight.
    # Let's try cores: [], [0], [1], [0,1], [0,2], [1,2], [0,1,2]
    # Index 0 is number 2. Index 1 is number 3. Index 2 is number 4.
    
    cores_to_try = [
        [],
        [0],
        [1],
        [0, 1],
        [0, 2],
        [1, 2],
        [0, 1, 2]
    ]
    
    for core in cores_to_try:
        if time.time() > deadline - 5.0:
            break
            
        # Check if core is a valid clique
        valid_core = True
        for i in range(len(core)):
            for j in range(i+1, len(core)):
                if not ((compat_mask[core[i]] >> core[j]) & 1):
                    valid_core = False
                    break
            if not valid_core:
                break
                
        if not valid_core:
            continue
            
        # Calculate initial weight and set
        core_weight = sum(weights[v] for v in core)
        core_set = set(core)
        
        # Determine P: vertices compatible with all vertices in core, and not in core
        P_mask = 0
        for v in range(N):
            if v in core_set:
                continue
            # v must be compatible with all u in core
            compatible_with_core = True
            for u in core:
                if not ((compat_mask[v] >> u) & 1):
                    compatible_with_core = False
                    break
            if compatible_with_core:
                P_mask |= (1 << v)
                
        # X is empty initially for this subproblem
        X_mask = 0
        
        # Run BK
        R_list = list(core)
        bron_kerbosch(P_mask, X_mask, core_weight, R_list)
        
        # Update best if found
        # bron_kerbosch updates nonlocal best_weight and best_clique directly
        
    # Convert best_clique indices to numbers
    result_numbers = [idx + 2 for idx in best_clique]
    
    # If no clique found (should not happen), return best single node
    if not result_numbers:
        max_w = 0
        max_idx = 0
        for i in range(N):
            if weights[i] > max_w:
                max_w = weights[i]
                max_idx = i
        result_numbers = [max_idx + 2]
        
    # Verification
    for i in range(len(result_numbers)):
        for j in range(i+1, len(result_numbers)):
            a = result_numbers[i]
            b = result_numbers[j]
            g = gcd(a, b)
            if a * b // g <= 256:
                return [2]

    return result_numbers

# EVOLVE-BLOCK-END
