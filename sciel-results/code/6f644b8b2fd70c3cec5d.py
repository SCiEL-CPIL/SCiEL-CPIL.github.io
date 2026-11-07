"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all maximal cliques
        # Bron-Kerbosch with pivot
        def bron_kerbosch(r, p_set, x):
            if not p_set and not x:
                if len(r) >= 2:
                    yield tuple(sorted(r))
            # Choose pivot u from p_set union x to minimize branching
            u = None
            if p_set:
                u = next(iter(p_set))
            elif x:
                u = next(iter(x))
            
            if u is None:
                return
                
            # Branch on vertices in p_set - neighbors(u)
            candidates = list(p_set - adj[u])
            for v in candidates:
                new_r = r | {v}
                new_p = p_set & adj[v]
                new_x = x & adj[v]
                yield from bron_kerbosch(new_r, new_p, new_x)
                p_set.remove(v)
                x.add(v)

        # Note: Standard BK finds maximal cliques. We need to be careful with the recursion to avoid modifying sets in place incorrectly or missing cliques.
        # A simpler approach for small n: enumerate all subsets? No, n=90 is too big for 2^n.
        # Let's use a standard implementation carefully.
        
        # Reset for a clean BK run
        # We need to find ALL maximal cliques.
        # The above generator has a bug if not handled correctly with sets. Let's use a recursive list-based approach for safety or a known correct lib.
        # Since we can't import external libs easily, let's write a robust BK.
        
        max_cliques = []
        
        def bk(r, p, x):
            if not p and not x:
                if len(r) >= 2:
                    max_cliques.append(tuple(sorted(r)))
                return
            # Pivot selection
            u = None
            if p:
                u = next(iter(p))
            elif x:
                u = next(iter(x))
            
            if u is None:
                return

            # Branch on v in P \ N(u)
            for v in list(p - adj[u]):
                new_r = r | {v}
                new_p = p & adj[v]
                new_x = x & adj[v]
                bk(new_r, new_p, new_x)
                p.remove(v)
                x.add(v)
        
        bk(set(), set(range(n)), set())
        
        # Filter cliques to ensure they are valid (all edges present) - BK guarantees this, but just in case
        valid_cliques = []
        for c in max_cliques:
            is_clique = True
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    if (min(c[i], c[j]), max(c[i], c[j])) not in edge_set:
                        is_clique = False
                        break
                if not is_clique: break
            if is_clique:
                valid_cliques.append(c)
        
        # If no cliques found (shouldn't happen for connected components with edges), fallback
        if not valid_cliques:
             # Fallback to edge partition
             return [[u, v] for u, v in edges]

        # Map edges to cliques
        # For each edge, which cliques contain it?
        edge_to_cliques = {e: [] for e in edge_set}
        for idx, c in enumerate(valid_cliques):
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    e = (min(c[i], c[j]), max(c[i], c[j]))
                    if e in edge_to_cliques:
                        edge_to_cliques[e].append(idx)
        
        # Set Covering Problem:
        # Universe U = edge_set
        # Sets S = valid_cliques, where S_i covers {e in edge_set | e in clique_i}
        # Minimize number of sets such that union covers U.
        
        # Precompute coverage bitmasks or lists for speed
        # n_edges is at most ~1200 for n=90. Bitmask in Python int is fine.
        edge_indices = list(edge_set)
        edge_to_idx = {e: i for i, e in enumerate(edge_indices)}
        
        clique_masks = []
        for c in valid_cliques:
            mask = 0
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    e = (min(c[i], c[j]), max(c[i], c[j]))
                    mask |= (1 << edge_to_idx[e])
            clique_masks.append(mask)
            
        full_mask = (1 << len(edge_indices)) - 1
        
        # Branch and Bound
        best_solution = []
        best_size = len(edge_indices) # Upper bound: each edge is a clique
        
        # Sort cliques by coverage size descending to help pruning
        # We need to keep track of original indices to reconstruct solution
        cliques_with_info = [(mask, idx) for idx, mask in enumerate(clique_masks) if mask > 0]
        cliques_with_info.sort(key=lambda x: bin(x[0]).count('1'), reverse=True)
        
        # Remove cliques that are subsets of others? 
        # If clique A's edges are a subset of clique B's edges, A is never better than B 
        # unless A allows covering something else? No, in set cover, if S_A subset S_B, 
        # choosing B covers at least what A covers, and costs same (1). So A is dominated.
        # However, we must be careful: if we pick A, we might not be able to pick B later? 
        # No, set cover is just a set of sets. If A subset B, replacing A with B in any solution 
        # doesn't increase cost and doesn't decrease coverage. So A is redundant.
        
        # Let's filter dominated cliques
        unique_masks = {}
        for mask, idx in cliques_with_info:
            if mask not in unique_masks:
                unique_masks[mask] = idx
        
        # This filtering is O(M^2) which might be slow if M is large. 
        # For n=90, number of maximal cliques can be exponential in worst case but typically manageable for split graphs?
        # Split graphs have a specific structure. The maximal cliques are subsets of K union {v} for v in I where v is adjacent to the subset.
        # Actually, in a split graph, every maximal clique contains all of K? No.
        # A maximal clique in a split graph is of the form C where C subset K, and C is adjacent to a set of vertices in I.
        # If we take K, it's a clique. If we add v in I to K, it's a clique if v is adjacent to all K.
        # If v is not adjacent to all K, we can't add it to K.
        # So maximal cliques are either K itself, or K \ N(v) + {v} ? No.
        # Let's rely on the BK result.
        
        # To speed up B&B, we can use a greedy upper bound
        def greedy_upper_bound():
            covered = 0
            count = 0
            remaining_cliques = list(range(len(clique_masks)))
            while covered != full_mask and remaining_cliques:
                best_idx = -1
                best_new = -1
                for idx in remaining_cliques:
                    new_cover = clique_masks[idx] & ~covered
                    cnt = bin(new_cover).count('1')
                    if cnt > best_new:
                        best_new = cnt
                        best_idx = idx
                if best_new <= 0: break
                covered |= clique_masks[best_idx]
                count += 1
                remaining_cliques.remove(best_idx)
            return count if covered == full_mask else float('inf')
        
        initial_ub = greedy_upper_bound()
        if initial_ub < best_size:
            best_size = initial_ub
            # Reconstruct greedy solution for tie-breaking or just use size
            # We need the actual list of cliques for the final answer
            # Let's store the best solution found so far
            # For now, just track size. We'll reconstruct at the end if we find a better one, 
            # or just return the greedy one if B&B doesn't improve.
            
        # Actually, we need to return the list of cliques.
        # Let's store the best set of indices.
        
        best_indices = None
        
        # Reconstruct greedy solution indices
        covered = 0
        greedy_indices = []
        remaining = set(range(len(clique_masks)))
        while covered != full_mask and remaining:
            best_idx = -1
            best_new = -1
            for idx in remaining:
                new_cover = clique_masks[idx] & ~covered
                cnt = bin(new_cover).count('1')
                if cnt > best_new:
                    best_new = cnt
                    best_idx = idx
            if best_new <= 0: break
            covered |= clique_masks[best_idx]
            greedy_indices.append(best_idx)
            remaining.remove(best_idx)
            
        if covered == full_mask:
            best_indices = greedy_indices
            best_size = len(greedy_indices)
        else:
            # Should not happen if graph is covered by edges
            best_indices = [edge_to_idx[e] for e in edge_indices] # Fallback to edges? No, indices are cliques.
            # Fallback: use edges as cliques
            return [[u, v] for u, v in edges]

        # B&B State:
        # current_covered (int mask)
        # current_count (int)
        # start_idx (int) - index in sorted cliques to avoid permutations
        
        # We use the sorted cliques for B&B to prune better?
        # If we sort, we lose the mapping to original indices easily? No, we kept (mask, idx).
        
        # Let's use the original unsorted list for B&B but sort by "potential" dynamically? 
        # Standard B&B for Set Cover:
        # Sort sets by size descending.
        
        # Let's re-sort clique_masks for B&B
        # We need to map back to original clique indices for the final answer.
        # sorted_cliques: list of (mask, original_idx)
        
        # We already have cliques_with_info sorted by size desc.
        # Let's use that.
        
        # Pruning: 
        # 1. current_count + lower_bound >= best_size
        # Lower bound: max(1, ceil(remaining_edges / max_cover_size))
        
        # To make it faster, we can precompute max cover size for each step? 
        # Or just use a global max.
        
        max_cover_size = max([bin(m).count('1') for m, _ in cliques_with_info] if cliques_with_info else [1])
        
        def bb(covered, count, start_idx):
            nonlocal best_size, best_indices
            
            if time.time() > DEADLINE:
                return
            
            if covered == full_mask:
                if count < best_size:
                    best_size = count
                    # We need to store the indices. 
                    # This requires passing the path. 
                    # To save memory/time, we can just store the count and reconstruct? 
                    # No, we need the specific cliques.
                    # Let's pass a list of indices.
                    # This might be slow for deep recursion. 
                    # Alternative: store the best solution in a global variable when updated.
                    pass 
                return
            
            # Lower bound
            remaining_bits = bin(full_mask & ~covered).count('1')
            if remaining_bits == 0:
                return
            lb = (remaining_bits + max_cover_size - 1) // max_cover_size
            if count + lb >= best_size:
                return
            
            # Heuristic: pick next set
            # We iterate through sets from start_idx onwards
            for i in range(start_idx, len(cliques_with_info)):
                mask, orig_idx = cliques_with_info[i]
                
                # If this set adds no new edges, skip? 
                # It might be part of a combination? No, if it adds nothing, it's useless in this branch.
                new_cover = mask & ~covered
                if new_cover == 0:
                    continue
                
                # Check if this set is a subset of another set we haven't considered? 
                # Complex to check on the fly.
                
                new_count = count + 1
                if new_count >= best_size:
                    continue
                
                # Recursive call
                # We need to track the path to update best_indices
                # To avoid passing a list (copy overhead), we can use a global path stack
                path_stack.append(orig_idx)
                bb(covered | mask, new_count, i + 1)
                path_stack.pop()
                
                if time.time() > DEADLINE:
                    return

        # We need to handle the path storage properly
        path_stack = []
        
        # Modify bb to update best_indices when a better solution is found
        # Since bb is recursive and we pop, we need to capture the state when we hit a leaf.
        
        # Let's rewrite bb to accept the current list of indices.
        # For small n, this is fine.
        
        def bb_safe(covered, count, start_idx, current_indices):
            nonlocal best_size, best_indices
            
            if time.time() > DEADLINE:
                return
            
            if covered == full_mask:
                if count < best_size:
                    best_size = count
                    best_indices = list(current_indices)
                return
            
            remaining_bits = bin(full_mask & ~covered).count('1')
            if remaining_bits == 0:
                return
            lb = (remaining_bits + max_cover_size - 1) // max_cover_size
            if count + lb >= best_size:
                return
            
            for i in range(start_idx, len(cliques_with_info)):
                mask, orig_idx = cliques_with_info[i]
                
                new_cover = mask & ~covered
                if new_cover == 0:
                    continue
                
                new_count = count + 1
                if new_count >= best_size:
                    continue
                
                # Pruning: if new_count + 0 >= best_size, skip (already checked above)
                
                current_indices.append(orig_idx)
                bb_safe(covered | mask, new_count, i + 1, current_indices)
                current_indices.pop()
                
                if time.time() > DEADLINE:
                    return

        # Start B&B
        # Initial state: covered=0, count=0, start_idx=0, indices=[]
        # But we already have a greedy solution. 
        # B&B will search for something better.
        
        # Optimization: If the number of cliques is small, B&B is fast.
        # If it's large, it might be slow.
        # The deadline guard handles the timeout.
        
        # One more optimization: If best_size is 1, we can stop early?
        # If best_size == 1, it's the minimum possible (for non-empty graph).
        
        if best_size > 1:
            bb_safe(0, 0, 0, [])
            
        if best_indices is None:
            # Fallback to greedy
            return [list(valid_cliques[i]) for i in greedy_indices]
        else:
            return [list(valid_cliques[i]) for i in best_indices]

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            # Emergency fallback: return empty or seed solution?
            # Better to return a valid solution for what we have or a simple one.
            # If we run out of time, we should return the seed solution for the remaining instances
            # to ensure validity, even if score is lower.
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        cliques = solve_one(*spec)
        # Ensure types are correct
        clean_cliques = []
        for c in cliques:
            # c is a list of ints
            if len(c) >= 2:
                clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
# EVOLVE-BLOCK-END
