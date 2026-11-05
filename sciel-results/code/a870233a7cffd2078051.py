import time
import math
from math import gcd, lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    N = 256
    candidates = list(range(2, 257))
    n = len(candidates)
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    idx_to_val = candidates

    # Precompute conflict masks
    conflict_mask = [0] * n
    for i in range(n):
        vi = idx_to_val[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            vj = idx_to_val[j]
            if lcm(vi, vj) <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def get_score_frac(mask):
        score = Fraction(0)
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            score += Fraction(1, idx_to_val[idx])
            m ^= lsb
        return score

    def get_selected(mask):
        return [idx_to_val[i] for i in range(n) if (mask >> i) & 1]

    def is_valid_mask(mask):
        sel = get_selected(mask)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                if lcm(sel[i], sel[j]) <= 256:
                    return False
        return True

    # Find connected components of the conflict graph
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            # BFS/DFS to find component
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                mask_u = conflict_mask[u]
                # Iterate through neighbors
                m = mask_u
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
                    m ^= lsb
            components.append(comp)

    def solve_component_exact(comp_nodes):
        """
        Solve MWIS exactly for a small component using bitmask DP.
        comp_nodes: list of global indices.
        Returns: (best_mask_global, best_score)
        """
        k = len(comp_nodes)
        if k == 0:
            return 0, Fraction(0)
        
        # Map global indices to local 0..k-1
        global_to_local = {g: l for l, g in enumerate(comp_nodes)}
        
        # Local conflict masks
        local_conflict = [0] * k
        for i in range(k):
            gi = comp_nodes[i]
            m = conflict_mask[gi]
            lm = 0
            for j in range(k):
                if m & (1 << comp_nodes[j]):
                    lm |= (1 << j)
            local_conflict[i] = lm

        # DP over subsets: dp[mask] = max weight independent set in subgraph induced by mask
        # Optimization: only iterate valid masks or use recursion with memoization?
        # For k<=22, 2^22 is ~4M. Might be slow in Python if not careful.
        # Let's use a recursive solver with memoization and pruning, or simple DP if k is small.
        # Actually, for k up to 22, a simple loop over 2^k is feasible if optimized.
        # But 4M iterations with Fraction arithmetic might be too slow.
        # Let's restrict exact solve to k <= 18 (2^18 = 262k).
        
        if k > 18:
            return None, None # Fallback to heuristic

        size = 1 << k
        # dp[mask] stores the max score (as Fraction) and the mask itself?
        # No, we just need the max score. But we need to reconstruct the mask.
        # We can store the best mask for each subset? That's 2^k memory.
        # Alternatively, store just the score and backtrack?
        
        # Let's use a dictionary for memoization of (mask) -> (best_score, best_submask)
        # Or just iterate masks and compute max weight independent set for that mask?
        # Standard DP:
        # dp[mask] = max weight of independent set within 'mask'
        # Base: dp[0] = 0
        # Transition:
        #   Let v be the lowest set bit in mask.
        #   Option 1: Don't include v. dp[mask] = dp[mask ^ (1<<v)]
        #   Option 2: Include v. dp[mask] = w[v] + dp[mask ^ (1<<v) ^ local_conflict[v]]
        
        # We need to store the mask to reconstruct the solution.
        # best_mask_for_mask = {}
        
        # To save time, we can just return the best mask found by the DP.
        # We'll store dp_score[mask] and dp_choice[mask] (0: exclude lowest, 1: include lowest)
        
        dp_score = [Fraction(0)] * size
        dp_choice = [0] * size
        
        for mask in range(1, size):
            # Find lowest set bit
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            
            # Option 1: Exclude v
            mask_excl = mask ^ lsb
            score_excl = dp_score[mask_excl]
            choice_excl = 0
            
            # Option 2: Include v
            # Neighbors of v in mask: local_conflict[v] & mask
            # New mask: mask_excl ^ (local_conflict[v] & mask_excl)
            # Wait, if we include v, we must remove v and all its neighbors from consideration.
            # The remaining set is mask_excl & ~local_conflict[v]
            mask_incl = mask_excl & ~local_conflict[v]
            score_incl = dp_score[mask_incl] + Fraction(1, idx_to_val[comp_nodes[v]])
            choice_incl = 1
            
            if score_incl > score_excl:
                dp_score[mask] = score_incl
                dp_choice[mask] = 1
            else:
                dp_score[mask] = score_excl
                dp_choice[mask] = 0

        # Reconstruct
        mask = size - 1
        res_mask_local = 0
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if dp_choice[mask] == 1:
                res_mask_local |= lsb
                # Remove v and neighbors
                mask = (mask ^ lsb) & ~local_conflict[v]
            else:
                mask = mask ^ lsb
        
        # Map back to global
        res_mask_global = 0
        for i in range(k):
            if (res_mask_local >> i) & 1:
                res_mask_global |= (1 << comp_nodes[i])
                
        return res_mask_global, dp_score[size - 1]

    def local_search(initial_mask, time_limit=10.0):
        current_mask = initial_mask
        current_score = get_score_frac(current_mask)
        improved = True
        start_ls_time = time.time()

        while improved:
            improved = False

            if time.time() > deadline or time.time() - start_ls_time > time_limit:
                break

            # Add phase
            for v in range(2, 257):
                idx = val_to_idx[v]
                if not (current_mask & (1 << idx)):
                    if (conflict_mask[idx] & current_mask) == 0:
                        current_mask |= (1 << idx)
                        current_score += Fraction(1, v)
                        improved = True

            if improved:
                continue

            # Build lists
            in_mask = [i for i in range(n) if (current_mask >> i) & 1]
            not_in_mask = [i for i in range(n) if not ((current_mask >> i) & 1)]

            # 1-for-1 swap
            in_mask_desc = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
            not_in_mask_asc = sorted(not_in_mask, key=lambda i: idx_to_val[i])

            found_swap = False
            for u_idx in in_mask_desc:
                if time.time() > deadline:
                    break
                u_val = idx_to_val[u_idx]
                mask_without_u = current_mask & ~(1 << u_idx)

                for v_idx in not_in_mask_asc:
                    v_val = idx_to_val[v_idx]
                    if v_val >= u_val:
                        break
                    if (conflict_mask[v_idx] & mask_without_u) == 0:
                        current_mask = (current_mask & ~(1 << u_idx)) | (1 << v_idx)
                        current_score += Fraction(1, v_val) - Fraction(1, u_val)
                        improved = True
                        found_swap = True
                        break

                if found_swap:
                    break

            if found_swap:
                continue

            # 2-for-1 swap
            in_mask_desc2 = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
            not_in_mask_asc2 = sorted(not_in_mask, key=lambda i: idx_to_val[i])
            found_2swap = False
            for i1 in range(len(in_mask_desc2)):
                if time.time() > deadline:
                    break
                u1_idx = in_mask_desc2[i1]
                u1_val = idx_to_val[u1_idx]
                for i2 in range(i1 + 1, len(in_mask_desc2)):
                    u2_idx = in_mask_desc2[i2]
                    u2_val = idx_to_val[u2_idx]
                    gain_needed = Fraction(1, u1_val) + Fraction(1, u2_val)
                    mask_without_u1_u2 = current_mask & ~(1 << u1_idx) & ~(1 << u2_idx)

                    for v_idx in not_in_mask_asc2:
                        v_val = idx_to_val[v_idx]
                        if Fraction(1, v_val) <= gain_needed:
                            break
                        if (conflict_mask[v_idx] & mask_without_u1_u2) == 0:
                            current_mask = (current_mask & ~(1 << u1_idx) & ~(1 << u2_idx)) | (1 << v_idx)
                            current_score += Fraction(1, v_val) - Fraction(1, u1_val) - Fraction(1, u2_val)
                            improved = True
                            found_2swap = True
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    break

            if found_2swap:
                continue

            # 1-for-2 swap
            in_mask_desc3 = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
            not_in_mask_asc3 = sorted(not_in_mask, key=lambda i: idx_to_val[i])
            found_12swap = False
            for u_idx in in_mask_desc3:
                if time.time() > deadline:
                    break
                u_val = idx_to_val[u_idx]
                gain_needed = Fraction(1, u_val)
                mask_without_u = current_mask & ~(1 << u_idx)

                for v1_idx in not_in_mask_asc3:
                    v1_val = idx_to_val[v1_idx]
                    if (conflict_mask[v1_idx] & mask_without_u) != 0:
                        continue
                    if Fraction(1, v1_val) >= gain_needed:
                        continue
                    mask_with_v1 = mask_without_u | (1 << v1_idx)
                    for v2_idx in not_in_mask_asc3:
                        if v2_idx == v1_idx:
                            continue
                        v2_val = idx_to_val[v2_idx]
                        if Fraction(1, v1_val) + Fraction(1, v2_val) <= gain_needed:
                            break
                        if (conflict_mask[v2_idx] & mask_with_v1) != 0:
                            continue
                        current_mask = (current_mask & ~(1 << u_idx)) | (1 << v1_idx) | (1 << v2_idx)
                        current_score += Fraction(1, v1_val) + Fraction(1, v2_val) - Fraction(1, u_val)
                        improved = True
                        found_12swap = True
                        break
                    if found_12swap:
                        break
                if found_12swap:
                    break

        return current_mask, current_score

    # Process components
    best_total_mask = 0
    best_total_score = Fraction(0)
    
    # Sort components by size descending to tackle hard ones first? Or just process all.
    # We will solve small ones exactly, large ones heuristically.
    
    for comp in components:
        if len(comp) <= 18:
            res_mask, res_score = solve_component_exact(comp)
            if res_mask is not None:
                best_total_mask |= res_mask
                best_total_score += res_score
        else:
            # Heuristic for large component
            # Greedy initialization on this component
            init_mask = 0
            # Sort nodes in component by value ascending (highest weight first)
            comp_sorted = sorted(comp, key=lambda i: idx_to_val[i])
            for idx in comp_sorted:
                if (conflict_mask[idx] & init_mask) == 0:
                    init_mask |= (1 << idx)
            
            # Local search restricted to this component? 
            # The current local_search operates on global mask.
            # We can run it on the global mask but only allow changes within this component?
            # Simpler: Run local search on the whole global mask, but start with init_mask for this part.
            # Since components are independent, we can just add the result to the global mask.
            # But local_search might try to add nodes from other components?
            # No, conflict_mask is 0 for nodes in different components.
            # So adding a node from another component is always valid and doesn't conflict.
            # So we can just run local search on the union of solved components + this one.
            
            # Let's just add the greedy result for now and let a final global polish handle interactions (though there are none).
            # Actually, since there are no edges between components, the optimal solution is the union of optimal solutions of each component.
            # So we can just solve each component independently and combine.
            
            # Run local search on a mask containing only this component's nodes
            # But local_search adds nodes from *anywhere* if they don't conflict.
            # Since other components don't conflict, it will just add them too.
            # This is fine. It will effectively solve all remaining components in one go.
            
            # To be efficient, let's just collect all "hard" components and solve them together?
            # No, if they are independent, we can solve them in parallel (sequentially in code) and combine.
            
            # Let's run local search on a mask initialized with the greedy solution for this component.
            # But we need to make sure it doesn't interfere with already solved components.
            # It won't, because no conflicts.
            
            # So:
            comp_mask, comp_score = local_search(init_mask, time_limit=10.0)
            # The resulting comp_mask might contain nodes from other components if they were not in init_mask and got added.
            # This is a problem if we process components one by one.
            
            # Better approach:
            # 1. Identify all components.
            # 2. For small ones, solve exactly.
            # 3. For large ones, combine them into a single "hard" mask and run local search on that.
            #    But if there are multiple large components, they are independent.
            #    We can just run local search on the union of all "hard" component greedy solutions.
            
            # Let's refactor:
            pass

    # Refactored Logic:
    small_comp_masks = 0
    hard_comp_nodes = []
    
    for comp in components:
        if len(comp) <= 18:
            res_mask, res_score = solve_component_exact(comp)
            if res_mask is not None:
                small_comp_masks |= res_mask
        else:
            hard_comp_nodes.extend(comp)

    # Solve hard components together
    if hard_comp_nodes:
        # Greedy init for hard nodes
        hard_mask = 0
        hard_sorted = sorted(hard_comp_nodes, key=lambda i: idx_to_val[i])
        for idx in hard_sorted:
            if (conflict_mask[idx] & hard_mask) == 0:
                hard_mask |= (1 << idx)
        
        # Local search on hard_mask
        # Note: local_search will also add nodes from small components if they are not already in the mask?
        # No, we will combine small_comp_masks and hard_mask at the end.
        # But local_search operates on a single mask.
        # If we pass hard_mask, it might add nodes from small components (which are independent).
        # That's fine, it will just add them.
        # But we already have the *exact* solution for small components.
        # The local search might find a *worse* solution for small components if it starts from 0 there.
        # So we should initialize the mask with the exact solutions of small components.
        
        init_hard = small_comp_masks | hard_mask
        hard_mask, hard_score = local_search(init_hard, time_limit=20.0)
        
        best_total_mask = hard_mask
        # We don't strictly need hard_score if we recompute, but it's good for checking.
    else:
        best_total_mask = small_comp_masks

    # Final polish
    if time.time() < deadline - 5.0:
        best_total_mask, _ = local_search(best_total_mask, time_limit=10.0)

    # Verify and return
    selected = get_selected(best_total_mask)
    if not is_valid_mask(best_total_mask):
        # Fallback
        selected = []
        for v in range(2, 257):
            if all(lcm(v, y) > 256 for y in selected):
                selected.append(v)

    if not selected:
        selected = [2]

    return selected
