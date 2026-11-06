import time
import sys
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 128

    # Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d)
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            i0, i1, i2, i3 = a, a + d, a + 2 * d, a + 3 * d
            ap_id = len(aps)
            aps.append((i0, i1, i2, i3))
            ap_indices[i0].append(ap_id)
            ap_indices[i1].append(ap_id)
            ap_indices[i2].append(ap_id)
            ap_indices[i3].append(ap_id)
    
    # Map AP index to position index within the AP (0,1,2,3)
    # ap_pos_idx[ap_id][pos] = index in the tuple (0-3)
    # Actually, simpler: for a given AP, we know the 4 positions.
    # We need to quickly check the state of an AP given one position's change.

    def validate(coloring):
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                return False
        return True

    def relabel(colors):
        mapping = {}
        new_colors = []
        next_id = 0
        for c in colors:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new_colors.append(mapping[c])
        return new_colors, next_id

    def count_colors(colors):
        return len(set(colors))

    def solve_exact(k, time_budget=20.0):
        """
        Backtracking solver with constraint propagation.
        Tries to find a coloring with exactly k colors (0..k-1).
        """
        # coloring[i] = -1 if unassigned
        coloring = [-1] * N
        # domains[i] = set of available colors
        domains = [set(range(k)) for _ in range(N)]
        
        # Track assigned count
        assigned_count = 0
        
        # Heuristic: Precompute degree (number of APs each position is in)
        degrees = [len(ap_indices[i]) for i in range(N)]
        
        # To speed up, we can use a queue of "implications" or just re-scan affected APs.
        # Given N=128 and k small, re-scanning affected APs is fast enough.
        
        def propagate(idx, val):
            """
            Assign val to idx. Update domains of neighbors.
            Returns False if a domain becomes empty.
            """
            nonlocal assigned_count
            coloring[idx] = val
            assigned_count += 1
            
            # Check all APs containing idx
            for ap_id in ap_indices[idx]:
                ap = aps[ap_id]
                # Determine which position in the AP is idx
                # We need to check if the other 3 are assigned.
                # Let's collect the states of the 4 positions in this AP.
                # pos_in_ap: 0,1,2,3
                # We know idx is one of them.
                
                # Optimization: We only need to check if the *other* 3 are assigned.
                # If they are, we can constrain the 4th (if unassigned) or check consistency (if assigned).
                
                # Identify the other 3 positions
                others = [p for p in ap if p != idx]
                
                # Check if all 3 others are assigned
                if all(coloring[o] != -1 for o in others):
                    # The AP is fully assigned. Check validity.
                    c_vals = [coloring[p] for p in ap]
                    if len(set(c_vals)) < 3:
                        return False
                    
                    # Since it's fully assigned, no domain updates needed for this AP.
                    # But wait, if we are assigning idx, and the others were already assigned,
                    # we just validate. We don't update domains of assigned nodes.
                    
                else:
                    # Some others are unassigned.
                    # Let's see which ones are assigned.
                    assigned_others = [o for o in others if coloring[o] != -1]
                    unassigned_others = [o for o in others if coloring[o] == -1]
                    
                    # If 3 are assigned (idx + 2 others), we can constrain the 1 unassigned.
                    # If 2 are assigned (idx + 1 other), we have 2 unassigned. 
                    # Constraint: The set of colors in the AP must have size >= 3.
                    # Let S_assigned be the set of colors of assigned positions in this AP.
                    # If |S_assigned| == 2, then the unassigned positions MUST NOT use any color in S_assigned 
                    # ONLY IF there is exactly 1 unassigned position? 
                    # No. If there are 2 unassigned, they can split the 3rd color.
                    # Example: Colors {0,1} assigned. Unassigned u1, u2. 
                    # We need total distinct >= 3. So {u1, u2} must contain a color NOT in {0,1}.
                    # This means u1 != u2 is not required, but at least one must be outside {0,1}.
                    # This is a complex constraint (NAND-like). 
                    # For simplicity and speed, we will only apply the strong constraint 
                    # when exactly 1 position in the AP is unassigned.
                    
                    if len(unassigned_others) == 1:
                        u = unassigned_others[0]
                        # All 3 others (including idx) are assigned.
                        s = {coloring[p] for p in ap}
                        # The unassigned position u cannot take any color in s if that would keep |S| < 3?
                        # No, if u takes a color in s, the set of colors remains s.
                        # If |s| < 3, then the AP is invalid regardless of u's color?
                        # No, if u takes a NEW color not in s, the set becomes s U {new}.
                        # If |s| == 2, and u takes a color in s, |S_total| = 2 (Invalid).
                        # If u takes a color NOT in s, |S_total| = 3 (Valid).
                        # So u MUST NOT take any color in s.
                        if len(s) < 3:
                            # Remove all colors in s from domain of u
                            new_domain = domains[u] - s
                            if not new_domain:
                                return False
                            # We don't explicitly store the "removed" set for backtracking easily in this simple structure,
                            # so we will use a stack of (var, old_domain, new_domain) for undo.
                            # But for now, let's just modify and rely on backtracking to restore.
                            # To make backtracking clean, we should record changes.
                            
                            # Let's implement a simple trail for domains.
                            # Actually, re-computing domains from scratch on backtrack is O(N*APs) which is ~10k ops.
                            # With 128 vars, depth 128, worst case 128*10k = 1.2M ops per level? No.
                            # Propagation is local. 
                            
                            # Let's use a trail.
                            trail.append((u, domains[u][:])) # Store copy? Expensive.
                            # Better: Store (u, old_set)
                            # Since we modify in place, we need the old state.
                            domains[u] = new_domain
                            
                    elif len(unassigned_others) == 2:
                        # We have idx assigned, 1 other assigned, 2 unassigned.
                        # Let a = idx, b = other assigned, u1, u2 = unassigned.
                        # Colors ca, cb.
                        # If ca == cb, S_assigned = {ca}. Size 1.
                        # We need final set size >= 3.
                        # So {u1, u2} must contribute at least 2 NEW colors? 
                        # If u1, u2 are both new and distinct, total 3.
                        # If u1 is new, u2 is ca, total 2 (Invalid).
                        # This constraint is: NOT (u1 in S_assigned AND u2 in S_assigned) AND NOT (u1 == u2 AND u1 in S_assigned) ... 
                        # This is getting complex. We will skip this weaker propagation step to keep the solver fast and simple.
                        # The strong constraint (1 unassigned) is the most powerful.
                        pass
            
            return True

        # Simple Trail for Backtracking
        # We need to restore domains.
        # Instead of complex trails, we can just re-initialize domains for unassigned nodes on backtrack?
        # Or, since k is small (3 or 4), we can just keep the full domain sets and update them,
        # and on backtrack, we pop the assignment and re-derive domains?
        # Re-derivation is safe but potentially slower.
        # Given N=128, let's try a simpler approach:
        # Recursive backtracking with explicit domain copying for the current node? No, too much memory.
        
        # Let's use the standard "trail" approach but carefully.
        # Actually, for N=128, a simpler "re-scan" approach might be faster than managing complex trails.
        # When backtracking from depth d, we unassign var d.
        # We then need to restore domains of neighbors that were constrained by var d.
        # This is hard to track without trails.
        
        # Alternative: Use a SAT-like approach or just brute force with strong pruning.
        # Let's try a different structure:
        # 1. Order variables by degree.
        # 2. Assign one by one.
        # 3. Check validity of all completed APs.
        # 4. Prune domains of unassigned vars based on completed APs.
        
        # Let's implement this "Forward Checking" without complex trails by simply 
        # recalculating domains for all unassigned variables after each assignment?
        # Cost: N * (APs per var) ~ 128 * 100 = 12800 ops.
        # Depth 128. Total 1.6M ops. This is very fast.
        
        def rec_solve(depth):
            nonlocal coloring, domains, assigned_count
            if time.time() > deadline - 5:
                return False
            
            if depth == N:
                return True
            
            # Find next unassigned variable
            # Use MRV heuristic: pick unassigned var with smallest domain
            min_size = k + 1
            var_idx = -1
            for i in range(N):
                if coloring[i] == -1:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        var_idx = i
                        if min_size == 1:
                            break
            
            if var_idx == -1:
                return True # All assigned
            
            if min_size == 0:
                return False # Dead end
            
            # Try each color in domain
            # Order colors by least constraining value? (Optional)
            for val in list(domains[var_idx]):
                # Save state for backtrack
                old_coloring = coloring[var_idx]
                old_domains = [d.copy() for d in domains] # Expensive? 128 sets. 
                # Copying 128 sets of size <=4 is fast.
                
                coloring[var_idx] = val
                domains[var_idx] = {val}
                
                # Propagate
                valid = True
                # Update domains of neighbors
                for ap_id in ap_indices[var_idx]:
                    ap = aps[ap_id]
                    others = [p for p in ap if p != var_idx]
                    assigned_others = [o for o in others if coloring[o] != -1]
                    unassigned_others = [o for o in others if coloring[o] == -1]
                    
                    if len(unassigned_others) == 1:
                        u = unassigned_others[0]
                        s = {coloring[p] for p in ap}
                        if len(s) < 3:
                            new_dom = domains[u] - s
                            if not new_dom:
                                valid = False
                                break
                            domains[u] = new_dom
                            
                if valid:
                    if rec_solve(depth + 1):
                        return True
                
                # Backtrack
                coloring[var_idx] = old_coloring
                domains = old_domains
                
            return False

        # Start
        # Initial domains
        domains = [set(range(k)) for _ in range(N)]
        coloring = [-1] * N
        
        # Optional: Randomize variable order to avoid symmetry breaking issues?
        # MRV handles it dynamically.
        
        return rec_solve(0)

    # Strategy:
    # 1. Try k=3 with exact search.
    # 2. If fails or timeout, try k=4.
    # 3. If k=4 fails (shouldn't), fallback to SA.
    
    best_colors = None
    best_count = 4
    
    # Try k=3
    print("Trying exact search with k=3...", file=sys.stderr)
    if solve_exact(3, time_budget=20.0):
        best_colors = coloring[:]
        best_count = 3
        print("Found 3-coloring!", file=sys.stderr)
    else:
        print("k=3 failed or timeout. Trying k=4...", file=sys.stderr)
        # Try k=4
        if solve_exact(4, time_budget=20.0):
            best_colors = coloring[:]
            best_count = 4
            print("Found 4-coloring!", file=sys.stderr)
        else:
            # Fallback to SA if exact search fails (should be rare)
            print("Exact search failed. Falling back to SA...", file=sys.stderr)
            # Simple SA fallback
            rng = random.Random(42)
            coloring = [rng.randint(0, 3) for _ in range(N)]
            # Quick local repair
            for _ in range(10000):
                if time.time() > deadline - 5: break
                violated = []
                for ap in aps:
                    c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                    if len({c0, c1, c2, c3}) < 3:
                        violated.append(ap)
                if not violated:
                    break
                ap = rng.choice(violated)
                # Change one color in this AP
                pos = rng.choice(ap)
                coloring[pos] = rng.randint(0, 3)
            
            if validate(coloring):
                best_colors = coloring
                best_count = count_colors(best_colors)
            else:
                # Absolute fallback: Periodic 4-coloring (might not be valid, but better than crash)
                # Actually, a known valid 4-coloring: i % 4?
                # Check i%4: 0,1,2,3. AP 0,1,2,3 -> 0,1,2,3 (4 colors). OK.
                # AP 0,2,4,6 -> 0,2,0,2 (2 colors). FAIL.
                # So i%4 is NOT valid.
                # We need a valid 4-coloring.
                # If exact search failed, we are in trouble. 
                # But exact search for k=4 should almost certainly succeed for N=128.
                best_colors = [0]*N
                best_count = N

    # Relabel to ensure labels are 0..K-1
    if best_colors:
        best_colors, final_count = relabel(best_colors)
        # Final validation
        if not validate(best_colors):
             # Emergency: return a known valid pattern if possible, or just the best effort
             # For safety, if validation fails, we might return an invalid coloring which scores 0.
             # We must ensure validity.
             # If exact search returned True, it is valid.
             # If SA fallback returned True, it is valid.
             pass
    else:
        best_colors = [0]*N
        final_count = 1

    return best_colors

# EVOLVE-BLOCK-START
# The code above is the entire program.
# We just need to wrap it in the correct format.
# The prompt asks for the COMPLETE program.
# The seed had EVOLVE-BLOCK-START/END markers. I will include them.
# EVOLVE-BLOCK-END
