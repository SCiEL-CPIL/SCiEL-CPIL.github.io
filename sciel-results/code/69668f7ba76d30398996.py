import time
import random
import sys
from itertools import combinations

# Precompute all 5-combinations for each n
MAX_N = 26
ALL_COMBS_BY_N = {}
for n in range(5, MAX_N + 1):
    ALL_COMBS_BY_N[n] = [list(e) for e in combinations(range(n), 5)]

def check_non_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses exact backtracking with unit propagation.
    """
    if not edges:
        return False
    
    # For very small n, brute force might be faster or comparable
    # But backtracking with UP is generally robust.
    # Let's use a bitmask approach for n <= 20 if edges are few?
    # Actually, standard backtracking on vertices is fine for n <= 26.
    
    # Precompute vertex to edge indices
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # State
    colors = [-1] * n
    # edge_state[i] = (count_0, count_1, remaining_mask?) 
    # Let's use count_0, count_1. If count_0 + count_1 == 5, it's satisfied.
    # If count_0 == 5 or count_1 == 5, it's monochromatic -> fail (for this edge)
    # Wait, the condition is: every edge must have BOTH colors.
    # So if an edge becomes all 0s or all 1s, it's a conflict.
    # If an edge has 4 of one color and 0 of the other, the 5th MUST be the other color.
    
    edge_c0 = [0] * len(edges)
    edge_c1 = [0] * len(edges)
    
    # Recursion stack or simple recursion?
    # n is small (<=26), so recursion depth is fine.
    
    def bt():
        # 1. Unit Propagation
        # Find an edge that is "critical"
        # Critical means:
        # a) 4 of one color, 0 of other -> force the 5th
        # b) 5 of one color -> contradiction (return False)
        
        forced_v = -1
        forced_c = -1
        
        for i in range(len(edges)):
            c0 = edge_c0[i]
            c1 = edge_c1[i]
            
            if c0 == 5 or c1 == 5:
                return False
            
            if c0 == 4 and c1 == 0:
                # Find the uncolored vertex in this edge
                for v in edges[i]:
                    if colors[v] == -1:
                        forced_v = v
                        forced_c = 1
                        break
            elif c1 == 4 and c0 == 0:
                for v in edges[i]:
                    if colors[v] == -1:
                        forced_v = v
                        forced_c = 0
                        break
                        
            if forced_v != -1:
                break
        
        # 2. Variable Selection
        if forced_v == -1:
            # Heuristic: Pick vertex with highest degree in uncolored part?
            # Or just first uncolored.
            # Let's pick first uncolored for simplicity, or one with max constraints.
            # Max constraints: number of edges containing v that are not yet satisfied?
            # For speed, first uncolored is okay, but better heuristics help.
            # Let's try: vertex that appears in the most "tight" edges (c0+c1 >= 3)?
            # Simple fallback: first uncolored.
            for v in range(n):
                if colors[v] == -1:
                    forced_v = v
                    forced_c = 0 # Try 0 first
                    break
            
            if forced_v == -1:
                # All vertices colored.
                # Check if all edges are bichromatic.
                # If we reached here without returning False, and all colored,
                # it means no edge became monochromatic.
                # However, we need to be sure.
                # The unit prop would have caught monochromatic edges as they formed.
                # So if all colored and no contradiction, it's a valid coloring.
                return True
        
        # 3. Assignment
        colors[forced_v] = forced_c
        saved_c0 = []
        saved_c1 = []
        
        for i in vertex_edges[forced_v]:
            saved_c0.append(edge_c0[i])
            saved_c1.append(edge_c1[i])
            if forced_c == 0:
                edge_c0[i] += 1
            else:
                edge_c1[i] += 1
        
        if bt():
            return True
        
        # Backtrack
        for i, (c0, c1) in zip(vertex_edges[forced_v], zip(saved_c0, saved_c1)):
            edge_c0[i] = c0
            edge_c1[i] = c1
        colors[forced_v] = -1
        
        # 4. Try other color if not forced by unit prop
        # How do we know if it was forced?
        # We can track a flag, or re-evaluate.
        # If forced_v was set in step 1, it's forced.
        # If set in step 2, it's not forced (we can try both).
        
        # To distinguish, we can check if any edge had 4-0 or 0-4 before assignment.
        # But we overwrote counts.
        # Let's add a flag.
        
        # Wait, I didn't add a flag in the code above. Let's refactor bt to return (found, forced_flag) or use a nonlocal variable.
        # Better: refactor to explicitly handle the "was it forced" logic.
        
        # Let's rewrite bt to be cleaner.
        pass
        
    # The above bt is incomplete. Let's write a proper one.
    
    def bt_correct():
        # Returns (found_coloring, is_forced_branch)
        # Actually, standard DPLL:
        # 1. Propagate.
        # 2. If contradiction, return False.
        # 3. If all assigned, return True.
        # 4. Pick variable.
        # 5. Try value 1. If success, return True.
        # 6. Try value 0. If success, return True.
        # 7. Return False.
        
        # Propagation Loop
        # We need to keep propagating until no more unit clauses or contradiction.
        
        changed = True
        while changed:
            changed = False
            for i in range(len(edges)):
                c0 = edge_c0[i]
                c1 = edge_c1[i]
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and c1 == 0:
                    # Find uncolored v in edge i
                    for v in edges[i]:
                        if colors[v] == -1:
                            # Force v to 1
                            colors[v] = 1
                            # Update counts for all edges containing v
                            for j in vertex_edges[v]:
                                edge_c1[j] += 1
                            changed = True
                            break
                elif c1 == 4 and c0 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            # Force v to 0
                            colors[v] = 0
                            for j in vertex_edges[v]:
                                edge_c0[j] += 1
                            changed = True
                            break
        
        # Check if all colored
        # Optimization: maintain a count of uncolored vertices?
        # For now, loop is fine for n=26.
        uncolored_v = -1
        for v in range(n):
            if colors[v] == -1:
                uncolored_v = v
                break
        
        if uncolored_v == -1:
            # All colored. Since we didn't hit contradiction, it's a valid coloring.
            return True
        
        # Pick variable
        v = uncolored_v
        # Heuristic: pick v that maximizes impact?
        # For now, just use first found.
        
        # Try 0
        colors[v] = 0
        saved_c0 = edge_c0[:] # Copy? Expensive.
        saved_c1 = edge_c1[:]
        
        # Actually, copying lists of size ~50-200 is fine.
        # But we can do it more efficiently by only saving what changes.
        # However, n is small, let's prioritize correctness and simple implementation first.
        # Wait, copying the whole edge_c0/1 arrays at every node might be slow if tree is deep.
        # Better to save only the edges affected by v.
        
        # Let's undo the copy approach and use manual undo.
        
        # Try 0
        colors[v] = 0
        affected_edges = vertex_edges[v]
        old_c0 = [edge_c0[i] for i in affected_edges]
        old_c1 = [edge_c1[i] for i in affected_edges]
        
        for i in affected_edges:
            edge_c0[i] += 1
            
        if bt_correct():
            return True
        
        # Undo 0
        for i, (c0, c1) in zip(affected_edges, zip(old_c0, old_c1)):
            edge_c0[i] = c0
            edge_c1[i] = c1
        colors[v] = -1
        
        # Try 1
        colors[v] = 1
        old_c0 = [edge_c0[i] for i in affected_edges]
        old_c1 = [edge_c1[i] for i in affected_edges]
        
        for i in affected_edges:
            edge_c1[i] += 1
            
        if bt_correct():
            return True
        
        # Undo 1
        for i, (c0, c1) in zip(affected_edges, zip(old_c0, old_c1)):
            edge_c0[i] = c0
            edge_c1[i] = c1
        colors[v] = -1
        
        return False
        
    # Need to handle the propagation loop inside bt_correct properly.
    # The current bt_correct has a while loop at the start.
    # But when we recurse, we call bt_correct again, which will do propagation.
    # This is correct.
    
    # One issue: The propagation loop in bt_correct modifies global state (colors, edge_c0, edge_c1).
    # When we backtrack, we must undo these modifications.
    # The current structure only undoes the manual assignment of v.
    # It does NOT undo the forced assignments made by the propagation loop!
    
    # This is a critical bug in the sketch above.
    
    # Correct approach:
    # 1. Run propagation. Record all forced assignments.
    # 2. If contradiction, undo forced assignments and return False.
    # 3. If no more forced, pick v.
    # 4. Assign v, recurse.
    # 5. Undo v and any new forced assignments made in the sub-call?
    #    No, the sub-call is responsible for its own state.
    #    So, before returning from a node, we must undo everything we did in that node.
    
    # Let's implement a cleaner version.
    
    def solve():
        # Propagation
        # Returns (status, forced_assignments)
        # status: -1 for contradiction, 0 for need_more, 1 for solved
        # forced_assignments: list of (v, c, affected_edges, old_c0s, old_c1s)
        
        forced_list = []
        
        while True:
            found_forced = False
            for i in range(len(edges)):
                c0 = edge_c0[i]
                c1 = edge_c1[i]
                if c0 == 5 or c1 == 5:
                    # Contradiction
                    # Undo forced_list
                    for v, c, aff, oc0, oc1 in forced_list:
                        for idx, (c0v, c1v) in zip(aff, zip(oc0, oc1)):
                            edge_c0[idx] = c0v
                            edge_c1[idx] = c1v
                        colors[v] = -1
                    return -1, forced_list
                if c0 == 4 and c1 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            # Force v to 1
                            colors[v] = 1
                            aff = vertex_edges[v]
                            oc0 = [edge_c0[idx] for idx in aff]
                            oc1 = [edge_c1[idx] for idx in aff]
                            for idx in aff:
                                edge_c1[idx] += 1
                            forced_list.append((v, 1, aff, oc0, oc1))
                            found_forced = True
                            break
                elif c1 == 4 and c0 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            # Force v to 0
                            colors[v] = 0
                            aff = vertex_edges[v]
                            oc0 = [edge_c0[idx] for idx in aff]
                            oc1 = [edge_c1[idx] for idx in aff]
                            for idx in aff:
                                edge_c0[idx] += 1
                            forced_list.append((v, 0, aff, oc0, oc1))
                            found_forced = True
                            break
            if not found_forced:
                break
        
        # Check if all colored
        uncolored_v = -1
        for v in range(n):
            if colors[v] == -1:
                uncolored_v = v
                break
        
        if uncolored_v == -1:
            # Solved
            # We don't need to undo yet, caller handles it?
            # Actually, if solved, we return success.
            # The caller (parent) will undo its own assignments.
            # But we need to undo our forced_list before returning?
            # No, if we return success, the parent will keep the state?
            # No, the parent checks `if solve(): return True`.
            # If success, we don't backtrack.
            # So we should NOT undo if success.
            return 1, forced_list
        
        # Pick variable
        v = uncolored_v
        
        # Try 0
        aff = vertex_edges[v]
        oc0 = [edge_c0[idx] for idx in aff]
        oc1 = [edge_c1[idx] for idx in aff]
        colors[v] = 0
        for idx in aff:
            edge_c0[idx] += 1
        
        status, sub_forced = solve()
        
        if status == 1:
            return 1, forced_list # Success
        
        # Undo 0 and sub_forced
        # First undo sub_forced
        for fv, fc, f_aff, foc0, foc1 in sub_forced:
            for idx, (c0v, c1v) in zip(f_aff, zip(foc0, foc1)):
                edge_c0[idx] = c0v
                edge_c1[idx] = c1v
            colors[fv] = -1
        
        # Undo v
        for idx, (c0v, c1v) in zip(aff, zip(oc0, oc1)):
            edge_c0[idx] = c0v
            edge_c1[idx] = c1v
        colors[v] = -1
        
        # Try 1
        aff = vertex_edges[v]
        oc0 = [edge_c0[idx] for idx in aff]
        oc1 = [edge_c1[idx] for idx in aff]
        colors[v] = 1
        for idx in aff:
            edge_c1[idx] += 1
        
        status, sub_forced = solve()
        
        if status == 1:
            return 1, forced_list # Success
        
        # Undo 1 and sub_forced
        for fv, fc, f_aff, foc0, foc1 in sub_forced:
            for idx, (c0v, c1v) in zip(f_aff, zip(foc0, foc1)):
                edge_c0[idx] = c0v
                edge_c1[idx] = c1v
            colors[fv] = -1
        
        for idx, (c0v, c1v) in zip(aff, zip(oc0, oc1)):
            edge_c0[idx] = c0v
            edge_c1[idx] = c1v
        colors[v] = -1
        
        # Failed
        return 0, forced_list
        
    status, _ = solve()
    return status == 0 # 0 means "Failed to find coloring" -> Not 2-colorable. Wait.
    # My status: 1 is Solved (Found Coloring). 0 is Failed (No Coloring). -1 is Contradiction (No Coloring).
    # So if status is 0 or -1, it is NOT 2-colorable.
    # If status is 1, it IS 2-colorable.
    
    # Let's re-read.
    # "Returns True if the hypergraph is NOT 2-colorable"
    # So if solve() finds a coloring (status 1), return False.
    # If solve() fails (status 0 or -1), return True.
    
    # But wait, my solve returns 0 for "Failed" at the end of the branch.
    # And -1 for contradiction.
    # So:
    # status 1 -> Coloring exists -> 2-colorable -> Return False.
    # status 0 or -1 -> No coloring -> Not 2-colorable -> Return True.
    
    # Let's adjust the return logic.
    
    # Actually, the `solve` function above is a bit messy with the forced_list passing.
    # Let's simplify by making `solve` not return forced_list, but just handle backtracking internally.
    
    # Revised `solve`:
    # def solve():
    #     # Propagate, recording changes in a local stack
    #     # If contradiction, undo stack, return False
    #     # If all assigned, return True
    #     # Pick v
    #     # Try 0:
    #     #    Push v=0 to stack
    #     #    if solve(): return True
    #     #    Pop v=0 from stack
    #     # Try 1:
    #     #    Push v=1 to stack
    #     #    if solve(): return True
    #     #    Pop v=1 from stack
    #     # Return False
    
    # This is much cleaner. The "stack" is just the list of changes to undo if this node fails.
    
    # Let's implement this version.

    def solve_clean():
        # Returns True if a valid coloring is found (i.e., 2-colorable)
        # We need to undo any changes made in this call if we return False.
        # But if we return True, we leave the state?
        # No, if we return True, the caller (parent) will also return True.
        # So the state doesn't matter if we succeed.
        # But if we fail, we must undo everything done in this branch.
        
        # Local stack for undoing
        undo_stack = []
        
        # Propagation
        while True:
            forced = False
            for i in range(len(edges)):
                c0 = edge_c0[i]
                c1 = edge_c1[i]
                if c0 == 5 or c1 == 5:
                    # Contradiction
                    # Undo all changes in undo_stack
                    for v, c, aff, oc0, oc1 in undo_stack:
                        for idx, (c0v, c1v) in zip(aff, zip(oc0, oc1)):
                            edge_c0[idx] = c0v
                            edge_c1[idx] = c1v
                        colors[v] = -1
                    return False
                
                if c0 == 4 and c1 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            # Force v=1
                            colors[v] = 1
                            aff = vertex_edges[v]
                            oc0 = [edge_c0[idx] for idx in aff]
                            oc1 = [edge_c1[idx] for idx in aff]
                            for idx in aff:
                                edge_c1[idx] += 1
                            undo_stack.append((v, 1, aff, oc0, oc1))
                            forced = True
                            break
                elif c1 == 4 and c0 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            # Force v=0
                            colors[v] = 0
                            aff = vertex_edges[v]
                            oc0 = [edge_c0[idx] for idx in aff]
                            oc1 = [edge_c1[idx] for idx in aff]
                            for idx in aff:
                                edge_c0[idx] += 1
                            undo_stack.append((v, 0, aff, oc0, oc1))
                            forced = True
                            break
            if not forced:
                break
        
        # Check if all colored
        uncolored_v = -1
        for v in range(n):
            if colors[v] == -1:
                uncolored_v = v
                break
        
        if uncolored_v == -1:
            return True # Valid coloring found
        
        v = uncolored_v
        
        # Try 0
        aff = vertex_edges[v]
        oc0 = [edge_c0[idx] for idx in aff]
        oc1 = [edge_c1[idx] for idx in aff]
        colors[v] = 0
        for idx in aff:
            edge_c0[idx] += 1
        undo_stack.append((v, 0, aff, oc0, oc1))
        
        if solve_clean():
            return True # Success propagates up
        
        # Undo 0
        # Pop the last item? No, we appended it.
        # But solve_clean might have added more items to the GLOBAL undo_stack?
        # No, undo_stack is local to this call.
        # Wait, solve_clean creates a NEW undo_stack.
        # So the changes made by solve_clean are undone by solve_clean if it fails.
        # So we only need to undo the assignment of v here.
        
        # Remove v=0 from undo_stack?
        # Actually, we don't need to remove it from the list if we just manually undo it.
        # But to keep it clean, let's just manually undo the specific change.
        
        for idx, (c0v, c1v) in zip(aff, zip(oc0, oc1)):
            edge_c0[idx] = c0v
            edge_c1[idx] = c1v
        colors[v] = -1
        
        # Try 1
        aff = vertex_edges[v]
        oc0 = [edge_c0[idx] for idx in aff]
        oc1 = [edge_c1[idx] for idx in aff]
        colors[v] = 1
        for idx in aff:
            edge_c1[idx] += 1
        undo_stack.append((v, 1, aff, oc0, oc1))
        
        if solve_clean():
            return True
        
        # Undo 1
        for idx, (c0v, c1v) in zip(aff, zip(oc0, oc1)):
            edge_c0[idx] = c0v
            edge_c1[idx] = c1v
        colors[v] = -1
        
        # If we get here, both branches failed.
        # We must undo the propagation changes made in THIS call (before branching).
        # These are in undo_stack.
        for v_u, c_u, aff_u, oc0_u, oc1_u in undo_stack:
            for idx, (c0v, c1v) in zip(aff_u, zip(oc0_u, oc1_u)):
                edge_c0[idx] = c0v
                edge_c1[idx] = c1v
            colors[v_u] = -1
        
        return False
    
    is_2_colorable = solve_clean()
    return not is_2_colorable

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_m = float('inf')
    best_edges = None
    best_n = 0
    
    # Strategy:
    # 1. Start with K_n^5 for small n.
    # 2. Try to remove edges greedily/randomly.
    # 3. Use local search to minimize edges.
    
    # We know m(5) is around 51.
    # Smallest n for non-2-colorable 5-uniform hypergraph?
    # K_9^5 has 126 edges.
    # Can we do better?
    
    # Let's try n=9, 10, 11...
    
    for n in [9, 10, 11, 12]:
        if time.time() > deadline - 10:
            break
            
        init_edges = [list(e) for e in combinations(range(n), 5)]
        m = len(init_edges)
        
        if not check_non_2_colorable(n, init_edges):
            continue
            
        cur_edges = init_edges[:]
        cur_m = m
        
        if cur_m < best_m:
            best_m = cur_m
            best_edges = cur_edges[:]
            best_n = n
            
        # Local Search: Randomized Greedy Removal
        # Try to remove edges one by one.
        # If removal keeps it non-2-colorable, accept.
        # If stuck, try swaps.
        
        # To speed up, we can prioritize removing edges that are "least important".
        # But random is fine.
        
        # Optimization: Precompute all possible edges for this n
        all_edges = [tuple(e) for e in combinations(range(n), 5)]
        all_edges_set = set(all_edges)
        
        # Shuffle the current edges to avoid bias
        random.shuffle(cur_edges)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to remove each edge
            for i in range(len(cur_edges)):
                if time.time() > deadline - 5:
                    break
                
                e = cur_edges[i]
                new_edges = cur_edges[:i] + cur_edges[i+1:]
                
                if check_non_2_colorable(n, new_edges):
                    # Accept removal
                    cur_edges = new_edges
                    cur_m -= 1
                    improved = True
                    
                    if cur_m < best_m:
                        best_m = cur_m
                        best_edges = cur_edges[:]
                        best_n = n
                        
                    # Restart the loop from the beginning?
                    # Since the list changed, indices are shifted.
                    # It's safer to restart the for loop.
                    # But breaking and restarting the while loop is cleaner.
                    break
            
            # If no edge was removed in this pass, try swaps?
            # Or just stop for this n.
            # Given time limits, simple greedy is often enough to get close to optimal.
            
    # Final check
    if best_edges is None:
        # Fallback: K_9^5
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
    
    # Verify best result
    if not check_non_2_colorable(best_n, best_edges):
        # Something went wrong?
        # Return a known valid one if possible?
        # K_9^5 is definitely valid.
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
    print("Verifying...")
    assert not check_non_2_colorable(res['n'], res['edges']), "Result is 2-colorable!"
    print("Valid.")
