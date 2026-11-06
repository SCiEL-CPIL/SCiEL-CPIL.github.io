import time
import sys

def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    # Precompute all 4-term APs (0-indexed)
    # AP is (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_aps = len(aps)

    # For each position, list of indices of APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    # For each AP, store the other 3 positions for quick access
    # ap_other[ap_idx][j] is the position in ap_idx different from pos j? 
    # Actually, let's just store the tuple of 4 positions.
    # To check validity of an AP given colors, we just look at the 4 colors.
    
    # We will use a backtracking solver.
    # State: 
    #   colors: list of length n, -1 if unassigned
    #   domains: list of sets of possible colors for each position
    #   We will use forward checking.
    
    # To make it faster, we can precompute for each AP and each position in it, 
    # the "impact" of assigning a color.
    
    # Let's define a function to check if an AP is currently "dead" or "satisfied" or "pending"
    # Given an AP (a,b,c,d) and current colors (some might be -1):
    # - If all 4 are assigned: check if len(set(colors)) >= 3.
    # - If not all assigned: 
    #   We need to ensure it's possible to satisfy the constraint.
    #   A 4-AP is invalid if it uses <= 2 colors.
    #   This is equivalent to: NOT (all 4 same) AND NOT (exactly 2 colors used).
    
    # Instead of complex propagation, we'll use simple forward checking:
    # When we assign a color to pos p, for every AP containing p, 
    # we check if the AP is now impossible to satisfy.
    # How to check if an AP is impossible?
    # Let the assigned colors in the AP be S_assigned.
    # Let the unassigned positions be U.
    # The final set of colors will be S_assigned union {colors of U}.
    # We need |final_set| >= 3.
    # This fails if it is impossible to add colors from U to S_assigned to make size >= 3.
    # This happens if:
    # 1. |S_assigned| == 0: We need to assign colors to all 4. We can always make size 3 or 4. (Unless domains are restricted)
    # 2. |S_assigned| == 1: Say color c. We need at least one of U to take a color != c.
    #    If all domains of U are subsets of {c}, it's impossible.
    # 3. |S_assigned| == 2: Say colors {c1, c2}. We need at least one of U to take a color not in {c1, c2}.
    #    If all domains of U are subsets of {c1, c2}, it's impossible.
    # 4. |S_assigned| == 3: We already have 3 colors. The 4th can be anything. (Always possible unless domain is empty)
    # 5. |S_assigned| == 4: Check if valid.
    
    # This logic is easy to implement.

    def is_ap_possible(ap_idx, colors, domains, k):
        ap = aps[ap_idx]
        assigned_colors = set()
        unassigned_pos = []
        for pos in ap:
            if colors[pos] != -1:
                assigned_colors.add(colors[pos])
            else:
                unassigned_pos.append(pos)
        
        num_assigned = len(assigned_colors)
        
        if num_assigned == 4:
            return len(assigned_colors) >= 3
        
        if num_assigned == 3:
            # We have 3 distinct colors. The 4th position just needs a valid color from its domain.
            # It can be any of the 3 or a new one.
            # So it's possible if the domain of the unassigned pos is not empty.
            for pos in unassigned_pos:
                if not domains[pos]:
                    return False
            return True
            
        if num_assigned == 2:
            # We have 2 distinct colors, say A and B.
            # We need at least one unassigned position to take a color NOT in {A, B}.
            # If for ALL unassigned positions, their domain is a subset of {A, B}, then 
            # any assignment will result in colors subset of {A, B}, so max 2 colors.
            # So it's impossible if for all u in unassigned_pos, domains[u] <= {A, B}.
            for pos in unassigned_pos:
                # If there exists a color in domain[pos] that is not in assigned_colors, 
                # then we can potentially make it 3 colors.
                # Wait, if we pick a color c not in assigned, the set becomes {A, B, c}, size 3. Valid.
                # So we just need to check if it's possible to pick such a color.
                # If there is at least one unassigned pos with a color outside assigned_colors in its domain, 
                # we are good? 
                # Yes, because we can assign that color to that pos.
                # But what if we have multiple unassigned? We just need ONE of them to take a new color.
                # So if ANY unassigned pos has a domain containing a color not in assigned_colors, 
                # we can force that pos to take that color (in our hypothetical check).
                # However, domains are shared. But for "possibility" check, we assume we can choose freely.
                # So: Is it impossible?
                # Impossible if for ALL unassigned pos, domains[pos] is subset of assigned_colors.
                possible = False
                for pos in unassigned_pos:
                    # Check if domain has any color not in assigned_colors
                    for c in domains[pos]:
                        if c not in assigned_colors:
                            possible = True
                            break
                    if possible:
                        break
                return possible
            return False # Should not reach here if unassigned_pos is empty, but num_assigned==2 implies 2 unassigned.

        if num_assigned == 1:
            # We have 1 distinct color, say A.
            # We need at least 2 unassigned positions to take colors NOT in {A}? 
            # No. We need the final set to have size >= 3.
            # Current set: {A}.
            # We have 3 unassigned positions.
            # If we assign colors c1, c2, c3 to them.
            # Final set is {A, c1, c2, c3}.
            # We need |{A, c1, c2, c3}| >= 3.
            # This fails only if {c1, c2, c3} subset of {A}. i.e., all are A.
            # So it is impossible if ALL unassigned positions are forced to take A.
            # i.e., if for ALL unassigned pos, domains[pos] is subset of {A}.
            possible = False
            for pos in unassigned_pos:
                for c in domains[pos]:
                    if c not in assigned_colors:
                        possible = True
                        break
                if possible:
                    break
            return possible
            # Wait, if I have {A} and unassigned u1, u2, u3.
            # If u1 takes B, u2 takes A, u3 takes A. Set is {A, B}. Size 2. Invalid.
            # So I need AT LEAST TWO new colors? Or one new color and one more?
            # If I have {A}, and I add B, C. Set {A, B, C}. Size 3.
            # If I add B, B. Set {A, B}. Size 2.
            # If I add B, A. Set {A, B}. Size 2.
            # So I need the set of colors assigned to unassigned positions to contain at least 2 colors distinct from A?
            # Or 1 color distinct from A and... wait.
            # Let S = {A}.
            # We add X (set of colors from unassigned).
            # We need |S union X| >= 3.
            # If X is empty (no unassigned, but we have 3 unassigned), this case doesn't apply.
            # If X = {B}, |{A, B}| = 2. Fail.
            # If X = {B, C}, |{A, B, C}| = 3. Pass.
            # If X = {B, B}, X is {B}. Fail.
            # So we need the unassigned positions to be able to generate at least 2 distinct colors not in S?
            # Or 1 color not in S, and... no.
            # Actually, if I have 3 unassigned spots, and I can assign B to one and C to another, I succeed.
            # So the condition for impossibility is:
            # It is impossible to assign colors to u1, u2, u3 such that the set of values {c1, c2, c3} 
            # combined with {A} has size >= 3.
            # This is equivalent to: It is impossible to have {c1, c2, c3} contain at least 2 elements outside {A}?
            # No. If c1=B, c2=C, c3=A. {A, B, C} size 3.
            # If c1=B, c2=B, c3=C. {A, B, C} size 3.
            # If c1=B, c2=B, c3=B. {A, B} size 2. Fail.
            # So we need the unassigned colors to NOT all be the same color (if that color is not A) 
            # AND NOT all be A?
            # Actually, simpler: 
            # The set of colors used by unassigned positions, let's call it Y.
            # We need |{A} union Y| >= 3.
            # This means Y must contain at least 2 colors, one of which is not A? 
            # If Y = {B, C}, union is {A, B, C}. Size 3.
            # If Y = {B}, union is {A, B}. Size 2.
            # So Y must have size at least 2? 
            # If Y = {B, B}, Y as a set is {B}. Size 1.
            # So we need the set of distinct colors assigned to unassigned positions to have size at least 2?
            # Wait. If I assign B to u1, B to u2, B to u3. Y={B}. Union {A, B}. Size 2.
            # If I assign B to u1, C to u2, C to u3. Y={B, C}. Union {A, B, C}. Size 3.
            # So yes, we need the unassigned positions to be able to take at least 2 distinct colors.
            # BUT, one of them could be A?
            # If I assign A to u1, B to u2, C to u3. Y={A, B, C}. Union {A, B, C}. Size 3.
            # So we need the unassigned positions to be able to take at least 2 distinct colors, 
            # such that at least one is not A? 
            # Actually, if Y has size 2, say {B, C}. If A is not B or C, union size is 3.
            # If A is B, union is {B, C} size 2? No, {A, B, C} where A=B is {B, C}. Size 2.
            # So if A is one of the colors in Y, we need Y to have size 3?
            # Let's re-evaluate.
            # We need |{A} union Y| >= 3.
            # Case 1: A not in Y. Then |Y| >= 2.
            # Case 2: A in Y. Then |Y| >= 3.
            
            # So, is it possible?
            # We have 3 unassigned positions.
            # Can we assign colors to them such that the condition holds?
            # This is a small sub-problem.
            # We can just check if there exists an assignment.
            # Given small k, we can brute force the 3 positions?
            # Or simpler: 
            # If there are at least 2 colors available in the union of domains of unassigned positions, 
            # and at least one of them is not A, AND (if A is in the available colors, we need a 3rd color?)
            
            # Let's just use a simple check:
            # Try all combinations of colors for the 3 unassigned positions from their domains?
            # Domains might be large.
            # But k is small (3, 4, 5).
            # So we can just iterate through possible assignments for the 3 unassigned spots.
            
            # Actually, for num_assigned == 1, we have 3 unassigned.
            # We can just check if it's possible to pick c1, c2, c3 from domains[u1], domains[u2], domains[u3]
            # such that |{A, c1, c2, c3}| >= 3.
            
            # For efficiency, we can precompute or just do a small loop.
            # Since this is called often, let's optimize.
            # But first, let's get the logic right.
            
            # For now, let's return True (optimistic) and let the backtracking handle it?
            # No, forward checking is key to pruning.
            # Let's implement a helper for this specific case.
            
            # Helper: can_satisfy_1_assigned(ap_idx, colors, domains, k)
            # We'll write a generic check for the AP.
            
            return True # Placeholder, will implement proper check below

        if num_assigned == 0:
            # 4 unassigned.
            # We need to be able to assign 4 colors such that >= 3 distinct.
            # This is possible if we can pick at least 3 distinct colors from the union of domains?
            # Or simpler: if we can pick 3 distinct colors for any 3 of the positions.
            # Since k is small, we can check if the union of domains has size >= 3?
            # Not quite. If domains are {0}, {0}, {1}, {1}. Union {0,1}. Size 2.
            # Can we assign 0,0,1,1? Set {0,1}. Size 2. Fail.
            # Can we assign 0,1,0,1? Set {0,1}. Fail.
            # So if union of domains has size < 3, it's impossible.
            # If union has size >= 3, is it always possible?
            # Yes, we can pick 3 distinct colors from the union and assign them to 3 positions.
            # The 4th can be anything.
            # So check: len(union(domains[u] for u in unassigned)) >= 3.
            
            union_dom = set()
            for pos in unassigned_pos:
                union_dom.update(domains[pos])
            return len(union_dom) >= 3

        return True

    # Let's refine the check for num_assigned == 1 and 2 using brute force for the unassigned parts
    # since k is small.
    
    def check_ap_possible(ap_idx, colors, domains, k):
        ap = aps[ap_idx]
        assigned = []
        unassigned = []
        for pos in ap:
            if colors[pos] != -1:
                assigned.append(colors[pos])
            else:
                unassigned.append(pos)
        
        num_a = len(assigned)
        num_u = len(unassigned)
        
        if num_u == 0:
            return len(set(assigned)) >= 3
        
        # If k is small, we can iterate through all possible assignments for unassigned positions
        # to see if ANY assignment leads to a valid AP.
        # Number of unassigned is 1, 2, or 3.
        # Max combinations: k^3. If k=5, 125. Very fast.
        
        # Get domains for unassigned
        doms = [domains[pos] for pos in unassigned]
        
        # Recursive helper to try assignments
        def try_assign(idx, current_colors):
            if idx == len(unassigned):
                # Check if current_colors + assigned forms a valid set
                all_colors = assigned + current_colors
                return len(set(all_colors)) >= 3
            
            pos = unassigned[idx]
            for c in doms[idx]:
                if try_assign(idx + 1, current_colors + [c]):
                    return True
            return False
            
        return try_assign(0, [])

    def solve(k):
        colors = [-1] * n
        domains = [set(range(k)) for _ in range(n)]
        
        # Symmetry breaking: Fix first few positions
        # color[0] = 0
        # color[1] = 1 (if k > 1)
        # This reduces the search space significantly.
        
        # We'll do a recursive backtracking
        # Order: MRV (Minimum Remaining Values)
        
        import sys
        sys.setrecursionlimit(10000)
        
        # Precompute list of positions to iterate? 
        # Dynamic MRV is better.
        
        # To speed up, we can precompute the "neighbors" (APs) for each position.
        # We already have pos_to_aps.
        
        # Forward Checking:
        # When we assign colors[pos] = c, we update domains of other positions in APs containing pos.
        
        def backtrack():
            # Find unassigned position with smallest domain
            min_size = k + 1
            min_pos = -1
            for i in range(n):
                if colors[i] == -1:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        min_pos = i
                        if min_size == 0:
                            return False
            if min_pos == -1:
                return True # All assigned
            
            # Try each color in domain of min_pos
            for c in list(domains[min_pos]):
                # Assign
                colors[min_pos] = c
                domains[min_pos] = {c}
                
                # Propagate
                consistent = True
                # For each AP containing min_pos, check if it's still possible
                # And update domains of other positions in those APs?
                # Simple FC: Just check if AP is possible.
                # Stronger FC: Reduce domains of other unassigned positions in the AP.
                
                # Let's do simple FC first: check if AP is possible.
                # If not, revert and try next color.
                
                affected_aps = pos_to_aps[min_pos]
                for ap_idx in affected_aps:
                    if not check_ap_possible(ap_idx, colors, domains, k):
                        consistent = False
                        break
                
                if consistent:
                    # Recurse
                    if backtrack():
                        return True
                else:
                    # Revert
                    colors[min_pos] = -1
                    domains[min_pos] = set(range(k)) # Restore original domain? 
                    # Wait, we modified domains[min_pos] to {c}. 
                    # We need to restore it to its previous state.
                    # This requires saving state or undoing changes.
                    # Since we only changed domains[min_pos], we can just restore it.
                    # But did we change other domains? 
                    # In this simple FC, we did NOT change other domains.
                    # So just restoring domains[min_pos] is enough.
                    colors[min_pos] = -1
                    domains[min_pos] = set(range(k)) # This is WRONG. We need the original domain.
                    
            # Revert after loop
            colors[min_pos] = -1
            # Restore domain
            # We need to keep track of original domains.
            # Let's pass domains as a parameter or use a stack.
            # For simplicity in this rewrite, let's assume we don't prune domains of neighbors, 
            # only check possibility.
            # But we MUST restore the domain of min_pos correctly.
            
            # Let's restructure to save the domain before assignment.
            return False
        
        # The above backtrack is flawed because it doesn't restore domains correctly 
        # if we had modified them in deeper recursion.
        # Let's use a cleaner approach: 
        # Copy domains? Too expensive.
        # Use a trail/undo log.
        
        # Given time constraints, let's try a simpler heuristic:
        # Just use random restarts with a simple greedy+backtrack?
        # Or use the library? No standard library for CSP.
        
        # Let's try a different structural approach:
        # Since n=128 is small, and k is small, we can use a SAT solver approach 
        # implemented in Python? Or just a very optimized backtracker.
        
        # Let's try to implement a proper backtracker with undo.
        
        class CSP:
            def __init__(self, k):
                self.k = k
                self.colors = [-1] * n
                self.domains = [set(range(k)) for _ in range(n)]
                self.trail = []
                
            def assign(self, pos, c):
                old_dom = self.domains[pos]
                self.colors[pos] = c
                self.domains[pos] = {c}
                self.trail.append((pos, old_dom))
                
            def unassign(self):
                if self.trail:
                    pos, old_dom = self.trail.pop()
                    self.colors[pos] = -1
                    self.domains[pos] = old_dom
                    
            def solve(self):
                # MRV
                min_size = self.k + 1
                min_pos = -1
                for i in range(n):
                    if self.colors[i] == -1:
                        if len(self.domains[i]) < min_size:
                            min_size = len(self.domains[i])
                            min_pos = i
                            if min_size == 0:
                                return False
                if min_pos == -1:
                    return True
                    
                for c in list(self.domains[min_pos]):
                    self.assign(min_pos, c)
                    
                    # Check constraints
                    ok = True
                    for ap_idx in pos_to_aps[min_pos]:
                        if not check_ap_possible(ap_idx, self.colors, self.domains, self.k):
                            ok = False
                            break
                            
                    if ok:
                        if self.solve():
                            return True
                            
                    self.unassign()
                    
                return False
                
        csp = CSP(k)
        # Symmetry breaking
        csp.assign(0, 0)
        if k > 1:
            csp.assign(1, 1)
            
        if csp.solve():
            return csp.colors
        return None

    # Try k=3, 4, 5...
    for k in range(3, 12):
        if time.time() > deadline - 5.0:
            break
        start_time = time.time()
        result = solve(k)
        if result is not None:
            return result
        # If solve takes too long, break?
        # The solve function is recursive and doesn't have a time limit inside.
        # We need to add a time limit to the backtracking.
        
    # Fallback: Identity
    return list(range(n))

# The above code is a draft. The `solve` function inside `construct` doesn't respect the deadline.
# We need to integrate the time limit into the backtracking.

# Let's rewrite with a global time limit check in the recursion.

def construct():
    n = 128
    deadline = time.time() + 85.0

    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
            
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def check_ap_possible(ap_idx, colors, domains, k):
        ap = aps[ap_idx]
        assigned = []
        unassigned = []
        for pos in ap:
            if colors[pos] != -1:
                assigned.append(colors[pos])
            else:
                unassigned.append(pos)
        
        num_u = len(unassigned)
        if num_u == 0:
            return len(set(assigned)) >= 3
        
        doms = [domains[pos] for pos in unassigned]
        
        def try_assign(idx, current_colors):
            if time.time() > deadline:
                return False
            if idx == len(unassigned):
                all_colors = assigned + current_colors
                return len(set(all_colors)) >= 3
            
            pos = unassigned[idx]
            for c in doms[idx]:
                if try_assign(idx + 1, current_colors + [c]):
                    return True
            return False
            
        return try_assign(0, [])

    def solve(k):
        colors = [-1] * n
        domains = [set(range(k)) for _ in range(n)]
        
        # Symmetry breaking
        colors[0] = 0
        domains[0] = {0}
        if k > 1:
            colors[1] = 1
            domains[1] = {1}
            
        # Recursive backtrack
        def backtrack():
            if time.time() > deadline:
                return False
                
            min_size = k + 1
            min_pos = -1
            for i in range(n):
                if colors[i] == -1:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        min_pos = i
                        if min_size == 0:
                            return False
            if min_pos == -1:
                return True
            
            # Try colors
            for c in list(domains[min_pos]):
                colors[min_pos] = c
                old_dom = domains[min_pos]
                domains[min_pos] = {c}
                
                ok = True
                for ap_idx in pos_to_aps[min_pos]:
                    if not check_ap_possible(ap_idx, colors, domains, k):
                        ok = False
                        break
                        
                if ok:
                    if backtrack():
                        return True
                        
                colors[min_pos] = -1
                domains[min_pos] = old_dom
                
            return False
            
        return backtrack()

    for k in range(3, 12):
        if time.time() > deadline - 5.0:
            break
        res = solve(k)
        if res is not None:
            return res
            
    return list(range(n))

