import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115.0
    
    def get_seed_32():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        
        log = {}
        x = 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
            
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return 32, colors

    def try_grow(n_target, time_limit):
        n, colors_list = get_seed_32()
        M = np.zeros((n_target, n_target), dtype=np.int8)
        
        idx = 0
        for i in range(32):
            for j in range(i+1, 32):
                M[i,j] = colors_list[idx]
                M[j,i] = colors_list[idx]
                idx += 1
                
        current_n = 32
        
        # CSP Solver for adding a new vertex k
        def add_vertex_csp(k, M, current_n):
            # We want to assign colors to M[k, 0], ..., M[k, k-1]
            # Variables: x_i = M[k, i] for i in 0..k-1
            # Constraints: For all i < j < k, x_i, x_j, M[i,j] are not all equal.
            
            # Domains
            domains = [set([0,1,2,3]) for _ in range(k)]
            
            # Forward checking state
            # We will process i in order 0 to k-1.
            # When we assign x_i, we update domains of x_j for j > i.
            
            # To speed up, we can order variables by "most constrained first"
            # But the constraints are forward-looking in the sequential assignment.
            # Actually, the constraint graph is a complete graph on 0..k-1.
            # Variable x_i is constrained by x_j for all j != i.
            # But in our sequential assignment, we only look at j < i for the "hard" constraint check?
            # No, the constraint is symmetric.
            # If we assign x_i, it affects x_j for j > i.
            # So we must process in increasing i.
            
            def backtrack(i, assignments):
                if i == k:
                    return True
                
                # Order of values: try colors that cause least constraint?
                # Just try 0,1,2,3
                for val in domains[i]:
                    # Check if val is consistent with previous assignments
                    # Constraint: for all j < i, x_j, val, M[j,i] not all equal.
                    consistent = True
                    for j in range(i):
                        if assignments[j] == val and assignments[j] == M[j, i]:
                            consistent = False
                            break
                    
                    if not consistent:
                        continue
                    
                    # Assign
                    assignments[i] = val
                    M[k, i] = val
                    M[i, k] = val
                    
                    # Forward checking: update domains for j > i
                    # For each j > i, if val == M[i, j], then x_j cannot be val.
                    new_domains = [None] * k
                    # Copy current domains? No, we need to restore.
                    # We will store the changes.
                    changes = []
                    feasible = True
                    for j in range(i + 1, k):
                        # Check if val is forbidden for j due to this assignment
                        if val == M[i, j]:
                            if val in domains[j]:
                                domains[j].remove(val)
                                changes.append((j, val))
                                if not domains[j]:
                                    feasible = False
                                    break
                        
                    if feasible:
                        if backtrack(i + 1, assignments):
                            return True
                    
                    # Undo
                    for (j, v) in changes:
                        domains[j].add(v)
                    # Note: We didn't copy domains, so we must be careful.
                    # The `changes` list records what we removed.
                    # We added them back.
                    # But what if `domains[j]` was modified by a deeper recursion?
                    # No, we restore before returning False.
                    # If we return True, we don't restore, which is correct.
                    
                return False

            assignments = [0] * k
            # Initialize domains
            # Actually, the domains are initially {0,1,2,3}.
            # We need to make sure we don't modify the original list of sets if we fail.
            # Let's create fresh sets for each call?
            # k is small (<=50).
            
            # Optimization: Check if any domain is empty initially? No.
            
            # One issue: The `domains` list is shared. We need to reset it or pass it.
            # Let's create a fresh list of sets for each call.
            domains = [set([0,1,2,3]) for _ in range(k)]
            
            if backtrack(0, assignments):
                return True
            return False

        start_time = time.time()
        
        for k in range(current_n, n_target):
            if time.time() > start_time + time_limit:
                break
                
            # Try to add vertex k using CSP
            if add_vertex_csp(k, M, current_n):
                current_n = k + 1
            else:
                # Failed to add vertex k
                # We can try to "repair" the graph?
                # Or just break.
                # Given the CSP is exact, if it fails, there is no way to add k
                # with the current coloring of the first k vertices.
                # So we must break.
                break
        
        final_colors = []
        for i in range(current_n):
            for j in range(i+1, current_n):
                final_colors.append(int(M[i,j]))
                
        return current_n, final_colors

    n, colors = try_grow(50, 110.0)
    
    return {'n': n, 'colors': colors}
