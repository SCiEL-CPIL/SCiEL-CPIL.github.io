import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Linearized Greedy Construction
    # We build the permutation sequentially.
    # At each step, we assign one unassigned facility to one unassigned location.
    # The cost of assigning facility i to location j, given the current partial assignment P,
    # is the marginal increase in the total QAP cost.
    
    # Let S be the set of assigned facilities and L be the set of assigned locations.
    # The total cost is sum_{u,v in S} A[u,v] B[P[u], P[v]].
    # If we add facility i and location j, the new cost is:
    # C_new = C_old + sum_{k in S} A[i,k] B[j, P[k]] + sum_{k in S} A[k,i] B[P[k], j] + A[i,i] B[j,j]
    # Note: The term A[i,i]B[j,j] is included in the sums if we are careful, but let's be explicit.
    # Actually, the sum is over all pairs.
    # Old cost: sum_{u,v in S} A[u,v] B[P[u], P[v]]
    # New cost: sum_{u,v in S} A[u,v] B[P[u], P[v]] 
    #           + sum_{k in S} A[i,k] B[j, P[k]] 
    #           + sum_{k in S} A[k,i] B[P[k], j] 
    #           + A[i,i] B[j,j]
    
    # So the marginal cost Delta(i, j) = sum_{k in S} A[i,k] B[j, P[k]] + sum_{k in S} A[k,i] B[P[k], j] + A[i,i] B[j,j]
    
    # We can maintain a cost matrix C where C[i, j] is the current marginal cost of assigning i to j.
    # Initially, S is empty, so C[i, j] = A[i,i] B[j,j] ? No, if S is empty, the sums are 0.
    # Wait, if S is empty, the first assignment has cost A[i,i]B[j,j].
    # Let's re-verify the formula.
    # Cost = sum_{u,v} A[u,v] B[P[u], P[v]]
    # If we add i->j:
    # New terms involving i or j:
    # 1. u=i, v=i: A[i,i] B[j,j]
    # 2. u=i, v=k (k in S): A[i,k] B[j, P[k]]
    # 3. u=k (k in S), v=i: A[k,i] B[P[k], j]
    # 4. u=i, v=j is not possible since j is not in S yet (it's being added).
    # 5. u=j, v=i is not possible.
    # So Delta(i, j) = A[i,i] B[j,j] + sum_{k in S} (A[i,k] B[j, P[k]] + A[k,i] B[P[k], j])
    
    # We want to minimize the total cost, so at each step we pick the pair (i, j) with the minimum Delta(i, j)
    # among unassigned i and unassigned j.
    
    # To update C efficiently:
    # When we assign i0 -> j0:
    # For any remaining facility i' and location j':
    # New Delta(i', j') = Old Delta(i', j') + A[i', i0] B[j', j0] + A[i0, i'] B[j0, j']
    # Wait, let's derive the update.
    # Delta(i, j) = A[i,i]B[j,j] + sum_{k in S} (A[i,k] B[j, P[k]] + A[k,i] B[P[k], j])
    # After adding i0->j0 to S:
    # Delta_new(i, j) = A[i,i]B[j,j] + sum_{k in S} (A[i,k] B[j, P[k]] + A[k,i] B[P[k], j]) + A[i,i0] B[j, j0] + A[i0,i] B[j0, j]
    # So Delta_new(i, j) = Delta_old(i, j) + A[i, i0] B[j, j0] + A[i0, i] B[j0, j]
    
    # This update is O(n^2) for the entire matrix C if we do it naively for all i, j.
    # But we only need to update C for unassigned i and j.
    # Number of unassigned is n-1. So O((n-1)^2) per step.
    # Total cost O(n^3) = 150^3 = 3.3M ops. Very fast.
    
    # Algorithm:
    # 1. Initialize C[i, j] = A[i, i] B[j, j] for all i, j.
    #    Wait, is this correct?
    #    If S is empty, Delta(i, j) = A[i, i] B[j, j]. Yes.
    # 2. Create sets U_F (unassigned facilities) and U_L (unassigned locations), initially all 0..n-1.
    # 3. P = array of -1.
    # 4. For step in range(n):
    #    a. Find (i, j) in U_F x U_L with min C[i, j].
    #    b. Assign P[i] = j.
    #    c. Remove i from U_F, j from U_L.
    #    d. Update C for all remaining i' in U_F, j' in U_L:
    #       C[i', j'] += A[i', i] B[j', j] + A[i, i'] B[j, j']
    
    # This is a greedy construction. It's not perfect, but it provides a structured starting point.
    # Then we refine with local search.
    
    C = np.outer(np.diag(a), np.diag(b))
    
    # We need to efficiently find the min in C over U_F x U_L.
    # We can set C[i, j] = inf for assigned i or j.
    
    U_F = list(range(n))
    U_L = list(range(n))
    
    P = np.full(n, -1, dtype=np.int64)
    
    for step in range(n):
        # Find min in C[U_F, :][:, U_L]
        # For speed, we can just search the whole C and mask out assigned ones.
        # Since n=150, C is 150x150. Searching is fast.
        
        # Mask assigned
        C_masked = C.copy()
        for i in U_F:
            for j in U_L:
                if P[i] != -1 or P[j] != -1: # This check is wrong, P is facility->location
                    pass
        
        # Better: maintain a boolean mask for available locations.
        # Let's just use np.inf for assigned.
        
        # Actually, simpler:
        # Find i in U_F, j in U_L with min C[i,j]
        min_val = np.inf
        min_i, min_j = -1, -1
        for i in U_F:
            for j in U_L:
                if C[i, j] < min_val:
                    min_val = C[i, j]
                    min_i = i
                    min_j = j
        
        # Assign
        P[min_i] = min_j
        
        # Remove from lists
        U_F.remove(min_i)
        U_L.remove(min_j)
        
        # Update C for remaining
        # C[i', j'] += A[i', min_i] B[j', min_j] + A[min_i, i'] B[min_j, j']
        # This is an outer product update.
        # vec_i = A[:, min_i] (column of A)
        # vec_j = B[min_j, :] (row of B)
        # Update term 1: A[i', min_i] B[j', min_j] = vec_i[i'] * vec_j[j']
        # Update term 2: A[min_i, i'] B[min_j, j'] = vec_i^T[i'] * vec_j[j'] ? No.
        # A[min_i, i'] is row min_i of A. Let's call it row_A.
        # B[min_j, j'] is row min_j of B. Let's call it row_B.
        
        col_A = a[:, min_i]
        row_A = a[min_i, :]
        row_B = b[min_j, :]
        # B[j', min_j] is col_B.
        col_B = b[:, min_j]
        
        # Term 1: A[i', min_i] B[j', min_j] = col_A[i'] * col_B[j']
        # Term 2: A[min_i, i'] B[min_j, j'] = row_A[i'] * row_B[j']
        
        # So C[i', j'] += col_A[i'] * col_B[j'] + row_A[i'] * row_B[j']
        
        # We can do this with outer products.
        # But we only update for i' in U_F and j' in U_L.
        # For simplicity and n=150, we can update the whole matrix and mask later,
        # or just update the submatrix.
        
        # Outer product 1: np.outer(col_A, col_B)
        # Outer product 2: np.outer(row_A, row_B)
        
        # C += np.outer(col_A, col_B) + np.outer(row_A, row_B)
        # This is O(n^2) per step. Total O(n^3).
        
        C += np.outer(col_A, col_B) + np.outer(row_A, row_B)
        
        # Set assigned to inf to be ignored in next min search
        C[min_i, :] = np.inf
        C[:, min_j] = np.inf
        
    # Now we have a permutation P.
    # P[i] is the location of facility i.
    
    # Let's verify P is a valid permutation.
    # It should be, by construction.
    
    # Compute initial cost
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = compute_cost(P)
    best_cost = current_cost
    best_p = P.copy()
    
    # Local Search: 2-opt (swap two locations)
    # We will use the same greedy hill climbing with random restarts as before,
    # but starting from this better initial solution.
    
    current_p = P.copy()
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Random restart or perturbation
            for _ in range(5):
                i, j = np.random.choice(n, 2, replace=False)
                current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost = compute_cost(current_p)
            no_improve_count = 0
            continue
            
        i, j = np.random.choice(n, 2, replace=False)
        
        # Swap
        current_p[i], current_p[j] = current_p[j], current_p[i]
        
        new_cost = compute_cost(current_p)
        
        if new_cost < current_cost:
            current_cost = new_cost
            no_improve_count = 0
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = current_p.copy()
        else:
            # Revert
            current_p[i], current_p[j] = current_p[j], current_p[i]
            no_improve_count += 1
            
    return best_p.tolist()
