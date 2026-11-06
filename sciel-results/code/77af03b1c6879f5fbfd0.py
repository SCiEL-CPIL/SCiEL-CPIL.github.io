import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    limit = 52.0  # Safety margin for 60s limit (90s hard kill)
    
    n = 256
    weights = [0.0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = 1.0 / i

    # Precompute conflict masks for fast SA moves
    # conflict_mask[x] has bit y set if x and y form a 3-AP with some third element
    # Actually, we need: two elements x, y conflict if there exists z such that {x,y,z} is a 3-AP
    # But for SA, we check if adding v to S creates any 3-AP.
    # A 3-AP involving v and existing elements: for each u in S, check if there's w in S such that {u,v,w} is AP.
    # More efficiently: precompute for each pair (u,v) whether they're in a 3-AP together.
    # Actually, the standard approach: x and y are "conflicting" if they appear in some 3-AP.
    # But that's not quite right either. Let me think...
    # 
    # If S is current set, adding v is valid iff for all u in S, there is no w in S ∪ {v} such that {u,v,w} is a 3-AP.
    # This means: for each u in S, check if u and v are in a 3-AP with some w that's also in S (or w=v, but that's constant AP).
    # 
    # Alternative: precompute conflict_mask[x] = set of all y such that x and y are in at least one 3-AP.
    # Then adding v to S is valid iff no u in S has v in conflict_mask[u].
    # Wait, that's not sufficient. x and y being in a 3-AP means there exists z with {x,y,z} AP.
    # If x and y are both in S and they're in a 3-AP, then we need z NOT in S.
    # So the constraint is: for every 3-AP {a,b,c}, at most 2 of {a,b,c} are in S.
    # 
    # For efficient checking: when adding v, for each u in S, check if there exists w in S such that {u,v,w} is a 3-AP.
    # The w values are determined: if v is middle, w = 2v - u. If u is middle, w = 2u - v. If u is end and v is end, w = (u+v)/2.
    # 
    # Let's precompute: for each pair (u,v), list the "required" w values that would form a 3-AP.
    # Actually simpler: just check on the fly since |S| is small (~100-150 elements).
    
    # Generate all 3-AP constraints for MILP
    ap_constraints = []
    for a in range(1, n + 1):
        max_d = (n - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            ap_constraints.append((a - 1, b - 1, c - 1))
            
    num_constraints = len(ap_constraints)
    
    # Build sparse constraint matrix
    rows = []
    cols = []
    data = []
    for i, (a, b, c) in enumerate(ap_constraints):
        rows.extend([i, i, i])
        cols.extend([a, b, c])
        data.extend([1, 1, 1])
        
    A = csr_matrix((data, (rows, cols)), shape=(num_constraints, n))
    b_ub = np.full(num_constraints, 2.0)
    
    # Objective: maximize sum(weights[i] * x_i) -> minimize -sum
    c_obj = np.array([-weights[i] for i in range(1, n + 1)])
    
    integrality = np.ones(n)
    lin_cons = LinearConstraint(A, lb=None, ub=b_ub)
    var_bounds = Bounds(lb=0, ub=1)
    
    best_S = set()
    best_score = 0.0
    
    # Try MILP first
    try:
        res = milp(c=c_obj, constraints=lin_cons, integrality=integrality, bounds=var_bounds,
                   options={'time_limit': 30, 'mip_gap': 0.05})
        
        if res.x is not None:
            solution_mask = (res.x > 0.5).astype(int)
            selected_indices = [i + 1 for i in range(n) if solution_mask[i] == 1]
            
            if selected_indices:
                # Verify validity
                S_set = set(selected_indices)
                valid = True
                for a, b, c in ap_constraints:
                    count = sum(1 for x in [a+1, b+1, c+1] if x in S_set)
                    if count >= 3:
                        valid = False
                        break
                
                if valid:
                    score = sum(weights[x] for x in S_set)
                    if score > best_score:
                        best_score = score
                        best_S = S_set
    except Exception:
        pass
    
    # Fallback / initial solution: greedy
    if not best_S:
        S = set()
        for x in range(1, n + 1):
            valid = True
            for y in S:
                # Check if {x, y, z} forms a 3-AP for some z in S
                # x is middle: z = 2x - y
                z = 2 * x - y
                if 1 <= z <= n and z in S:
                    valid = False
                    break
                # y is middle: z = 2y - x
                z = 2 * y - x
                if 1 <= z <= n and z in S:
                    valid = False
                    break
                # x and y are ends: z = (x+y)/2
                if (x + y) % 2 == 0:
                    z = (x + y) // 2
                    if 1 <= z <= n and z in S:
                        valid = False
                        break
            if valid:
                S.add(x)
        
        score = sum(weights[x] for x in S)
        if score > best_score:
            best_score = score
            best_S = S
    
    # Simulated Annealing to refine
    if time.time() - start_time < limit - 5:  # Leave 5s buffer
        S = set(best_S)
        current_score = best_score
        
        # Precompute for faster conflict checking
        # For each element, precompute which other elements it conflicts with
        # conflict[x] = set of y such that x and y are in a 3-AP
        # Actually, let's just use a helper function
        
        def can_add(v, S_set, weights_local):
            """Check if v can be added to S_set without creating a 3-AP"""
            for u in S_set:
                # Check all 3-APs involving v and u
                # v is middle: w = 2v - u
                w = 2 * v - u
                if 1 <= w <= n and w in S_set and w != v and w != u:
                    return False
                # u is middle: w = 2u - v
                w = 2 * u - v
                if 1 <= w <= n and w in S_set and w != v and w != u:
                    return False
                # v and u are ends: w = (v+u)/2
                if (v + u) % 2 == 0:
                    w = (v + u) // 2
                    if 1 <= w <= n and w in S_set and w != v and w != u:
                        return False
            return True
        
        T = 0.02
        cooling_rate = 0.9995
        min_T = 0.0005
        
        all_nodes = list(range(1, n + 1))
        
        while time.time() - start_time < limit:
            if T < min_T:
                T = min_T
            
            # Try to add a random element
            add_success = False
            for _ in range(3):
                v = random.choice(all_nodes)
                if v not in S:
                    if can_add(v, S, weights):
                        S.add(v)
                        current_score += weights[v]
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(S)
                        add_success = True
                        break
            
            if add_success:
                T *= 0.999
                continue
            
            # Try swap: remove u, add v
            if S:
                u = random.choice(list(S))
                v = random.choice(all_nodes)
                
                if v == u:
                    continue
                
                # Check if we can remove u and add v
                S_temp = S - {u}
                if can_add(v, S_temp, weights):
                    delta = weights[v] - weights[u]
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        S = S_temp | {v}
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(S)
                        T *= 0.999
                    else:
                        T *= cooling_rate
                else:
                    T *= cooling_rate
            else:
                T *= cooling_rate
    
    return list(best_S)
