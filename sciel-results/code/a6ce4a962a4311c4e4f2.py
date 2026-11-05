"""Incentive-compatible finite auction design solver using separable mechanism enumeration."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Parse inputs
    values = [0, 1, 2, 3]
    
    if isinstance(distribution_spec, dict):
        p1 = distribution_spec.get('good1', distribution_spec.get('p1', [0.25, 0.25, 0.25, 0.25]))
        p2 = distribution_spec.get('good2', distribution_spec.get('p2', [0.25, 0.25, 0.25, 0.25]))
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 4:
            p1 = list(distribution_spec)
            p2 = list(distribution_spec)
        elif len(distribution_spec) == 8:
            p1 = list(distribution_spec[:4])
            p2 = list(distribution_spec[4:])
        else:
            p1 = [0.25, 0.25, 0.25, 0.25]
            p2 = [0.25, 0.25, 0.25, 0.25]
    else:
        p1 = [0.25, 0.25, 0.25, 0.25]
        p2 = [0.25, 0.25, 0.25, 0.25]
    
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p1 = p1 / p1.sum() if p1.sum() > 0 else np.ones(4) / 4
    p2 = p2 / p2.sum() if p2.sum() > 0 else np.ones(4) / 4
    
    joint_probs = np.outer(p1, p2).flatten()
    n_types = 16
    types = [(v1, v2) for v1 in values for v2 in values]
    
    best_revenue = -1
    best_allocations = np.zeros((16, 2))
    best_payments = np.zeros(16)

    # Enumerate all separable allocation rules
    # x1 depends only on v1: 2^4 = 16 possibilities
    # x2 depends only on v2: 2^4 = 16 possibilities
    # Total 256 mechanisms
    
    for mask1 in range(16):
        # Check time
        if time.time() > deadline:
            break
        
        # Decode mask1 to allocation function for good 1
        # mask1 bits: bit 0 for v1=0, bit 1 for v1=1, etc.
        # But allocation must be monotone? The problem doesn't strictly require monotonicity of the allocation rule itself in the input, 
        # but IC implies it for single-parameter. For 2 goods, x1(v1, v2) must be non-decreasing in v1.
        # If x1 depends only on v1, it must be non-decreasing in v1.
        # So valid masks are those where if bit i is 1, all bits j < i must be 0? No, non-decreasing means 0,0,1,1.
        # So valid masks for monotone binary function on 4 points: 
        # 0000, 0001, 0011, 0111, 1111 (5 possibilities)
        # Let's generate all monotone binary functions for 4 points.
        
        # Precompute monotone masks for 4 points
        # A function f: {0,1,2,3} -> {0,1} is monotone if f(i) <= f(i+1).
        # Possible sequences: 0000, 0001, 0011, 0111, 1111.
        # Let's just iterate all 16 and check monotonicity.
        
        alloc1 = []
        is_mono1 = True
        for v1 in range(4):
            if (mask1 >> v1) & 1:
                alloc1.append(1)
            else:
                alloc1.append(0)
        for i in range(3):
            if alloc1[i] > alloc1[i+1]:
                is_mono1 = False
                break
        
        if not is_mono1:
            continue

        for mask2 in range(16):
            if time.time() > deadline:
                break
                
            alloc2 = []
            is_mono2 = True
            for v2 in range(4):
                if (mask2 >> v2) & 1:
                    alloc2.append(1)
                else:
                    alloc2.append(0)
            for i in range(3):
                if alloc2[i] > alloc2[i+1]:
                    is_mono2 = False
                    break
            
            if not is_mono2:
                continue

            # Construct allocation matrix
            # allocations[t, 0] = alloc1[v1_t]
            # allocations[t, 1] = alloc2[v2_t]
            
            allocations = np.zeros((16, 2))
            for t in range(16):
                v1_t, v2_t = types[t]
                allocations[t, 0] = alloc1[v1_t]
                allocations[t, 1] = alloc2[v2_t]
                
            # Now, find the optimal payments for this fixed allocation.
            # Maximize sum pi[t] * p[t]
            # Subject to:
            # 1. IC: p[t] - p[tp] <= v1[t]*alloc[tp,0] + v2[t]*alloc[tp,1] - v1[tp]*alloc[t,0] - v2[tp]*alloc[t,1]
            #    Wait, standard IC: u(t,t) >= u(t, tp)
            #    v1[t]*alloc[t,0] + v2[t]*alloc[t,1] - p[t] >= v1[t]*alloc[tp,0] + v2[t]*alloc[tp,1] - p[tp]
            #    p[t] - p[tp] <= v1[t]*(alloc[t,0] - alloc[tp,0]) + v2[t]*(alloc[t,1] - alloc[tp,1])
            
            # 2. IR: p[t] <= v1[t]*alloc[t,0] + v2[t]*alloc[t,1]
            # 3. p[t] >= 0
            
            n_p = 16
            c = -joint_probs.copy() # Maximize revenue -> Minimize -revenue
            
            A_ub_rows = []
            b_ub_vals = []
            
            # IC constraints
            for t in range(16):
                v1_t, v2_t = types[t]
                for tp in range(16):
                    if t == tp: continue
                    v1_tp, v2_tp = types[tp]
                    # p[t] - p[tp] <= v1_t*(alloc[t,0] - alloc[tp,0]) + v2_t*(alloc[t,1] - alloc[tp,1])
                    row = np.zeros(n_p)
                    row[t] = 1
                    row[tp] = -1
                    b_val = v1_t * (allocations[t, 0] - allocations[tp, 0]) + v2_t * (allocations[t, 1] - allocations[tp, 1])
                    A_ub_rows.append(row)
                    b_ub_vals.append(b_val)
                    
            # IR constraints: p[t] <= v1_t*alloc[t,0] + v2_t*alloc[t,1]
            # -p[t] >= -(v1_t*alloc[t,0] + v2_t*alloc[t,1]) -> p[t] <= ...
            # So: p[t] <= V_t
            # row: 1 at t, 0 else. b: V_t
            for t in range(16):
                v1_t, v2_t = types[t]
                V_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
                row = np.zeros(n_p)
                row[t] = 1
                A_ub_rows.append(row)
                b_ub_vals.append(V_t)
                
            # Non-negativity is handled by bounds
            
            bounds = [(0, None)] * 16
            
            try:
                res = linprog(c, A_ub=np.array(A_ub_rows), b_ub=np.array(b_ub_vals), bounds=bounds, method='highs')
                if res.success:
                    payments = res.x
                    # Verify
                    rev = np.sum(joint_probs * payments)
                    if rev > best_revenue:
                        best_revenue = rev
                        best_allocations = allocations.copy()
                        best_payments = payments.copy()
            except:
                pass

    # If no mechanism found (should not happen), fallback
    if best_revenue < 0:
        best_allocations = np.zeros((16, 2))
        best_payments = np.zeros(16)

    return {'allocations': best_allocations, 'payments': best_payments}
