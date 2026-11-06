# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            for key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    probs = np.array(val, dtype=float)
                    break
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    if len(probs) != n_types:
        if len(probs) == 4:
            probs = np.outer(probs, probs).flatten()
        else:
            probs = np.ones(n_types) / n_types
    probs = probs / probs.sum()

    # Strategy:
    # 1. Generate a set of candidate allocation rules (binary).
    # 2. For each rule, solve a small LP for optimal payments (utilities).
    # 3. Pick the best one.
    
    # Candidate 1: Threshold at 1
    q1 = np.array([[1.0 if v[0] >= 1 else 0.0, 1.0 if v[1] >= 1 else 0.0] for v in types])
    # Candidate 2: Threshold at 2
    q2 = np.array([[1.0 if v[0] >= 2 else 0.0, 1.0 if v[1] >= 2 else 0.0] for v in types])
    # Candidate 3: Threshold at 1 for good 1, 2 for good 2
    q3 = np.array([[1.0 if v[0] >= 1 else 0.0, 1.0 if v[1] >= 2 else 0.0] for v in types])
    # Candidate 4: Always allocate
    q4 = np.ones((n_types, 2))
    # Candidate 5: Never allocate
    q5 = np.zeros((n_types, 2))
    
    candidates = [q1, q2, q3, q4, q5]
    
    best_revenue = -1
    best_alloc = None
    best_pay = None
    
    for q in candidates:
        # Solve for U(v) to maximize revenue
        # Maximize sum pi(v) * (v.q(v) - U(v))
        # Minimize sum pi(v) * U(v) - sum pi(v) * v.q(v)
        # Let c[v] = -pi(v)
        c = -probs
        
        # Constraints:
        # 1. U(v) >= 0
        # 2. U(v) >= U(v') + v.q(v) - v.q(v')
        #    U(v) - U(v') >= v.q(v) - v.q(v')
        #    -U(v) + U(v') <= v.q(v') - v.q(v)
        
        A_rows = []
        b_rows = []
        
        # IR: -U(v) <= 0
        for v in range(n_types):
            row = np.zeros(n_types)
            row[v] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
        
        # IC: For all v, v'
        for v in range(n_types):
            for v_prime in range(n_types):
                if v == v_prime:
                    continue
                v1, v2 = types[v]
                vp1, vp2 = types[v_prime]
                # Utility difference
                # U(v) - U(v') >= v1*q1(v) + v2*q2(v) - (vp1*q1(v') + vp2*q2(v'))
                # -U(v) + U(v') <= vp1*q1(v') + vp2*q2(v') - (v1*q1(v) + v2*q2(v))
                
                row = np.zeros(n_types)
                row[v] = -1.0
                row[v_prime] = 1.0
                
                val_v = v1 * q[v, 0] + v2 * q[v, 1]
                val_vp = vp1 * q[v_prime, 0] + vp2 * q[v_prime, 1]
                b_val = val_vp - val_v
                
                A_rows.append(row)
                b_rows.append(b_val)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        # Bounds: U(v) >= 0
        bounds = [(0, None)] * n_types
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                U = res.x
                # Calculate payments
                payments = np.zeros(n_types)
                for v in range(n_types):
                    v1, v2 = types[v]
                    payments[v] = v1 * q[v, 0] + v2 * q[v, 1] - U[v]
                
                # Verify feasibility strictly
                valid = True
                for v in range(n_types):
                    if payments[v] < -1e-8:
                        valid = False
                        break
                    if U[v] < -1e-8:
                        valid = False
                        break
                
                if valid:
                    # Check IC strictly
                    for v in range(n_types):
                        for v_prime in range(n_types):
                            v1, v2 = types[v]
                            vp1, vp2 = types[v_prime]
                            util_v = v1 * q[v, 0] + v2 * q[v, 1] - payments[v]
                            util_rep = v1 * q[v_prime, 0] + v2 * q[v_prime, 1] - payments[v_prime]
                            if util_v < util_rep - 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        revenue = np.sum(probs * payments)
                        if revenue > best_revenue:
                            best_revenue = revenue
                            best_alloc = q
                            best_pay = payments
        except Exception:
            continue
    
    if best_alloc is not None:
        return {'allocations': best_alloc, 'payments': best_pay}
    else:
        # Fallback: Zero mechanism
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}

# EVOLVE-BLOCK-END
