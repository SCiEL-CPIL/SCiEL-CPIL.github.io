import random
import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 256
    
    # Precompute 3-APs and 4-APs
    ap3s = []
    for d in range(1, n):
        for a in range(0, n - 2 * d):
            ap3s.append((a, a + d, a + 2 * d))
            
    ap4s = []
    for d in range(1, n):
        for a in range(0, n - 3 * d):
            ap4s.append((a, a + d, a + 2 * d, a + 3 * d))
            
    num_3aps = len(ap3s)
    num_4aps = len(ap4s)
    
    # Try MILP first
    try:
        # Variables: x[0..n-1], y[0..num_3aps-1]
        num_vars = n + num_3aps
        c = np.zeros(num_vars)
        c[n:] = 1.0 # Maximize sum of y
        
        # Constraints:
        # 1. y_k <= x_i for i in ap3s[k]  => -x_i + y_k <= 0
        # 2. sum(x_i for i in ap4s[j]) <= 3
        num_rows = 3 * num_3aps + num_4aps
        A_ub = np.zeros((num_rows, num_vars))
        b_ub = np.zeros(num_rows)
        
        row_idx = 0
        for k, ap in enumerate(ap3s):
            y_idx = n + k
            for i in ap:
                A_ub[row_idx, i] = -1.0
                A_ub[row_idx, y_idx] = 1.0
                b_ub[row_idx] = 0.0
                row_idx += 1
                
        for ap in ap4s:
            for i in ap:
                A_ub[row_idx, i] = 1.0
            b_ub[row_idx] = 3.0
            row_idx += 1
            
        lb = np.zeros(num_vars)
        ub = np.ones(num_vars)
        integrality = np.ones(num_vars)
        
        start_time = time.time()
        res = milp(
            c=c,
            constraints=LinearConstraint(A_ub, -np.inf, b_ub),
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if res.status == 0 and res.x is not None:
            x_sol = res.x[:n].astype(int)
            # Verify validity
            valid = True
            for ap in ap4s:
                if all(x_sol[i] == 1 for i in ap):
                    valid = False
                    break
            if valid:
                result = [i + 1 for i in range(n) if x_sol[i] == 1]
                if len(result) > 0:
                    return result
    except Exception:
        pass
        
    # Fallback: Simulated Annealing
    # Precompute indices for SA
    index_to_ap3 = [[] for _ in range(n)]
    for i, ap in enumerate(ap3s):
        for idx in ap:
            index_to_ap3[idx].append(i)
            
    index_to_ap4 = [[] for _ in range(n)]
    for i, ap in enumerate(ap4s):
        for idx in ap:
            index_to_ap4[idx].append(i)

    state = [0] * n
    for i in range(n):
        if random.random() < 0.5:
            state[i] = 1
            
    def repair(state):
        changed = True
        while changed:
            changed = False
            for ap in ap4s:
                if all(state[i] == 1 for i in ap):
                    victim = random.choice(ap)
                    state[victim] = 0
                    changed = True
        return state

    state = repair(state)
    
    def count_3aps(state):
        count = 0
        for ap in ap3s:
            if all(state[i] == 1 for i in ap):
                count += 1
        return count
        
    def count_4aps(state):
        count = 0
        for ap in ap4s:
            if all(state[i] == 1 for i in ap):
                count += 1
        return count

    current_3ap_count = count_3aps(state)
    current_4ap_count = count_4aps(state)
    
    T = 100.0
    T_min = 0.01
    alpha = 0.9995
    penalty = 10000
    best_state = state[:]
    best_score = current_3ap_count - penalty * current_4ap_count
    
    start_time = time.time()
    max_time = 50.0
    
    while T > T_min:
        if time.time() - start_time > max_time:
            break
            
        idx = random.randint(0, n - 1)
        old_val = state[idx]
        new_val = 1 - old_val
        
        delta_3 = 0
        for ap_idx in index_to_ap3[idx]:
            ap = ap3s[ap_idx]
            other_present = 0
            for i in ap:
                if i != idx and state[i] == 1:
                    other_present += 1
            if new_val == 1:
                if other_present == 2:
                    delta_3 += 1
            else:
                if other_present == 2:
                    delta_3 -= 1
                    
        delta_4 = 0
        for ap_idx in index_to_ap4[idx]:
            ap = ap4s[ap_idx]
            other_present = 0
            for i in ap:
                if i != idx and state[i] == 1:
                    other_present += 1
            if new_val == 1:
                if other_present == 3:
                    delta_4 += 1
            else:
                if other_present == 3:
                    delta_4 -= 1
        
        delta_score = delta_3 - penalty * delta_4
        
        if delta_score >= 0:
            accept = True
        else:
            accept = random.random() < math.exp(delta_score / T)
            
        if accept:
            state[idx] = new_val
            current_3ap_count += delta_3
            current_4ap_count += delta_4
            current_score = current_3ap_count - penalty * current_4ap_count
            
            if current_score > best_score:
                best_score = current_score
                best_state = state[:]
                
        T *= alpha
        
    if count_4aps(best_state) > 0:
        best_state = repair(best_state)
        
    result = [i + 1 for i in range(n) if best_state[i] == 1]
    
    return result
