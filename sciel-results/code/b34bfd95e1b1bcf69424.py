import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 5.0

    def get_elapsed():
        return time.time() - start_time

    def is_deadline_exceeded():
        return get_elapsed() > time_limit - safety_margin

    def greedy_assign(fac_indices):
        """
        Greedily assigns customers to facilities in fac_indices.
        Returns (assignment_array, total_cost) or (None, inf) if infeasible.
        """
        if not fac_indices:
            return None, np.inf
            
        m = len(fac_indices)
        
        remaining_cap = [capacities[j] for j in fac_indices]
        assignment = np.full(n_cust, -1, dtype=int)
        
        # Sort customers by demand descending
        cust_order = np.argsort(-demands)
        
        # Precompute costs for these facilities
        sub_costs = np.zeros((n_cust, m))
        for j_local, j_global in enumerate(fac_indices):
            sub_costs[:, j_local] = assign_cost[:, j_global]
            
        for i in cust_order:
            best_j = -1
            best_cost = np.inf
            for j_local in range(m):
                if remaining_cap[j_local] >= demands[i]:
                    if sub_costs[i, j_local] < best_cost:
                        best_cost = sub_costs[i, j_local]
                        best_j = j_local
                        
            if best_j == -1:
                return None, np.inf
                
            assignment[i] = best_j
            remaining_cap[best_j] -= demands[i]
            
        total_assign_cost = 0
        used_facilities = set(assignment)
        total_open_cost = 0
        for j_local in used_facilities:
            total_open_cost += open_cost[fac_indices[j_local]]
            
        for i in range(n_cust):
            total_assign_cost += sub_costs[i, assignment[i]]
            
        return assignment, total_assign_cost + total_open_cost

    # 1. Initial Solution: Greedy Set Construction
    S = []
    best_S = []
    best_cost = np.inf
    best_assignment = None
    
    # Start with the facility that has the largest capacity
    init_fac = np.argmax(capacities)
    S = [init_fac]
    
    current_assignment, current_cost = greedy_assign(S)
    
    if current_assignment is None:
        remaining_facs = [k for k in range(n_fac) if k not in S]
        remaining_facs.sort(key=lambda x: open_cost[x])
        
        for k in remaining_facs:
            if is_deadline_exceeded():
                break
            S.append(k)
            current_assignment, current_cost = greedy_assign(S)
            if current_assignment is not None:
                break
    else:
        pass
        
    if current_assignment is not None:
        best_S = S.copy()
        best_cost = current_cost
        best_assignment = current_assignment
    else:
        S = list(range(n_fac))
        current_assignment, current_cost = greedy_assign(S)
        if current_assignment is not None:
            best_S = S.copy()
            best_cost = current_cost
            best_assignment = current_assignment
        else:
            return np.zeros(n_cust, dtype=int)

    # 2. Local Search on Set S
    S = best_S.copy()
    current_assignment = best_assignment
    current_cost = best_cost
    
    improved = True
    while improved and not is_deadline_exceeded():
        improved = False
        
        for idx, k in enumerate(list(S)):
            if len(S) <= 1:
                continue
            new_S = S[:idx] + S[idx+1:]
            new_assignment, new_cost = greedy_assign(new_S)
            if new_assignment is not None and new_cost < current_cost:
                S = new_S
                current_assignment = new_assignment
                current_cost = new_cost
                improved = True
                break
                
        if improved:
            continue
            
        for k in range(n_fac):
            if k in S:
                continue
            new_S = S + [k]
            new_assignment, new_cost = greedy_assign(new_S)
            if new_assignment is not None and new_cost < current_cost:
                S = new_S
                current_assignment = new_assignment
                current_cost = new_cost
                improved = True
                break

    # 3. Polish Assignment for final S
    final_fac_indices = S
    m = len(final_fac_indices)
    
    load = np.zeros(m, dtype=int)
    for i in range(n_cust):
        load[current_assignment[i]] += demands[i]
        
    polish_improved = True
    while polish_improved and not is_deadline_exceeded():
        polish_improved = False
        
        for i in range(n_cust):
            curr_j = current_assignment[i]
            for new_j in range(m):
                if new_j == curr_j:
                    continue
                if load[new_j] + demands[i] <= capacities[final_fac_indices[new_j]]:
                    old_cost = assign_cost[i, final_fac_indices[curr_j]]
                    new_cost = assign_cost[i, final_fac_indices[new_j]]
                    
                    old_load_curr = load[curr_j]
                    old_load_new = load[new_j]
                    
                    open_cost_change = 0
                    if old_load_curr == demands[i]:
                        open_cost_change -= open_cost[final_fac_indices[curr_j]]
                    if old_load_new == 0:
                        open_cost_change += open_cost[final_fac_indices[new_j]]
                        
                    total_change = (new_cost - old_cost) + open_cost_change
                    
                    if total_change < 0:
                        load[curr_j] -= demands[i]
                        load[new_j] += demands[i]
                        current_assignment[i] = new_j
                        current_cost += total_change
                        polish_improved = True
                        break
                        
        if polish_improved:
            continue
            
        for i in range(n_cust):
            for l in range(i+1, n_cust):
                j_i = current_assignment[i]
                j_l = current_assignment[l]
                if j_i == j_l:
                    continue
                
                if load[j_i] - demands[i] + demands[l] <= capacities[final_fac_indices[j_i]] and \
                   load[j_l] - demands[l] + demands[i] <= capacities[final_fac_indices[j_l]]:
                   
                    old_cost_i = assign_cost[i, final_fac_indices[j_i]]
                    old_cost_l = assign_cost[l, final_fac_indices[j_l]]
                    new_cost_i = assign_cost[i, final_fac_indices[j_l]]
                    new_cost_l = assign_cost[l, final_fac_indices[j_i]]
                    
                    new_load_j_i = load[j_i] - demands[i] + demands[l]
                    new_load_j_l = load[j_l] - demands[l] + demands[i]
                    
                    open_cost_change = 0
                    if load[j_i] > 0 and new_load_j_i == 0:
                        open_cost_change -= open_cost[final_fac_indices[j_i]]
                    if load[j_l] > 0 and new_load_j_l == 0:
                        open_cost_change -= open_cost[final_fac_indices[j_l]]
                    
                    total_change = (new_cost_i + new_cost_l - old_cost_i - old_cost_l) + open_cost_change
                    
                    if total_change < 0:
                        load[j_i] = new_load_j_i
                        load[j_l] = new_load_j_l
                        current_assignment[i] = j_l
                        current_assignment[l] = j_i
                        current_cost += total_change
                        polish_improved = True
                        break
            if polish_improved:
                break
                
    final_assignment = np.zeros(n_cust, dtype=int)
    for i in range(n_cust):
        final_assignment[i] = final_fac_indices[current_assignment[i]]
        
    return final_assignment
