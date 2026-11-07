import numpy as np
import time
import scipy.optimize as opt
from scipy.sparse import csr_matrix

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Fallback greedy function
    def greedy_assign():
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        # Sort customers by demand descending
        order = np.argsort(-demands)
        for i in order:
            # Sort facilities by cost for this customer
            fac_order = np.argsort(assign_cost[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # Should not happen if capacities are sufficient, 
                # but if it does, assign to facility with most remaining space?
                # Or just the first one? The problem implies a feasible solution exists.
                # If no facility has space, the instance is infeasible. 
                # The prompt says "assigns each of 100 customers", implying feasibility.
                # We'll just pick the one with most remaining capacity to be safe, 
                # though it might violate capacity if truly infeasible.
                rem = capacities - load
                f_max = np.argmax(rem)
                result[i] = f_max
                load[f_max] += demands[i]
        return result

    try:
        # Variables:
        # x[i, j] for i in 0..N-1, j in 0..M-1  (N*M vars)
        # y[j] for j in 0..M-1                  (M vars)
        # Total vars: N*M + M
        
        n_vars = n_customers * n_facilities + n_facilities
        
        # Objective coefficients
        c = np.zeros(n_vars)
        # Cost for x[i,j] is assign_cost[i,j]
        c[:n_customers * n_facilities] = assign_cost.flatten()
        # Cost for y[j] is open_cost[j]
        c[n_customers * n_facilities:] = open_cost
        
        # Integrality
        # x are continuous (0), y are integer (1)
        integrality = np.zeros(n_vars, dtype=int)
        integrality[n_customers * n_facilities:] = 1
        
        # Variable bounds
        var_lb = np.zeros(n_vars)
        var_ub = np.full(n_vars, np.inf)
        # y[j] <= 1
        var_ub[n_customers * n_facilities:] = 1.0
        
        # Constraints
        # 1. Demand: sum_j x[i,j] = demands[i]
        # 2. Capacity: sum_i x[i,j] <= capacities[j]
        # 3. Linking: x[i,j] <= capacities[j] * y[j]  =>  x[i,j] - capacities[j]*y[j] <= 0
        
        n_constraints = n_customers + n_facilities + n_customers * n_facilities
        
        rows = []
        cols = []
        vals = []
        lb = np.zeros(n_constraints)
        ub = np.zeros(n_constraints)
        
        # Demand constraints (first N rows)
        for i in range(n_customers):
            for j in range(n_facilities):
                var_idx = i * n_facilities + j
                rows.append(i)
                cols.append(var_idx)
                vals.append(1.0)
            lb[i] = demands[i]
            ub[i] = demands[i]
            
        # Capacity constraints (next M rows)
        for j in range(n_facilities):
            for i in range(n_customers):
                var_idx = i * n_facilities + j
                row_idx = n_customers + j
                rows.append(row_idx)
                cols.append(var_idx)
                vals.append(1.0)
            lb[n_customers + j] = 0.0
            ub[n_customers + j] = capacities[j]
            
        # Linking constraints (next N*M rows)
        for i in range(n_customers):
            for j in range(n_facilities):
                var_idx_x = i * n_facilities + j
                var_idx_y = n_customers * n_facilities + j
                row_idx = n_customers + n_facilities + i * n_facilities + j
                
                rows.append(row_idx)
                cols.append(var_idx_x)
                vals.append(1.0)
                
                rows.append(row_idx)
                cols.append(var_idx_y)
                vals.append(-capacities[j])
                
                lb[row_idx] = 0.0
                ub[row_idx] = 0.0
                
        A = csr_matrix((vals, (rows, cols)), shape=(n_constraints, n_vars))
        
        # Solve MILP
        res = opt.milp(
            c=c,
            constraints=A,
            integrality=integrality,
            bounds=list(zip(var_lb, var_ub)),
            options={"time_limit": 10}  # Hard limit inside solver
        )
        
        if res.status == 0 and res.x is not None:
            x_vals = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
            # Find the facility with max flow for each customer
            # Since sum x[i,j] = demand[i], and x[i,j] >= 0,
            # there should be one dominant facility, or split if fractional (but x is continuous in MILP? 
            # Wait, x is continuous in standard CFLP formulation. 
            # If x is split, e.g., 5 units to fac 1 and 5 to fac 2, 
            # the assignment cost is linear, so splitting doesn't change cost vs sending all to one? 
            # No, assign_cost is linear per unit. 
            # However, the return type must be "one facility index per customer".
            # If the optimal solution splits demand, we must choose one facility.
            # Since the cost is linear, any split is optimal for the variable cost part.
            # But we need to return a single index.
            # We can just pick the facility with the largest x[i,j].
            # Is it possible that splitting allows opening fewer facilities?
            # No, opening cost is fixed per facility. If you send >0 to a facility, you pay open cost.
            # The linking constraint x[i,j] <= cap[j] y[j] ensures if x[i,j] > 0, y[j] must be 1 (since y is binary).
            # So the MILP solution will have y[j]=1 for any facility receiving flow.
            # If a customer's demand is split, it means multiple facilities are open.
            # We must assign the customer to ONE facility.
            # Which one?
            # The problem statement: "assigns each of 100 customers to one of 20 facilities".
            # This implies an integer assignment x[i,j] is either demand[i] or 0?
            # "Return an integer NumPy array ... with one facility index per customer."
            # If I return index k, it implies the entire demand of i goes to k.
            # Does the scoring assume this?
            # "sum of assign_cost[customer, assigned facility]".
            # Yes, the score is calculated based on the returned indices.
            # If the true optimal solution splits demand, but I must return a single index,
            # I should return the index that minimizes the cost given the split?
            # Actually, if I return index k, the score uses assign_cost[i, k] * demand[i].
            # If the MILP solution had x[i,k] = 5 and x[i,l] = 5 (demand 10),
            # and I return k, the score will be 10 * cost[i,k].
            # But the MILP cost was 5*cost[i,k] + 5*cost[i,l].
            # If cost[i,k] != cost[i,l], the score will be different.
            # However, for the specific problem instance, if splitting is optimal,
            # it's usually because of capacity constraints.
            # But here, customers are assigned to ONE facility.
            # This is the "Uncapacitated" or "Single Assignment" variant?
            # No, it's "Capacitated", but the interface forces single assignment.
            # This means the problem is actually:
            # Minimize sum open_cost + sum assign_cost[i, k(i)]
            # Subject to: sum_{i: k(i)=j} demand[i] <= capacity[j]
            # This is an Integer Program where x[i,j] is binary (1 if assigned, 0 otherwise) * demand[i].
            # My MILP formulation allowed splitting (x[i,j] continuous).
            # If I allow splitting, I might find a solution that is "cheaper" in the LP sense but infeasible for the "single assignment" constraint of the output interface?
            # No, if I split, I can't represent it in the output.
            # I must pick one.
            # If I pick the one with largest flow, I might incur higher cost than the split solution.
            # Is it better to force integer assignment in the MILP?
            # Yes. Let x[i,j] be 0 or demand[i].
            # This makes it a pure Integer Program.
            # Variables: z[i,j] in {0, 1}.
            # x[i,j] = z[i,j] * demand[i].
            # Objective: sum open_cost[j] y[j] + sum assign_cost[i,j] * demand[i] * z[i,j].
            # Constraints:
            # sum_j z[i,j] = 1 for all i.
            # sum_i demand[i] * z[i,j] <= capacity[j] for all j.
            # z[i,j] <= y[j].
            
            # This is a smaller and more precise model for the required output format.
            # Number of binary vars: 100 * 20 = 2000.
            # This is still very small for HiGHS.
            
            # Let's re-evaluate. Should I use this integer formulation?
            # Yes, it aligns perfectly with the output requirement.
            
            # I will restart the construction with this formulation.
            pass
        else:
            # Fallback if MILP failed
            return greedy_assign()
            
    except Exception as e:
        return greedy_assign()

    # Re-implement with Integer Formulation
    # Variables: z[i,j] binary, y[j] binary.
    # Total vars: 2000 + 20 = 2020.
    
    try:
        n_vars = n_customers * n_facilities + n_facilities
        c = np.zeros(n_vars)
        
        # Objective: sum open_cost[j] y[j] + sum assign_cost[i,j] * demand[i] * z[i,j]
        for i in range(n_customers):
            for j in range(n_facilities):
                c[i * n_facilities + j] = assign_cost[i, j] * demands[i]
        for j in range(n_facilities):
            c[n_customers * n_facilities + j] = open_cost[j]
            
        integrality = np.ones(n_vars, dtype=int) # All binary
        
        var_lb = np.zeros(n_vars)
        var_ub = np.ones(n_vars)
        
        # Constraints
        # 1. Assignment: sum_j z[i,j] = 1
        # 2. Capacity: sum_i demand[i] * z[i,j] <= capacity[j]
        # 3. Linking: z[i,j] <= y[j]
        
        n_constraints = n_customers + n_facilities + n_customers * n_facilities
        rows = []
        cols = []
        vals = []
        lb = np.zeros(n_constraints)
        ub = np.zeros(n_constraints)
        
        # 1. Assignment
        for i in range(n_customers):
            for j in range(n_facilities):
                rows.append(i)
                cols.append(i * n_facilities + j)
                vals.append(1.0)
            lb[i] = 1.0
            ub[i] = 1.0
            
        # 2. Capacity
        for j in range(n_facilities):
            for i in range(n_customers):
                rows.append(n_customers + j)
                cols.append(i * n_facilities + j)
                vals.append(demands[i])
            lb[n_customers + j] = 0.0
            ub[n_customers + j] = capacities[j]
            
        # 3. Linking: z[i,j] - y[j] <= 0
        for i in range(n_customers):
            for j in range(n_facilities):
                row_idx = n_customers + n_facilities + i * n_facilities + j
                rows.append(row_idx)
                cols.append(i * n_facilities + j)
                vals.append(1.0)
                
                rows.append(row_idx)
                cols.append(n_customers * n_facilities + j)
                vals.append(-1.0)
                
                lb[row_idx] = 0.0
                ub[row_idx] = 0.0
                
        A = csr_matrix((vals, (rows, cols)), shape=(n_constraints, n_vars))
        
        res = opt.milp(
            c=c,
            constraints=A,
            integrality=integrality,
            bounds=list(zip(var_lb, var_ub)),
            options={"time_limit": 10}
        )
        
        if res.status == 0 and res.x is not None:
            z = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
            # Round to nearest integer (should be 0 or 1)
            z = np.round(z).astype(int)
            # Find argmax for each customer
            result = np.argmax(z, axis=1)
            
            # Validate capacity
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[result[i]] += demands[i]
            if np.all(load <= capacities):
                return result
            else:
                # Fallback if rounding caused issues (unlikely)
                return greedy_assign()
        else:
            return greedy_assign()
            
    except Exception:
        return greedy_assign()
