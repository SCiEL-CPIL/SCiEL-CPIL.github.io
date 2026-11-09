import numpy as np
import time
import sys

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n_customers = len(demands)
    n_facilities = len(capabilities) if 'capabilities' in dir() else len(capacities)
    
    # Fallback greedy function in case MILP fails or is too slow
    def greedy_assign():
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        # Sort customers by demand descending
        order = np.argsort(-demands)
        for i in order:
            # Sort facilities by cost for this customer
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
            else:
                # If no feasible facility found (should not happen if total cap > total demand),
                # assign to facility with most remaining capacity (even if it violates cap? 
                # No, task says invalid if capacity exceeded. 
                # But greedy largest-first usually works if instance is feasible.
                # If not, we might return an invalid solution or try to fix.
                # For now, assume feasible.
                pass
        return result

    # Try MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Variables: x_{i,j} for all i,j (100*20 = 2000 vars)
        # y_j for all j (20 vars)
        # Total vars: 2020
        # Order: [x_{0,0}, ..., x_{0,19}, x_{1,0}, ..., x_{99,19}, y_0, ..., y_19]
        
        c = np.zeros(n_customers * n_facilities + n_facilities)
        
        # Objective coefficients for x_{i,j}
        c[:n_customers * n_facilities] = assign_cost.flatten()
        
        # Objective coefficients for y_j
        c[n_customers * n_facilities:] = open_cost
        
        # Constraints:
        # 1. Sum_j x_{i,j} = 1 for each i
        # 2. Sum_i demand_i x_{i,j} <= capacity_j for each j
        # 3. x_{i,j} - y_j <= 0 for each i,j
        
        A_eq = np.zeros((n_customers, n_customers * n_facilities + n_facilities))
        for i in range(n_customers):
            for j in range(n_facilities):
                A_eq[i, i * n_facilities + j] = 1.0
        
        b_eq = np.ones(n_customers)
        
        A_ub = np.zeros((n_facilities + n_customers * n_facilities, n_customers * n_facilities + n_facilities))
        b_ub = np.zeros(n_facilities + n_customers * n_facilities)
        
        # Capacity constraints: rows 0 to n_facilities-1
        for j in range(n_facilities):
            for i in range(n_customers):
                A_ub[j, i * n_facilities + j] = demands[i]
            b_ub[j] = capacities[j]
        
        # Linking constraints: x_{i,j} - y_j <= 0
        # Rows n_facilities to n_facilities + n_customers*n_facilities - 1
        row_offset = n_facilities
        for i in range(n_customers):
            for j in range(n_facilities):
                A_ub[row_offset, i * n_facilities + j] = 1.0
                A_ub[row_offset, n_customers * n_facilities + j] = -1.0
                b_ub[row_offset] = 0.0
                row_offset += 1
        
        # Bounds: x_{i,j} in [0,1], y_j in [0,1]
        # milp expects bounds as (lb, ub) pairs or LinearConstraint
        # scipy.optimize.milp uses bounds as (lower, upper) for each variable
        bounds = Bounds(lb=np.zeros(n_customers * n_facilities + n_facilities), 
                        ub=np.ones(n_customers * n_facilities + n_facilities))
        
        # Integrality: all variables are integer
        integrality = np.ones(n_customers * n_facilities + n_facilities)
        
        # Solve
        res = milp(c=c, 
                   constraints=LinearConstraint(A_ub, -np.inf, b_ub), 
                   integrality=integrality, 
                   bounds=bounds)
        
        # Check if solution is valid
        if res.status == 0: # Optimal
            x_flat = res.x[:n_customers * n_facilities]
            x = x_flat.reshape(n_customers, n_facilities)
            
            # Ensure x is close to 0 or 1
            x = (x > 0.5).astype(int)
            
            # Extract assignment
            result = np.argmax(x, axis=1)
            
            # Verify capacity constraints just in case of numerical issues
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[result[i]] += demands[i]
            
            if np.all(load <= capacities + 1e-6):
                return result
            else:
                # Fallback if numerical issues caused capacity violation
                return greedy_assign()
        else:
            # Failed to solve or infeasible
            # Try to get a feasible solution from the solver's best effort?
            # milp might return a feasible solution even if not optimal? 
            # status 1: infeasible, 2: unbounded, 3: numerical error, 4: no solution, 5: numerical error
            # If res.x is available and feasible, use it.
            if res.x is not None:
                x_flat = res.x[:n_customers * n_facilities]
                x = x_flat.reshape(n_customers, n_facilities)
                # Round to nearest integer
                x = np.round(x).astype(int)
                # Ensure sum is 1?
                # This is risky. Better to fallback to greedy.
                return greedy_assign()
            else:
                return greedy_assign()
                
    except Exception as e:
        # If any error occurs, fallback to greedy
        return greedy_assign()
        
    # If time is running out, we should have already returned.
    # This is a safety net.
    if time.time() > deadline:
        return greedy_assign()

    # Default return if something goes wrong unexpectedly
    return greedy_assign()
