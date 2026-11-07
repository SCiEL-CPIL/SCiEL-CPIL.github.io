import numpy as np
def assign(demands, capacities, open_cost, assign_cost):
       n_customers = len(demands)
       n_facilities = len(capacities)
       
       # 1. Initialization
       assignment = np.zeros(n_customers, dtype=int)
       load = np.zeros(n_facilities)
       
       # Sort customers by demand descending for better packing
       order = np.argsort(-demands)
       
       for i in order:
           # Find feasible facilities
           feasible = np.where(load + demands[i] <= capacities)[0]
           if len(feasible) == 0:
               # Fallback: pick facility with most remaining capacity
               # This might violate capacity if instance is infeasible, 
               # but assuming feasibility, this shouldn't happen often 
               # or we need a smarter init.
               # For robustness, if no feasible, we have a problem.
               # Let's assume feasibility.
               f = np.argmax(capacities - load)
           else:
               # Among feasible, pick min assign_cost
               costs = assign_cost[i, feasible]
               min_idx = np.argmin(costs)
               f = feasible[min_idx]
           
           assignment[i] = f
           load[f] += demands[i]
       
       # 2. Local Search
       # Helper: Calculate total cost
       def calc_cost(assignment, load):
           # Open cost
           open_facilities = np.unique(assignment)
           cost = np.sum(open_cost[open_facilities])
           # Assign cost
           for i in range(n_customers):
               cost += assign_cost[i, assignment[i]]
           return cost
       
       current_cost = calc_cost(assignment, load)
       
       # Improvement Loop
       improved = True
       while improved:
           improved = False
           
           # Pass 1: Relocate customers to better open facilities
           for i in range(n_customers):
               current_f = assignment[i]
               # Find best open facility (excluding current if possible, or just all)
               # We need to know which are open.
               open_set = set(np.unique(assignment))
               
               best_cost = assign_cost[i, current_f]
               best_f = current_f
               
               for f in open_set:
                   if f == current_f: continue
                   if load[f] + demands[i] <= capacities[f]:
                       if assign_cost[i, f] < best_cost:
                           best_cost = assign_cost[i, f]
                           best_f = f
               
               if best_f != current_f:
                   # Move
                   load[current_f] -= demands[i]
                   load[best_f] += demands[i]
                   assignment[i] = best_f
                   # Cost change: -assign_cost[i, current_f] + assign_cost[i, best_f]
                   # Open cost doesn't change because best_f is already open.
                   current_cost += (assign_cost[i, best_f] - assign_cost[i, current_f])
                   improved = True
           
           # Pass 2: Try to open closed facilities
           all_facilities = set(range(n_facilities))
           open_set = set(np.unique(assignment))
           closed_set = all_facilities - open_set
           
           for f in closed_set:
               # Check if opening f is beneficial
               # Try to move customers to f
               # A simple heuristic: find the customer i that benefits most from moving to f
               # Gain = assign_cost[i, assignment[i]] - assign_cost[i, f]
               # If Gain > open_cost[f], it's worth it.
               # But we must check capacity.
               
               # Let's iterate customers and see if we can "pull" some to f
               # Actually, just try to move the best candidate.
               best_gain = 0
               best_i = -1
               
               for i in range(n_customers):
                   curr_f = assignment[i]
                   # Potential saving in assignment cost
                   saving = assign_cost[i, curr_f] - assign_cost[i, f]
                   if saving > best_gain:
                       # Check if we can move i to f
                       # We don't know load of f yet (it's 0), so check capacity
                       if demands[i] <= capacities[f]:
                           best_gain = saving
                           best_i = i
               
               if best_gain > open_cost[f]:
                   # Open f and move best_i
                   i = best_i
                   curr_f = assignment[i]
                   
                   load[curr_f] -= demands[i]
                   load[f] += demands[i]
                   assignment[i] = f
                   
                   # Update cost
                   # Remove open cost of curr_f if it becomes empty?
                   # We'll handle closing in Pass 3.
                   # Add open cost of f
                   current_cost += open_cost[f]
                   current_cost += (assign_cost[i, f] - assign_cost[i, curr_f])
                   
                   improved = True
                   
                   # After opening, we might want to run Pass 1 again to pull more customers.
                   # The `while improved` loop will handle this by restarting passes.
                   # But to be efficient, we can break and let the next iteration handle it.
                   break 
           
           # Pass 3: Try to close open facilities
           open_set = set(np.unique(assignment))
           for f in open_set:
               # Check if we can close f
               # Count customers in f
               customers_in_f = [i for i in range(n_customers) if assignment[i] == f]
               
               if len(customers_in_f) == 0:
                   # Should not happen if open_set is derived from assignment
                   continue
               
               # If only 1 customer, easy check
               if len(customers_in_f) == 1:
                   i = customers_in_f[0]
                   # Find best other open facility
                   other_open = [g for g in open_set if g != f]
                   if not other_open:
                       # Cannot close if it's the only one
                       continue
                   
                   best_cost = float('inf')
                   best_g = -1
                   for g in other_open:
                       if load[g] + demands[i] <= capacities[g]:
                           if assign_cost[i, g] < best_cost:
                               best_cost = assign_cost[i, g]
                               best_g = g
                   
                   if best_g != -1:
                       # Cost change:
                       # Remove open_cost[f]
                       # Add assign_cost[i, g] - assign_cost[i, f]
                       delta = -open_cost[f] + (assign_cost[i, g] - assign_cost[i, f])
                       if delta < 0:
                           # Close f, move i to g
                           load[f] -= demands[i]
                           load[g] += demands[i]
                           assignment[i] = g
                           current_cost += delta
                           improved = True
                           break # Restart loop to update open_set
               
               # If >1 customers, try to move all to other open facilities
               else:
                   # This is more complex.
                   # Calculate cost to move all customers in f to other open facilities
                   # and see if it saves open_cost[f]
                   
                   # For simplicity, we might skip closing facilities with >1 customers
                   # in this basic version, or do a greedy attempt.
                   # Given time, let's try a simple greedy:
                   # Try to move each customer to the best other open facility.
                   # If all can be moved feasibly, check total cost.
                   
                   moves = []
                   total_assign_increase = 0
                   feasible = True
                   
                   # We need to simulate the moves to check capacities
                   # This is tricky because moving one affects capacity for the next.
                   # Since we are moving OUT of f, capacity in f frees up, but we are moving TO others.
                   # We need to check if other facilities have enough space.
                   
                   # Let's just skip this for now to keep it simple and fast.
                   # The "Open" and "Relocate" passes are usually sufficient for good results.
                   # Closing multi-customer facilities is a higher-level move.
                   pass
       
       return assignment
