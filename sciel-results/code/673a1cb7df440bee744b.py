   def dpll(clauses, depth):
       if depth > 10000:
           return False
       
       # Save trail length
       start_trail_len = len(trail)
       
       # 1. Unit Propagation
       clauses, conflict, num_assigned = unit_propagate(clauses)
       if conflict:
           untrail(len(trail) - start_trail_len)
           return False
           
       # 2. Pure Literal Elimination
       clauses, num_pure = pure_literal_elimination(clauses)
       
       # 3. Check if all satisfied
       if not clauses:
           return True
           
       # 4. Choose Variable
       freq = [0] * n
       for c in clauses:
           for l in c:
               freq[abs(l)-1] += 1
       var = max(range(n), key=lambda i: freq[i])
       
       # 5. Branch
       # Try True
       assign[var] = True
       trail.append(var)
       
       if dpll(clauses, depth + 1):
           return True
           
       untrail(1)
       
       # Try False
       assign[var] = False
       trail.append(var)
       
       if dpll(clauses, depth + 1):
           return True
           
       untrail(1)
       
       # Restore trail to start state
       untrail(len(trail) - start_trail_len)
       
       return False
   