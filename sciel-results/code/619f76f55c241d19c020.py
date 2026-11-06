import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
       deadline = time.time() + 55
       
       # 1. Generate triples
       triples = []
       # x < y
       for x in range(1, 181):
           for y in range(x+1, 181):
               max_k = 180 // (y * (x + y))
               for k in range(1, max_k + 1):
                   a = k * x * y
                   b = k * x * (x + y)
                   c = k * y * (x + y)
                   if a <= 180 and b <= 180 and c <= 180:
                       # Check distinctness
                       if len({a, b, c}) == 3:
                           triples.append((a, b, c))
       
       # 2. Identify V
       V = set()
       for t in triples:
           V.update(t)
       
       # 3. Elements not in V are always included
       safe_elements = [i for i in range(1, 181) if i not in V]
       
       # 4. Solve MIS on V
       if not V:
           return safe_elements
       
       v_list = sorted(V)
       v_index = {v: i for i, v in enumerate(v_list)}
       n_vars = len(v_list)
       
       # Objective: maximize sum x_i
       c_obj = -np.ones(n_vars) # Minimize -sum x_i
       
       # Constraints: sum x_i <= 2 for each triple
       A = []
       for t in triples:
           row = np.zeros(n_vars)
           for v in t:
               row[v_index[v]] = 1
           A.append(row)
       
       if A:
           A = np.array(A)
           b_ub = np.full(len(A), 2)
           b_lb = np.full(len(A), 0) # or -inf, but sum is non-negative
           constraints = [LinearConstraint(A, b_lb, b_ub)]
       else:
           constraints = []
       
       bounds = Bounds(0, 1)
       
       try:
           res = milp(c=c_obj, constraints=constraints, integrality=np.ones(n_vars), bounds=bounds)
           if res.status == 0:
               # res.x contains the solution
               x_sol = res.x
               selected_v = [v_list[i] for i in range(n_vars) if x_sol[i] > 0.5]
               result = safe_elements + selected_v
               return result
           else:
               # Fallback to greedy
               pass
       except Exception:
               pass
       
       # Fallback: Greedy Removal
       current_set = set(range(1, 181))
       # Sort triples by... ?
       # Just remove one element from each triple greedily
       # Better: Local search
       
       # Simple Greedy:
       # Start with all, remove worst offender
       # This is what the seed did, but we can do better with the MILP result if it works.
       
       # If MILP failed, use a heuristic
       # Heuristic:
       # 1. Start with safe_elements
       # 2. For elements in V, try to add them one by one if valid
       #    Or use the conflict graph to pick a large independent set
       
       # Let's implement a simple Greedy Independent Set on the hypergraph V
       # Sort elements in V by degree (number of triples they appear in)
       degrees = {v: 0 for v in V}
       for t in triples:
           for v in t:
               degrees[v] += 1
       
       # Greedy: Pick element with lowest degree? Or highest?
       # For MIS, usually picking low degree first helps keep more options?
       # Or high degree to resolve conflicts?
       # Let's try: Add elements to set. If conflict, remove one.
       
       # Actually, if MILP fails, it's likely due to timeout or error.
       # Given small N, it shouldn't fail.
       
       # Fallback logic:
       best_set = set(safe_elements)
       # Try to add elements from V
       # Sort V by frequency (descending)
       v_sorted = sorted(V, key=lambda v: degrees[v], reverse=True)
       
       for v in v_sorted:
           # Try to add v
           # Check if adding v creates a conflict
           # Conflict if there exist b, c in best_set such that (v, b, c) is a triple
           # Or (b, c, v) etc.
           # We can just check all triples involving v
           can_add = True
           for t in triples:
               if v in t:
                   others = [x for x in t if x != v]
                   if all(o in best_set for o in others):
                       can_add = False
                       break
           if can_add:
               best_set.add(v)
       
       return list(best_set)
