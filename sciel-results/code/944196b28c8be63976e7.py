import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    def solve_sat_dpll(n, timeout=10.0):
        """
        Solves the SAT problem for R(4,5;3) > n.
        Variables: x_{a,b,c} for 0 <= a < b < c < n.
        x=0 means Red, x=1 means Blue.
        Constraints:
        1. No Red K4: For any 4 vertices, not all 4 triples are Red (0).
           Clause: (x_{a,b,c} OR x_{a,b,d} OR x_{a,c,d} OR x_{b,c,d})
           In 1-based SAT indices: If x is 0, it's False. So clause is (v1 OR v2 OR v3 OR v4).
           Wait, standard SAT: True=1, False=0.
           If we want NOT (all 0), we need (x1=1 OR x2=1 OR x3=1 OR x4=1).
           So clause contains positive literals of the triple variables.
        2. No Blue K5: For any 5 vertices, not all 10 triples are Blue (1).
           Clause: (NOT x_{t1} OR ... OR NOT x_{t10})
           Clause contains negative literals.
        """
        triples = list(combinations(range(n), 3))
        num_vars = len(triples)
        triple_to_idx = {t: i + 1 for i, t in enumerate(triples)} # 1-based for SAT
        
        clauses = []
        
        # Generate Red K4 clauses (4-clauses, positive)
        for quad in combinations(range(n), 4):
            lits = []
            for triple in combinations(quad, 3):
                lits.append(triple_to_idx[triple])
            clauses.append(tuple(lits))
            
        # Generate Blue K5 clauses (10-clauses, negative)
        for quint in combinations(range(n), 5):
            lits = []
            for triple in combinations(quint, 3):
                lits.append(-triple_to_idx[triple])
            clauses.append(tuple(lits))
        
        # Pre-compute variable to clauses index for unit propagation
        var_to_clauses = [[] for _ in range(num_vars + 1)]
        for i, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit)
                var_to_clauses[var].append(i)
                
        # SAT Solver State
        assignment = [0] * (num_vars + 1) # 0: unassigned, 1: True, -1: False
        clause_status = [0] * len(clauses) # 0: active, 1: satisfied, -1: conflict
        watch_lists = [[] for _ in range(num_vars + 1)] # Simplified: just keep active clauses
        
        # Actually, for speed, a simple DPLL with unit propagation and pure literal is better than full CDCL for this size if implemented carefully in Python, but CDCL is more robust.
        # Given the time limit, a pure DPLL with good heuristics might be faster to implement correctly without bugs.
        # Let's use a recursive DPLL with:
        # 1. Unit Propagation
        # 2. Pure Literal Heuristic
        # 3. Variable Selection: Smallest Set of Unassigned Variables (SUS) or simply first unassigned.
        
        # To avoid recursion depth issues and overhead, we'll implement a stack-based search or limit depth.
        # However, R(4,5;3) is a known "hard" instance. A pure Python DPLL might be too slow for n=34.
        
        # Alternative Strategy:
        # Since we know n=34 is possible, we can use a **Hybrid Approach**:
        # 1. Use a fast local search (WalkSAT style) to find a *near* solution or a solution for smaller n.
        # 2. Use the SAT solver to *verify* or *complete* the solution.
        # 3. Actually, the most reliable "different strategy" that works in Python within 120s for this specific known problem is often **Randomized Local Search with Restart** (WalkSAT) which is much faster to implement robustly than a full CDCL in Python.
        # BUT the prompt asks for a MATERIAL change. The current is SA. WalkSAT is similar.
        # Let's try **Genetic Algorithm** or **Simulated Annealing with Multi-start and Intensive Local Search**.
        
        # Wait, looking at the inspirations, p65 tried `solve_sat_dpll`. It scored 1.0.
        # The current program is SA.
        # Let's implement a **WalkSAT** variant which is often superior to SA for SAT-like problems, combined with a **Clause Learning** lite version (remembering conflicts).
        
        # Actually, let's look at the structure.
        # Red K4: 4 vars.
        # Blue K5: 10 vars.
        # 
        # A very effective strategy for these hypergraph Ramsey problems is **Min-Conflict Local Search**.
        # Objective: Minimize the number of violated constraints.
        # Move: Flip a variable involved in a violated clause.
        # Heuristic: Pick the variable that reduces conflicts the most.
        
        # This is structurally different from SA (which accepts worse states probabilistically) because Min-Conflict is a greedy hill-climbing with backtracking/restart, focusing only on conflicting variables.
        
        pass # Placeholder for the actual implementation below

    # We will implement Min-Conflict Local Search (MCL) with Random Restarts.
    # This is a different optimization family: Greedy Stochastic Local Search vs Simulated Annealing.
    
    def run_mcl(n, time_limit):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Precompute constraints
        red_k4s = []
        for quad in combinations(range(n), 4):
            red_k4s.append(tuple(triple_index[t] for t in combinations(quad, 3)))
            
        blue_k5s = []
        for quint in combinations(range(n), 5):
            blue_k5s.append(tuple(triple_index[t] for t in combinations(quint, 3)))
            
        num_red = len(red_k4s)
        num_blue = len(blue_k5s)
        
        # Map triple -> constraints it participates in
        t_to_red = [[] for _ in range(num_triples)]
        t_to_blue = [[] for _ in range(num_triples)]
        for i, k4 in enumerate(red_k4s):
            for t in k4:
                t_to_red[t].append(i)
        for i, k5 in enumerate(blue_k5s):
            for t in k5:
                t_to_blue[t].append(i)

        def get_conflicts(colors):
            """
            Returns list of (constraint_index, type) that are violated.
            Also returns count of conflicts per variable.
            """
            red_conflicts = []
            blue_conflicts = []
            var_conf_count = [0] * num_triples
            
            # Check Red K4s: Violation if all 4 are 0
            for i, k4 in enumerate(red_k4s):
                is_violated = True
                for t in k4:
                    if colors[t] != 0:
                        is_violated = False
                        break
                if is_violated:
                    red_conflicts.append(i)
                    for t in k4:
                        var_conf_count[t] += 1
                        
            # Check Blue K5s: Violation if all 10 are 1
            for i, k5 in enumerate(blue_k5s):
                is_violated = True
                for t in k5:
                    if colors[t] != 1:
                        is_violated = False
                        break
                if is_violated:
                    blue_conflicts.append(i)
                    for t in k5:
                        var_conf_count[t] += 1
                            
            return red_conflicts, blue_conflicts, var_conf_count

        def total_conflicts(rc, bc):
            return len(rc) + len(bc)

        # Main Loop
        best_n_found = 0
        best_colors = None
        
        # Try to find solution for n
        # We start with n=34 and go down if we fail
        # But MCL is good at finding solutions if they exist.
        
        # Optimization: Use numpy for fast conflict checks?
        # For n=34, num_triples = 5984.
        # Red K4s = C(34,4) = 46376.
        # Blue K5s = C(34,5) = 278256.
        # Checking all constraints every step is O(300k * 10) ~ 3M ops.
        # 120s / 3M ~ 40 steps/sec. Too slow for Python loops.
        
        # We need incremental update.
        
        def run_single_mcl_trial(n, time_limit):
            triples = list(combinations(range(n), 3))
            num_triples = len(triples)
            triple_index = {t: i for i, t in enumerate(triples)}
            
            red_k4s = []
            for quad in combinations(range(n), 4):
                red_k4s.append(tuple(triple_index[t] for t in combinations(quad, 3)))
                
            blue_k5s = []
            for quint in combinations(range(n), 5):
                blue_k5s.append(tuple(triple_index[t] for t in combinations(quint, 3)))
                
            num_red = len(red_k4s)
            num_blue = len(blue_k5s)
            
            t_to_red = [[] for _ in range(num_triples)]
            t_to_blue = [[] for _ in range(num_triples)]
            for i, k4 in enumerate(red_k4s):
                for t in k4:
                    t_to_red[t].append(i)
            for i, k5 in enumerate(blue_k5s):
                for t in k5:
                    t_to_blue[t].append(i)
            
            # State
            colors = np.random.randint(0, 2, size=num_triples)
            
            # Conflict counts per constraint
            # red_conf[i] = number of RED (0) triples in k4 i. Violation if 4.
            # blue_conf[i] = number of BLUE (1) triples in k5 i. Violation if 10.
            red_conf = np.zeros(num_red, dtype=np.int32)
            blue_conf = np.zeros(num_blue, dtype=np.int32)
            
            # Initial conflict calculation
            for i, k4 in enumerate(red_k4s):
                cnt = 0
                for t in k4:
                    if colors[t] == 0:
                        cnt += 1
                red_conf[i] = cnt
                
            for i, k5 in enumerate(blue_k5s):
                cnt = 0
                for t in k5:
                    if colors[t] == 1:
                        cnt += 1
                blue_conf[i] = cnt
                
            total_viol = np.sum(red_conf == 4) + np.sum(blue_conf == 10)
            
            if total_viol == 0:
                return colors.tolist()
            
            # Conflict count per variable
            var_conf = np.zeros(num_triples, dtype=np.int32)
            for i in range(num_red):
                if red_conf[i] == 4:
                    for t in red_k4s[i]:
                        var_conf[t] += 1
            for i in range(num_blue):
                if blue_conf[i] == 10:
                    for t in blue_k5s[i]:
                        var_conf[t] += 1
                        
            start = time.time()
            step = 0
            restarts = 0
            max_restarts = 100
            
            while time.time() - start < time_limit:
                if total_viol == 0:
                    return colors.tolist()
                    
                # Select variable to flip
                # Pick a variable involved in a conflict
                # Weighted random selection based on var_conf
                # If no conflicts (shouldn't happen if total_viol > 0), pick random
                if np.sum(var_conf) == 0:
                    # This implies our conflict tracking is out of sync or we have a bug.
                    # Reset
                    break
                
                # Create a weighted list of indices with var_conf > 0
                # To be fast, we can just pick a random conflicting constraint, then a random variable in it.
                
                # Pick a random violated constraint
                violated_reds = np.where(red_conf == 4)[0]
                violated_blues = np.where(blue_conf == 10)[0]
                
                if len(violated_reds) > 0 and (len(violated_blues) == 0 or random.random() < len(violated_reds) / (len(violated_reds) + len(violated_blues))):
                    c_idx = violated_reds[random.randint(0, len(violated_reds)-1)]
                    t_idx = red_k4s[c_idx][random.randint(0, 3)]
                else:
                    if len(violated_blues) == 0:
                        break
                    c_idx = violated_blues[random.randint(0, len(violated_blues)-1)]
                    t_idx = blue_k5s[c_idx][random.randint(0, 9)]
                    
                # Flip t_idx
                old_val = colors[t_idx]
                new_val = 1 - old_val
                colors[t_idx] = new_val
                
                # Update conflicts
                # For each red_k4 containing t_idx:
                #   If old_val == 0, red_conf decreases by 1
                #   If old_val == 1, red_conf increases by 1
                #   Update var_conf for t_idx and other members if violation status changes
                
                # We need to update var_conf efficiently.
                # Let's just recompute var_conf for the affected constraints.
                
                # Delta for red constraints
                for r_idx in t_to_red[t_idx]:
                    old_rc = red_conf[r_idx]
                    if old_val == 0:
                        new_rc = old_rc - 1
                    else:
                        new_rc = old_rc + 1
                    red_conf[r_idx] = new_rc
                    
                    # Update var_conf for this constraint's variables
                    # If old_rc == 4 and new_rc != 4: violation removed. Decrement var_conf for all 4.
                    # If old_rc != 4 and new_rc == 4: violation added. Increment var_conf for all 4.
                    if old_rc == 4 and new_rc != 4:
                        for t in red_k4s[r_idx]:
                            var_conf[t] -= 1
                    elif old_rc != 4 and new_rc == 4:
                        for t in red_k4s[r_idx]:
                            var_conf[t] += 1
                            
                    # Update total_viol
                    if old_rc == 4:
                        total_viol -= 1
                    if new_rc == 4:
                        total_viol += 1

                # Delta for blue constraints
                for b_idx in t_to_blue[t_idx]:
                    old_bc = blue_conf[b_idx]
                    if old_val == 1:
                        new_bc = old_bc - 1
                    else:
                        new_bc = old_bc + 1
                    blue_conf[b_idx] = new_bc
                    
                    if old_bc == 10 and new_bc != 10:
                        for t in blue_k5s[b_idx]:
                            var_conf[t] -= 1
                    elif old_bc != 10 and new_bc == 10:
                        for t in blue_k5s[b_idx]:
                            var_conf[t] += 1
                            
                    if old_bc == 10:
                        total_viol -= 1
                    if new_bc == 10:
                        total_viol += 1
                        
                step += 1
                if step % 10000 == 0:
                    # Check for stagnation?
                    # If no improvement in last X steps, restart?
                    # MCL usually handles this by just continuing, but restarts help.
                    pass
                    
            return None

        # Strategy: Try n=34. If fails, try 33, etc.
        # But we want to maximize n.
        # We will try n=34 first.
        
        for n in range(34, 0, -1):
            if time.time() > deadline - 10:
                break
                
            # Try a few restarts for this n
            success = False
            for restart in range(5):
                if time.time() > deadline - 10:
                    break
                res = run_single_mcl_trial(n, time_limit=10.0)
                if res is not None:
                    return {'n': n, 'colors': res}
                    
        # Fallback: Return a known small valid coloring
        # n=9 is valid?
        # R(4,5;3) is known to be 35? No, R(4,5;3) = 35 means there is NO coloring for 35, but there IS for 34.
        # So if we fail to find 34, we should return the best we found or a valid small one.
        # The task says "Maximize n".
        # If we can't find 34, we should try to find 33, 32...
        
        # Let's just return the best valid n we found in the loop above.
        # The loop above returns immediately on success.
        # If it finishes without success, we need a fallback.
        
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return {'n': n, 'colors': colors}

    return run_mcl(34, 110)

if __name__ == "__main__":
    res = construct()
    print(res['n'])
