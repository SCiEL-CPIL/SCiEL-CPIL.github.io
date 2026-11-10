import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290

    # Preprocessing
    clause_vars = []
    var_occurrences = [[] for _ in range(n)]
    
    for ci, clause in enumerate(clauses):
        lits = []
        for l in clause:
            vi = abs(l) - 1
            sign = 1 if l > 0 else -1
            lits.append((vi, sign))
            var_occurrences[vi].append((ci, sign))
        clause_vars.append(lits)

    def compute_initial_state(assignment):
        counts = [0] * m
        for ci in range(m):
            cnt = 0
            for vi, sign in clause_vars[ci]:
                if assignment[vi] == (sign == 1):
                    cnt += 1
            counts[ci] = cnt
        unsat = sum(1 for c in counts if c == 0)
        return counts, unsat

    # Spectral initialization using literal co-occurrence
    # Build a matrix M where M[i,j] = 1 if literals i and j appear in the same clause
    # Then find the principal eigenvector. If component > 0, literal is likely true.
    # For speed, we can use a simple power iteration on a smaller representation or
    # just use frequency if n is large. But n=150 is small enough.
    
    # Actually, building a 150x150 matrix and doing eigenvector computation is fine.
    # Let's use numpy for this if available, or a simple power method.
    try:
        import numpy as np
        # Build co-occurrence matrix for literals
        # There are 2*n literals. Let's index them as (var, sign) -> index
        # Or simpler: just count how often each literal appears in clauses where 
        # the other literals also appear.
        # Actually, the spectral method for SAT typically looks at the matrix of 
        # literal correlations.
        
        # Let's try a simpler heuristic first: count literal frequencies.
        # But the prompt specifically suggests spectral method.
        # Let's implement a quick power iteration on the co-occurrence matrix.
        
        # Matrix size: 2n x 2n
        # lit_index: (var, sign) -> var*2 + (0 if sign==1 else 1)
        # M[i,j] = 1 if lit i and lit j are in the same clause
        
        # This is O(m * 3^2) to build, then O(k * (2n)^2) for power iteration.
        # 2n = 300. 300^2 = 90000. k=10 iterations -> 900k ops. Fast.
        
        num_lits = 2 * n
        M = [[0.0] * num_lits for _ in range(num_lits)]
        
        for clause in clause_vars:
            # clause has 3 literals
            for i in range(3):
                vi, si = clause[i]
                li = vi * 2 + (0 if si == 1 else 1)
                for j in range(3):
                    if i != j:
                        vj, sj = clause[j]
                        lj = vj * 2 + (0 if sj == 1 else 1)
                        M[li][lj] += 1.0
        
        # Power iteration
        v = [1.0] * num_lits
        for _ in range(10):
            new_v = [0.0] * num_lits
            for i in range(num_lits):
                s = 0.0
                row = M[i]
                for j in range(num_lits):
                    s += row[j] * v[j]
                new_v[i] = s
            # Normalize
            norm = math.sqrt(sum(x*x for x in new_v))
            if norm > 0:
                new_v = [x/norm for x in new_v]
            v = new_v
        
        # Assign based on sign of eigenvector components
        spectral_assignment = [False] * n
        for vi in range(n):
            # Component for (vi, True) is at vi*2
            # Component for (vi, False) is at vi*2+1
            val_true = v[vi*2]
            val_false = v[vi*2+1]
            spectral_assignment[vi] = val_true > val_false
            
    except ImportError:
        spectral_assignment = [False] * n

    best_unsat = m + 1
    best_assignment = None

    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 12345 + 6789)
        
        # Use spectral assignment for the first few restarts, then random
        if restart < 5:
            assignment = spectral_assignment[:]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
        counts, unsat_count = compute_initial_state(assignment)
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        # SA Parameters
        T0 = 5.0
        T = T0
        cooling_rate = 0.999
        
        start_time = time.time()
        steps_in_restart = 0
        max_steps_per_restart = 200000
        
        while time.time() < deadline - 5 and steps_in_restart < max_steps_per_restart:
            if unsat_count == 0:
                return assignment
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
                
            # Pick random variable
            vi = rng.randrange(n)
            
            # Calculate delta
            delta = 0
            for ci, sign in var_occurrences[vi]:
                was_sat = assignment[vi] == (sign == 1)
                if was_sat:
                    if counts[ci] == 1:
                        delta += 1
                else:
                    if counts[ci] == 0:
                        delta -= 1
            
            # Acceptance
            if delta < 0 or (T > 1e-6 and rng.random() < math.exp(-delta / T)):
                # Accept move
                for ci, sign in var_occurrences[vi]:
                    was_sat = assignment[vi] == (sign == 1)
                    if was_sat:
                        counts[ci] -= 1
                    else:
                        counts[ci] += 1
                assignment[vi] = not assignment[vi]
                unsat_count += delta
                
            # Cool
            T *= cooling_rate
            steps_in_restart += 1
            
    if best_assignment is not None:
        return best_assignment
    return [False] * n
