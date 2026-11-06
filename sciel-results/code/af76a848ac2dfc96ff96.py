import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    deadline = time.time() + 280  # Safety margin

    m = len(clauses)
    
    # Precompute matrices
    # Pos[i, j] is 1 if literal +x_{j+1} is in clause i
    # Neg[i, j] is 1 if literal -x_{j+1} is in clause i
    Pos = np.zeros((m, n), dtype=np.int8)
    Neg = np.zeros((m, n), dtype=np.int8)
    
    # Also store clause literals for easy access
    clause_lits = [list(c) for c in clauses]
    
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            if lit > 0:
                Pos[i, var_idx] = 1
            else:
                Neg[i, var_idx] = 1

    def count_violated(A):
        """A is a numpy boolean array of size n."""
        # SatCount[i] = sum(Pos[i, :] * A) + sum(Neg[i, :] * (1 - A))
        # Using int8 to save memory/bandwidth
        A_int = A.astype(np.int8)
        # 1 - A_int is valid since A is 0 or 1
        SatCount = np.sum(Pos * A_int, axis=1) + np.sum(Neg * (1 - A_int), axis=1)
        return np.sum(SatCount == 0)

    def solve_walksat(A, max_steps):
        """
        A: numpy boolean array of size n
        max_steps: int
        Returns True if solved, False otherwise.
        """
        A_int = A.astype(np.int8)
        # Initial SatCount
        SatCount = np.sum(Pos * A_int, axis=1) + np.sum(Neg * (1 - A_int), axis=1)
        
        violated_indices = np.where(SatCount == 0)[0]
        
        if len(violated_indices) == 0:
            return True

        p_greedy = 0.5
        
        for step in range(max_steps):
            if time.time() > deadline:
                return False
            
            if len(violated_indices) == 0:
                return True

            # Pick a random violated clause
            ci = rng.choice(violated_indices.tolist())
            clause = clause_lits[ci]
            
            # Candidates are the variables in this clause
            # Compute gain for each
            best_gain = -100
            best_var = -1
            gains = []
            
            for lit in clause:
                var_idx = abs(lit) - 1
                # Determine delta for this variable
                # If A[var_idx] is 0, flipping to 1: Delta = Pos[:, var_idx] - Neg[:, var_idx]
                # If A[var_idx] is 1, flipping to 0: Delta = Neg[:, var_idx] - Pos[:, var_idx]
                
                if A_int[var_idx] == 0:
                    Delta = Pos[:, var_idx] - Neg[:, var_idx]
                else:
                    Delta = Neg[:, var_idx] - Pos[:, var_idx]
                
                # Gain calculation:
                # Clauses that become satisfied: SatCount == 0 AND Delta == 1
                # Clauses that become violated: SatCount == 1 AND Delta == -1
                
                # Note: Delta can be -1, 0, 1
                become_sat = np.sum((SatCount == 0) & (Delta == 1))
                become_violated = np.sum((SatCount == 1) & (Delta == -1))
                gain = become_sat - become_violated
                
                gains.append((gain, var_idx))
            
            # Selection
            if rng.random() < p_greedy:
                # Greedy
                gains.sort(key=lambda x: -x[0])
                best_gain, best_var = gains[0]
            else:
                # Random from clause
                best_var = rng.choice([abs(lit)-1 for lit in clause])
                # Recalculate gain for this var to update SatCount correctly? 
                # No, we just need to update SatCount with the correct Delta.
                # We can recompute Delta for the chosen var.
                if A_int[best_var] == 0:
                    Delta = Pos[:, best_var] - Neg[:, best_var]
                else:
                    Delta = Neg[:, best_var] - Pos[:, best_var]
            
            # Update A and A_int
            A_int[best_var] = 1 - A_int[best_var]
            A[best_var] = bool(A_int[best_var])
            
            # Update SatCount
            # We need the Delta for the chosen best_var
            # If we used greedy, we already have Delta? No, we computed it in the loop.
            # Let's just recompute Delta for the chosen var to be safe and simple.
            # It's O(m) which is fine.
            if A_int[best_var] == 1:
                # It was 0, now 1. Delta was Pos - Neg
                Delta = Pos[:, best_var] - Neg[:, best_var]
            else:
                # It was 1, now 0. Delta was Neg - Pos
                Delta = Neg[:, best_var] - Pos[:, best_var]
                
            SatCount = SatCount + Delta
            
            # Update violated_indices
            violated_indices = np.where(SatCount == 0)[0]

        return False

    # Main loop with restarts
    best_assignment = None
    best_violated = m + 1

    # Allow many restarts since each step is fast
    max_restarts = 2000
    base_steps = 20000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initial assignment
        A = np.random.rand(n) < 0.5
        
        # Try WalkSAT
        # Increase steps slightly for later restarts? Or keep constant.
        steps = base_steps + rng.randint(0, 5000)
        
        if solve_walksat(A, steps):
            return A.tolist()
            
        # Track best
        v = count_violated(A)
        if v < best_violated:
            best_violated = v
            best_assignment = A.copy()

    if best_assignment is not None and count_violated(best_assignment) == 0:
        return best_assignment.tolist()
        
    if best_assignment is not None:
        return best_assignment.tolist()
        
    return [rng.random() < 0.5 for _ in range(n)]
