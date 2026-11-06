def solve(instance):
        import time
        import random
        
        n = instance.get("n", 0)
        raw_clauses = instance.get("clauses", [])
        
        # Robust parsing to avoid TypeError
        clauses = []
        if isinstance(raw_clauses, (list, tuple)):
            if len(raw_clauses) > 0:
                if isinstance(raw_clauses[0], (list, tuple)):
                    clauses = [list(c) for c in raw_clauses]
                else:
                    # Assume flat list of 3-literal clauses
                    clauses = [list(raw_clauses[i:i+3]) for i in range(0, len(raw_clauses), 3)]
            else:
                clauses = []
        elif isinstance(raw_clauses, int):
            # If clauses is an int, we can't proceed with standard parsing, 
            # but we avoid the crash. Return random assignment.
            clauses = []
        else:
            clauses = []

        if n == 0 or not clauses:
            return [False] * n

        deadline = time.time() + 290 # Safety margin

        def evaluate(assignment):
            unsat = []
            for ci, clause in enumerate(clauses):
                sat = False
                for lit in clause:
                    v = abs(lit) - 1
                    if 0 <= v < n:
                        if assignment[v] == (lit > 0):
                            sat = True
                            break
                if not sat:
                    unsat.append(ci)
            return unsat

        # 1. Try Gaussian Elimination (Linear/XOR assumption)
        # Construct matrix A and vector b for x1 ^ x2 ^ x3 = 1
        # Note: This is a heuristic. If the instances are truly linear, this will solve them.
        # If not, it will likely fail verification and we fall back.
        try:
            if time.time() < deadline:
                m = len(clauses)
                # Matrix size (n+1) x (m+1) to handle augmented form? 
                # Or just A (m x n) and b (m).
                # We want to solve A x = b.
                # Since b is unknown (could be 0 or 1 for each clause), we can't directly solve.
                # However, if the planted solution is unique and the system is full rank, 
                # we might be able to infer? No.
                # Strategy: Try solving for b=1 for all clauses.
                A = [[0] * n for _ in range(m)]
                for i, clause in enumerate(clauses):
                    for lit in clause:
                        v = abs(lit) - 1
                        if 0 <= v < n:
                            A[i][v] = 1
                
                # Gaussian Elimination over GF(2)
                # Augmented matrix A | b (b=1)
                aug = [row[:] + [1] for row in A]
                
                rank = 0
                for col in range(n):
                    # Find pivot
                    pivot = -1
                    for r in range(rank, m):
                        if aug[r][col] == 1:
                            pivot = r
                            break
                    if pivot == -1:
                        continue
                    # Swap
                    aug[rank], aug[pivot] = aug[pivot], aug[rank]
                    # Eliminate
                    for r in range(m):
                        if r != rank and aug[r][col] == 1:
                            for c in range(n + 1):
                                aug[r][c] ^= aug[rank][c]
                    rank += 1
                    if rank == n:
                        break

                # Check consistency
                consistent = True
                for r in range(rank, m):
                    # If row is 0...0 | 1, inconsistent
                    if aug[r][n] == 1:
                        consistent = False
                        break
                
                if consistent:
                    # Extract solution (free variables set to 0)
                    assignment = [False] * n
                    # Back substitution
                    # Since we did full elimination, rows are in reduced form
                    for r in range(rank):
                        # Find the pivot column for this row
                        pivot_col = -1
                        for c in range(n):
                            if aug[r][c] == 1:
                                pivot_col = c
                                break
                        if pivot_col != -1:
                            # x_pivot = aug[r][n] ^ sum(x_free)
                            # We set free vars to 0, so x_pivot = aug[r][n]
                            assignment[pivot_col] = bool(aug[r][n])
                    
                    # Verify against 3-CNF
                    if not evaluate(assignment):
                        return assignment
        except Exception:
            pass

        # 2. Fallback: WalkSAT
        # Precompute variable occurrences for faster evaluation? 
        # For n=50, simple evaluation is fast enough.
        
        best_assignment = None
        best_unsat_count = float('inf')
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            rng = random.Random(attempt * 7919 + 13)
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            unsat = evaluate(assignment)
            if not unsat:
                return assignment
            
            current_unsat = len(unsat)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
            
            # WalkSAT loop
            weight = 0.5
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                if not unsat:
                    return assignment
                    
                # Pick random unsat clause
                ci = rng.choice(unsat)
                clause = clauses[ci]
                
                # Pick literal to flip
                # With prob weight, pick a literal that is currently false in this clause
                # (which is all of them, since clause is unsat)
                # So just pick random literal from clause
                lit = rng.choice(clause)
                v = abs(lit) - 1
                
                # Flip
                assignment[v] = not assignment[v]
                
                # Recompute unsat
                # Optimization: Only recompute affected clauses? 
                # For simplicity and robustness, recompute all. n=50 is small.
                unsat = evaluate(assignment)
                
                cur_unsat = len(unsat)
                if cur_unsat < best_unsat_count:
                    best_unsat_count = cur_unsat
                    best_assignment = assignment.copy()

        if best_assignment is not None:
            return best_assignment
            
        # Final fallback
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]
