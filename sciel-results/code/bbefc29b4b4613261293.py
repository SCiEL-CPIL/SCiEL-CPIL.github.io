"""3-SAT solver with optimized WalkSAT + incremental state tracking."""
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["instance_clauses"] if "instance_clauses" in instance else instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 330s kill

    # Preprocess: remove tautological clauses, dedupe
    seen = set()
    filtered_clauses = []
    for c in clauses:
        # Check for tautology (both x and -x in clause)
        lits = set(c)
        if any(-l in lits for l in lits):
            continue
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            filtered_clauses.append(c)
    clauses = filtered_clauses

    clause_count = len(clauses)
    if clause_count == 0:
        return [True] * n

    # Build variable occurrence lists
    # var_pos[var] = list of clause indices where var appears positively
    # var_neg[var] = list of clause indices where var appears negatively
    var_pos = [[] for _ in range(n)]
    var_neg = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            idx = abs(l) - 1
            if l > 0:
                var_pos[idx].append(i)
            else:
                var_neg[idx].append(i)

    # Precompute clause literals as (var_index, is_positive) for faster access
    # clause_lits[i] = list of (var_idx, is_pos)
    clause_lits = []
    for c in clauses:
        lits = []
        for l in c:
            idx = abs(l) - 1
            is_pos = l > 0
            lits.append((idx, is_pos))
        clause_lits.append(lits)

    rng = random.Random(42)

    # State variables
    # assignment[var] is True/False
    # sat_count[clause] is the number of literals in the clause that are currently satisfied
    # unsat_clauses is a set of clause indices where sat_count == 0

    def update_sat_counts(assignment):
        """Initialize sat_count and unsat_clauses based on assignment."""
        sat_counts = [0] * clause_count
        unsat = set()
        for i in range(clause_count):
            count = 0
            for var_idx, is_pos in clause_lits[i]:
                if assignment[var_idx] == is_pos:
                    count += 1
            sat_counts[i] = count
            if count == 0:
                unsat.add(i)
        return sat_counts, unsat

    def flip_variable(var_idx, assignment, sat_counts, unsat):
        """
        Flip variable var_idx. Update sat_counts and unsat set incrementally.
        Returns the number of clauses that became unsatisfied (breaking count) 
        and the number of clauses that became satisfied (fixing count) for heuristics if needed, 
        but here we just update state.
        """
        new_val = not assignment[var_idx]
        
        # Clauses containing var_idx positively
        for ci in var_pos[var_idx]:
            # Literal (var_idx, True)
            # Before flip: if assignment[var_idx] was True, literal was SAT. Now it becomes UNSAT.
            # If assignment[var_idx] was False, literal was UNSAT. Now it becomes SAT.
            if assignment[var_idx]:
                # Was SAT, becomes UNSAT
                sat_counts[ci] -= 1
                if sat_counts[ci] == 0:
                    unsat.add(ci)
            else:
                # Was UNSAT, becomes SAT
                sat_counts[ci] += 1
                if sat_counts[ci] == 1: # Was 0, now 1
                    unsat.discard(ci)
        
        # Clauses containing var_idx negatively
        for ci in var_neg[var_idx]:
            # Literal (var_idx, False)
            # Before flip: if assignment[var_idx] was False, literal was SAT. Now it becomes UNSAT.
            # If assignment[var_idx] was True, literal was UNSAT. Now it becomes SAT.
            if not assignment[var_idx]:
                # Was SAT, becomes UNSAT
                sat_counts[ci] -= 1
                if sat_counts[ci] == 0:
                    unsat.add(ci)
            else:
                # Was UNSAT, becomes SAT
                sat_counts[ci] += 1
                if sat_counts[ci] == 1: # Was 0, now 1
                    unsat.discard(ci)
        
        assignment[var_idx] = new_val

    best_assignment = None
    best_unsat_count = clause_count + 1

    max_restarts = 10000
    steps_per_restart = 10000
    noise = 0.25  # probability of random flip

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize assignment
        if restart == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Perturb from best or random
            if best_assignment is not None and rng.random() < 0.7:
                assignment = best_assignment[:]
                # Flip a few random variables
                for _ in range(n // 4 + 1):
                    assignment[rng.randint(0, n - 1)] = not assignment[rng.randint(0, n - 1)]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]

        sat_counts, unsat = update_sat_counts(assignment)
        current_unsat = len(unsat)
        
        if current_unsat == 0:
            return assignment

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            if not unsat:
                return assignment

            # With some probability, do a random flip
            if rng.random() < noise:
                var = rng.randint(0, n - 1)
                flip_variable(var, assignment, sat_counts, unsat)
                continue

            # Choose a random unsatisfied clause
            clause_idx = next(iter(unsat)) # Randomly pick from set? next(iter()) is arbitrary. 
            # To be truly random, we should pick randomly from the set.
            # Since sets are unordered, let's convert to list if small, or use random choice.
            # For performance, if unsat is large, random.choice(list(unsat)) is expensive.
            # However, WalkSAT usually picks randomly.
            # Let's use random.choice on a list for correctness, but optimize if possible.
            # Actually, just picking any unsat clause works for the algorithm, but random is better for escaping local minima.
            # Let's stick to random choice for robustness.
            clause_idx = rng.choice(list(unsat))
            clause = clause_lits[clause_idx]

            # Score each variable in the clause
            # Heuristic: 
            # 1. Count how many unsatisfied clauses this variable appears in (Fixing potential)
            # 2. Count how many satisfied clauses this variable appears in (Breaking cost)
            # Score = Fixes - Breaks * weight? 
            # Standard WalkSAT: 
            #   For each literal in clause:
            #     Count 'fixes' = number of unsat clauses containing this literal that would be satisfied
            #     Count 'breaks' = number of sat clauses containing this literal that would be unsatisfied
            #     Choose var with max (fixes), tie-break min (breaks)
            
            best_var = -1
            best_score = (-1, 1000000) # (fixes, -breaks) -> maximize fixes, minimize breaks
            
            for var_idx, is_pos in clause:
                # Determine which lists to check
                # The literal is (var_idx, is_pos).
                # If is_pos is True, the literal is x. 
                #   It appears in var_pos[var_idx] as a positive literal.
                #   Flipping x to True satisfies positive occurrences, breaks negative occurrences.
                # If is_pos is False, the literal is -x.
                #   It appears in var_neg[var_idx] as a negative literal.
                #   Flipping x to False satisfies negative occurrences, breaks positive occurrences.
                
                fixes = 0
                breaks = 0
                
                if is_pos:
                    # Variable is x. We want to make it True.
                    # Fixes: Unsat clauses where x is present positively (and currently False, so x is False).
                    #        Since clause is unsat, all literals are False. So x is False.
                    #        Flipping x to True satisfies x.
                    #        We need to check if this is the ONLY unsat literal? No, WalkSAT just counts.
                    #        Actually, standard WalkSAT counts:
                    #        Fixes: Number of unsat clauses containing this literal.
                    #        Breaks: Number of sat clauses containing this literal.
                    
                    # Check unsat clauses containing x positively
                    for ci in var_pos[var_idx]:
                        if sat_counts[ci] == 0:
                            fixes += 1
                    # Check sat clauses containing x positively (these will be broken if we flip x to True? No, if x is False and we flip to True, positive x becomes True (sat). 
                    # Wait.
                    # Current state: x is False (because clause is unsat and x is in it, so x must be False to make literal False? No.
                    # If clause is unsat, ALL literals are False.
                    # If literal is x, then x is False.
                    # If we flip x to True:
                    #   Literal x becomes True.
                    #   Other clauses with x (positive): x was False (unsat contrib), becomes True (sat contrib). 
                    #     If clause was unsat (count 0), it becomes sat (count 1). Fix.
                    #     If clause was sat (count > 0), it remains sat (count increases). No break.
                    #   Other clauses with -x (negative): x was False (so -x was True, sat contrib).
                    #     If we flip x to True, -x becomes False.
                    #     If clause was sat (count > 0), count decreases. If it was 1, it becomes 0 (Unsat). Break.
                    #     If clause was unsat (count 0), it remains unsat.
                    
                    # So:
                    # Fixes: Unsat clauses with x (positive).
                    # Breaks: Sat clauses with -x (negative).
                    
                    for ci in var_pos[var_idx]:
                        if sat_counts[ci] == 0:
                            fixes += 1
                    for ci in var_neg[var_idx]:
                        if sat_counts[ci] > 0:
                            breaks += 1

                else:
                    # Variable is -x. We want to make it True (so x becomes False).
                    # Current state: -x is False, so x is True.
                    # Flipping x to False:
                    #   Literal -x becomes True.
                    #   Clauses with -x (negative):
                    #     If unsat, becomes sat. Fix.
                    #     If sat, remains sat.
                    #   Clauses with x (positive):
                    #     x was True (sat contrib). Becomes False.
                    #     If sat, count decreases. If 1->0, Break.
                    #     If unsat, remains unsat.
                    
                    for ci in var_neg[var_idx]:
                        if sat_counts[ci] == 0:
                            fixes += 1
                    for ci in var_pos[var_idx]:
                        if sat_counts[ci] > 0:
                            breaks += 1

                # Score: Maximize fixes, Minimize breaks
                # We can use a tuple (fixes, -breaks) and take max
                score = (fixes, -breaks)
                if score > best_score:
                    best_score = score
                    best_var = var_idx

            if best_var >= 0:
                flip_variable(best_var, assignment, sat_counts, unsat)
            else:
                # Random flip within clause
                var_idx, is_pos = clause[rng.randint(0, len(clause) - 1)]
                flip_variable(var_idx, assignment, sat_counts, unsat)

            # Periodically check unsat count for best tracking
            if step % 500 == 0:
                current_unsat = len(unsat)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]

    # Phase 2: Local search from best assignment (optional, often WalkSAT is enough with enough restarts)
    # Given the speedup, we might not need a separate phase, but let's add a quick polish
    if best_assignment is not None:
        assignment = best_assignment[:]
        sat_counts, unsat = update_sat_counts(assignment)
        for _ in range(10000):
            if time.time() > deadline:
                break
            if not unsat:
                return assignment
            
            # Greedy: pick variable that fixes most
            clause_idx = rng.choice(list(unsat))
            clause = clause_lits[clause_idx]
            
            best_var = -1
            best_score = (-1, 1000000)
            
            for var_idx, is_pos in clause:
                fixes = 0
                breaks = 0
                if is_pos:
                    for ci in var_pos[var_idx]:
                        if sat_counts[ci] == 0:
                            fixes += 1
                    for ci in var_neg[var_idx]:
                        if sat_counts[ci] > 0:
                            breaks += 1
                else:
                    for ci in var_neg[var_idx]:
                        if sat_counts[ci] == 0:
                            fixes += 1
                    for ci in var_pos[var_idx]:
                        if sat_counts[ci] > 0:
                            breaks += 1
                            
                score = (fixes, -breaks)
                if score > best_score:
                    best_score = score
                    best_var = var_idx

            if best_var >= 0:
                flip_variable(best_var, assignment, sat_counts, unsat)
                current_unsat = len(unsat)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                if current_unsat == 0:
                    return assignment
            else:
                break

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
