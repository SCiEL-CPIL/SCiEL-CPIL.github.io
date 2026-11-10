"""
Maximum subset of {1..192} such that no element divides the sum of any 3 distinct other elements.
Uses a WalkSAT-style local search over the Max-SAT formulation:
  - Variable x_i (i=1..192): whether integer i is selected.
  - For every 4-tuple (d, a, b, c) of distinct integers with d | (a+b+c),
    we have a clause: NOT(x_d AND x_a AND x_b AND x_c).
  - Objective: maximize sum(x_i) subject to all clauses being satisfied.
  
The WalkSAT approach:
  1. Start with a random assignment (or greedy seed).
  2. While there are violated clauses, pick a random violated clause and flip a variable in it
     (with some probability picking the "best" flip that minimizes total violations).
  3. Track the best valid (or near-valid) assignment found.
  4. Periodically, if we have a valid assignment, try to add more variables.
"""
import time
import random
import sys

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin before 90s kill

N = 192
MAX_K = 48


def _build_clauses():
    """
    Build the list of conflict clauses.
    Each clause is a tuple (d, a, b, c) meaning: NOT(x_d AND x_a AND x_b AND x_c).
    d divides (a+b+c), and d, a, b, c are all distinct.
    
    To keep memory manageable, we store for each variable d, the list of clauses
    where d is the "divisor" variable, and for each variable a, the list of clauses
    where a is one of the "summand" variables.
    """
    # clauses_by_d[d] = list of (a, b, c) such that d | (a+b+c) and d != a,b,c
    # clauses_by_summand[a] = list of (d, b, c) such that d | (a+b+c) and a != d,b,c
    
    # For efficiency, we'll precompute for each d, for each pair (a,b) with a<b,
    # the set of c values (c != a, c != b, c != d) such that d | (a+b+c).
    
    # Actually, let's just enumerate all 4-tuples directly but store them compactly.
    # Number of 4-tuples: C(192,4) ≈ 52 million, but most won't satisfy d|(a+b+c).
    # Expected fraction: for random d,a,b,c, P(d | (a+b+c)) ≈ 1/d on average.
    # Sum over d of C(191,3)/d ≈ C(191,3) * H_192 ≈ 1.13M * 5.7 ≈ 6.4M clauses.
    # That's manageable.
    
    clauses_by_d = [[] for _ in range(N + 1)]  # clauses_by_d[d] = list of (a, b, c)
    clauses_by_summand = [[] for _ in range(N + 1)]  # clauses_by_summand[a] = list of (d, b, c)
    
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            if a == d:
                continue
            for b in range(a + 1, N + 1):
                if b == d:
                    continue
                # Need c such that c != a, c != b, c != d, and (a+b+c) % d == 0
                # c ≡ (-a-b) mod d
                target = (-a - b) % d
                # Find all c in [1, N] with c % d == target, c != a, c != b, c != d
                # c values: target, target+d, target+2d, ... <= N
                # But target could be 0, meaning c is a multiple of d
                start = target if target > 0 else d
                for c in range(start, N + 1, d):
                    if c == a or c == b or c == d:
                        continue
                    # We have a valid clause (d, a, b, c)
                    # But we need to avoid duplicates: each 4-set {d,a,b,c} should appear once
                    # with the unique d that divides the sum of the other three.
                    # Actually, multiple d's could divide the same triple sum.
                    # The constraint is: for each selected d, it must not divide sum of any 3 others.
                    # So the clause is per-d. If d1 and d2 both divide (a+b+c), we have two clauses.
                    # This is correct: both d1 and d2 cannot be selected along with a,b,c.
                    
                    clauses_by_d[d].append((a, b, c))
                    clauses_by_summand[a].append((d, b, c))
                    clauses_by_summand[b].append((d, a, c))
                    clauses_by_summand[c].append((d, a, b))
    
    return clauses_by_d, clauses_by_summand


def _count_violations(assignment, clauses_by_d, clauses_by_summand):
    """
    Count the number of violated clauses given a boolean assignment.
    assignment: list of booleans, assignment[i] = whether integer i+1 is selected.
    Returns (total_violations, violated_clause_indices).
    """
    total_violations = 0
    violated = []
    
    for d in range(1, N + 1):
        if not assignment[d - 1]:
            continue
        for (a, b, c) in clauses_by_d[d]:
            if assignment[a - 1] and assignment[b - 1] and assignment[c - 1]:
                total_violations += 1
                violated.append((d, a, b, c))
    
    return total_violations, violated


def _walksat(assignment, clauses_by_d, clauses_by_summand, time_limit, seed=0):
    """
    WalkSAT-style local search to minimize violated clauses.
    Returns the best assignment found (maximizing cardinality among valid ones,
    or minimizing violations if no valid assignment found).
    """
    random.seed(seed)
    start_time = time.time()
    
    best_assignment = list(assignment)
    best_cardinality = sum(assignment)
    best_violations = _count_violations(assignment, clauses_by_d, clauses_by_summand)[0]
    
    if best_violations == 0:
        return best_assignment
    
    # WalkSAT parameters
    p = 0.5  # probability of picking the "best" flip vs random flip
    max_iterations = 500000
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Count violations
        violations, violated_clauses = _count_violations(assignment, clauses_by_d, clauses_by_summand)
        
        if violations == 0:
            # Valid assignment! Update best if cardinality is higher
            card = sum(assignment)
            if card > best_cardinality:
                best_cardinality = card
                best_assignment = list(assignment)
                best_violations = 0
            # Try to improve by adding more elements
            # Try flipping 0s to 1s
            improved = False
            for i in range(N):
                if assignment[i]:
                    continue
                # Check if flipping i to 1 keeps us valid
                old_val = assignment[i]
                assignment[i] = True
                new_violations, _ = _count_violations(assignment, clauses_by_d, clauses_by_summand)
                if new_violations == 0:
                    card = sum(assignment)
                    if card > best_cardinality:
                        best_cardinality = card
                        best_assignment = list(assignment)
                    improved = True
                else:
                    assignment[i] = old_val
            
            if not improved:
                # No more additions possible, break
                break
        
        # Pick a random violated clause
        clause = random.choice(violated_clauses)
        d, a, b, c = clause
        # The variables in this clause are d, a, b, c (all currently True)
        # We need to flip one of them to False
        candidates = [d, a, b, c]
        
        if random.random() < p:
            # Best flip: pick the variable whose flip minimizes new violations
            best_var = None
            best_new_violations = float('inf')
            for var in candidates:
                old_val = assignment[var - 1]
                assignment[var - 1] = False
                new_violations, _ = _count_violations(assignment, clauses_by_d, clauses_by_summand)
                assignment[var - 1] = old_val
                if new_violations < best_new_violations:
                    best_new_violations = new_violations
                    best_var = var
            assignment[best_var - 1] = False
        else:
            # Random flip
            var = random.choice(candidates)
            assignment[var - 1] = False
        
        # Track best
        card = sum(assignment)
        cur_violations = _count_violations(assignment, clauses_by_d, clauses_by_summand)[0]
        if cur_violations == 0 and card > best_cardinality:
            best_cardinality = card
            best_assignment = list(assignment)
            best_violations = 0
    
    return best_assignment


def _greedy_construct(order, clauses_by_d, clauses_by_summand):
    """Greedy construction with given order."""
    assignment = [False] * N
    for c in order:
        # Check if adding c creates any violations
        # c is 1-indexed
        idx = c - 1
        # Check if c divides sum of any 3 already selected
        selected = [i + 1 for i in range(N) if assignment[i]]
        if len(selected) >= 3:
            valid = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    for k in range(j + 1, len(selected)):
                        if (selected[i] + selected[j] + selected[k]) % c == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
        
        # Check if any already selected a divides (c + x + y) for distinct x,y in selected\{a}
        for a in selected:
            target = (-c) % a
            # Need distinct x, y from selected\{a} with (x+y) % a == target
            res_seen = set()
            for x in selected:
                if x == a:
                    continue
                xr = x % a
                needed = (target - xr) % a
                if needed in res_seen:
                    valid = False
                    break
                res_seen.add(xr)
            if not valid:
                break
        else:
            valid = True
        
        if valid:
            assignment[idx] = True
    
    return assignment


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Build clauses
    clauses_by_d, clauses_by_summand = _build_clauses()
    
    best_assignment = [False] * N
    best_cardinality = 0
    
    # Multiple greedy starts
    random.seed(42)
    
    # Descending
    if time.time() < deadline - 10:
        order = list(range(N, 0, -1))
        assign = _greedy_construct(order, clauses_by_d, clauses_by_summand)
        card = sum(assign)
        if card > best_cardinality:
            best_cardinality = card
            best_assignment = list(assign)
    
    # Ascending
    if time.time() < deadline - 10:
        order = list(range(1, N + 1))
        assign = _greedy_construct(order, clauses_by_d, clauses_by_summand)
        card = sum(assign)
        if card > best_cardinality:
            best_cardinality = card
            best_assignment = list(assign)
    
    # Random orders
    for i in range(20):
        if time.time() > deadline - 20:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        assign = _greedy_construct(order, clauses_by_d, clauses_by_summand)
        card = sum(assign)
        if card > best_cardinality:
            best_cardinality = card
            best_assignment = list(assign)
    
    # WalkSAT from best assignment and from random starts
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Run WalkSAT from the best greedy solution
        time_per_walksat = max(1.0, remaining_time / 5)
        
        # From best assignment
        if time.time() < deadline - 2:
            result = _walksat(best_assignment, clauses_by_d, clauses_by_summand, 
                              time_limit=min(time_per_walksat, deadline - time.time() - 2), seed=1)
            card = sum(result)
            viol = _count_violations(result, clauses_by_d, clauses_by_summand)[0]
            if viol == 0 and card > best_cardinality:
                best_cardinality = card
                best_assignment = list(result)
        
        # From random starts
        for seed in range(2, 12):
            if time.time() > deadline - 3:
                break
            random.seed(seed)
            # Start with a random sparse assignment
            assign = [random.random() < 0.25 for _ in range(N)]
            result = _walksat(assign, clauses_by_d, clauses_by_summand,
                              time_limit=min(time_per_walksat, deadline - time.time() - 3), seed=seed)
            card = sum(result)
            viol = _count_violations(result, clauses_by_d, clauses_by_summand)[0]
            if viol == 0 and card > best_cardinality:
                best_cardinality = card
                best_assignment = list(result)
    
    # Final local improvement: try to add more elements to best_assignment
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for c in range(N, 0, -1):
                if best_assignment[c - 1]:
                    continue
                # Try adding c
                old_val = best_assignment[c - 1]
                best_assignment[c - 1] = True
                viol, _ = _count_violations(best_assignment, clauses_by_d, clauses_by_summand)
                if viol == 0:
                    improved = True
                    best_cardinality = sum(best_assignment)
                else:
                    best_assignment[c - 1] = old_val
    
    # Convert to list of integers
    result = [i + 1 for i in range(N) if best_assignment[i]]
    
    # Final validation
    if len(result) > 0:
        valid = True
        n = len(result)
        for i in range(n):
            d = result[i]
            others = [result[j] for j in range(n) if j != i]
            m = len(others)
            for x in range(m):
                for y in range(x + 1, m):
                    for z in range(y + 1, m):
                        if (others[x] + others[y] + others[z]) % d == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            # Fallback: use greedy descending which is usually valid
            order = list(range(N, 0, -1))
            result = _greedy_construct(order, clauses_by_d, clauses_by_summand)
            result = [i + 1 for i in range(N) if result[i]]
    
    return result


if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} elements")
    print(result)
