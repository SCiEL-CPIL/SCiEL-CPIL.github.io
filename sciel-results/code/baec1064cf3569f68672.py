import time
import random
from collections import defaultdict

def _precompute_conflicts(n):
    """
    Precompute all triple violations (a, b, c) with a < b <= c where a | (b+c).
    Returns a list of tuples (a, b, c).
    """
    conflicts = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b, n + 1):
                if (b + c) % a == 0:
                    conflicts.append((a, b, c))
    return conflicts

def _check_validity(S):
    """
    Check if set S is valid: for all a < b <= c in S, a does not divide b+c.
    """
    S_list = sorted(S)
    for i, a in enumerate(S_list):
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            for k in range(j, len(S_list)):
                c = S_list[k]
                if (b + c) % a == 0:
                    return False
    return True

def _count_conflicts(S, conflicts):
    """
    Count how many conflict triples are violated by set S.
    A conflict (a, b, c) is violated if all three are in S.
    """
    S_set = set(S)
    count = 0
    for a, b, c in conflicts:
        if a in S_set and b in S_set and c in S_set:
            count += 1
    return count

def _walksat_with_clauses(n, conflicts, initial_set, max_steps, deadline):
    """
    WalkSAT-style local search with clause weighting.
    Each number 1..n is a Boolean variable (in/out of set).
    Each conflict (a,b,c) is a clause that should be falsified (at least one of a,b,c not in S).
    We want to maximize the number of variables set to True while satisfying all clauses.
    
    Strategy: Start from initial_set. If there are violated clauses, pick a violated clause
    and flip one of its variables (preferably removing a variable that's in many violated clauses).
    Also try adding variables that don't create new violations.
    """
    S = set(initial_set)
    S_list = sorted(S)
    
    # Build conflict index: for each variable, list of conflicts it participates in
    var_conflicts = defaultdict(list)
    for idx, (a, b, c) in enumerate(conflicts):
        var_conflicts[a].append(idx)
        var_conflicts[b].append(idx)
        var_conflicts[c].append(idx)
    
    # Track which conflicts are currently violated
    violated = set()
    for a, b, c in conflicts:
        if a in S and b in S and c in S:
            violated.add((a, b, c))
    
    best_size = len(S)
    best_S = set(S)
    
    step = 0
    while time.time() < deadline and step < max_steps:
        step += 1
        
        # If no violations, try to add a variable
        if not violated:
            added = False
            for v in range(1, n + 1):
                if v in S:
                    continue
                # Check if adding v creates any violation
                new_violations = []
                for conf_idx in var_conflicts[v]:
                    a, b, c = conflicts[conf_idx]
                    others = [x for x in (a, b, c) if x != v]
                    if all(x in S for x in others):
                        new_violations.append(conf_idx)
                if not new_violations:
                    S.add(v)
                    # Update violated set
                    for conf_idx in new_violations:
                        violated.add(conflicts[conf_idx])
                    added = True
                    break
            if not added:
                # No more variables can be added without violation
                # Try flipping: remove one variable to allow adding others
                # Find variable that's in the most violated clauses
                if violated:
                    var_violation_count = defaultdict(int)
                    for conf in violated:
                        for v in conf:
                            var_violation_count[v] += 1
                    # Remove variable with highest violation count
                    v_to_remove = max(var_violation_count.items(), key=lambda x: x[1])[0]
                    if v_to_remove in S:
                        S.remove(v_to_remove)
                        # Update violated set
                        for conf_idx in var_conflicts[v_to_remove]:
                            a, b, c = conflicts[conf_idx]
                            if not (a in S and b in S and c in S):
                                violated.discard((a, b, c))
                        continue
                # Stuck, break
                break
        else:
            # There are violations. Pick a violated clause and flip one of its variables.
            # Choose the variable that appears in the most violated clauses (to fix more at once)
            conf = random.choice(list(violated))
            a, b, c = conf
            candidates = [v for v in (a, b, c) if v in S]
            if not candidates:
                violated.discard(conf)
                continue
            
            # Score each candidate by how many other violated clauses it's in
            var_scores = {}
            for v in candidates:
                score = 0
                for conf_idx in var_conflicts[v]:
                    if conflicts[conf_idx] in violated:
                        score += 1
                var_scores[v] = score
            
            # Pick the variable with highest score (most impactful removal)
            v_to_remove = max(var_scores.items(), key=lambda x: x[1])[0]
            
            S.remove(v_to_remove)
            
            # Update violated set: remove all conflicts that involve v_to_remove
            for conf_idx in var_conflicts[v_to_remove]:
                a2, b2, c2 = conflicts[conf_idx]
                if not (a2 in S and b2 in S and c2 in S):
                    violated.discard((a2, b2, c2))
            
            # After removal, try to add v_to_remove back or other variables if no more violations
            # Actually, let's just continue the loop; next iteration will try to add variables
        
        # Track best
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    return best_S

def _greedy_with_order(n, order):
    """
    Runs the greedy algorithm using the specified order of candidates.
    Returns the set of selected numbers.
    """
    S = set()
    S_list = []
    
    for v in order:
        valid = True
        L = len(S_list)
        for i in range(L):
            b = S_list[i]
            for j in range(i, L):
                c = S_list[j]
                if v < b:
                    if (b + c) % v == 0:
                        valid = False
                        break
                elif b < v <= c:
                    if (v + c) % b == 0:
                        valid = False
                        break
                elif b < c < v:
                    if (c + v) % b == 0:
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            continue
        
        S.add(v)
        S_list.append(v)
        
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    n = 256
    numbers = list(range(1, n + 1))
    
    # Precompute all conflicts
    conflicts = _precompute_conflicts(n)
    
    best_set = set()
    
    # 1. Deterministic greedy (descending)
    order_desc = sorted(numbers, reverse=True)
    S = _greedy_with_order(n, order_desc)
    if len(S) > len(best_set):
        best_set = S
    
    # 2. Random restarts with greedy
    attempts = 0
    while time.time() < deadline - 30:  # Leave 30s for WalkSAT
        random.shuffle(numbers)
        S = _greedy_with_order(n, numbers)
        if len(S) > len(best_set):
            best_set = S
        attempts += 1
    
    # 3. WalkSAT local search starting from best_set
    if time.time() < deadline - 5:
        ws_result = _walksat_with_clauses(n, conflicts, best_set, max_steps=50000, deadline=deadline - 5)
        if len(ws_result) > len(best_set):
            best_set = ws_result
    
    # 4. Final local search: try to add any number
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for v in range(1, n + 1):
            if v in best_set:
                continue
            valid = True
            S_list = list(best_set)
            L = len(S_list)
            for i in range(L):
                b = S_list[i]
                for j in range(i, L):
                    c = S_list[j]
                    if v < b:
                        if (b + c) % v == 0:
                            valid = False
                            break
                    elif b < v <= c:
                        if (v + c) % b == 0:
                            valid = False
                            break
                    elif b < c < v:
                        if (c + v) % b == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_set.add(v)
                improved = True
                break
    
    return sorted(best_set)
