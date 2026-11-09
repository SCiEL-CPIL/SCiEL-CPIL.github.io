import time
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin
    
    # 1-indexed assignment: 0=unassigned, 1=True, -1=False
    assignment = [0] * (n + 1)
    trail = []
    trail_lim = []
    
    def get_status(c):
        unassigned = []
        for l in c:
            v = abs(l)
            val = assignment[v]
            if val == 0:
                unassigned.append(l)
            else:
                if (l > 0 and val == 1) or (l < 0 and val == -1):
                    return 'satisfied', 0
        if len(unassigned) == 0:
            return 'empty', 0
        if len(unassigned) == 1:
            return 'unit', unassigned[0]
        return 'other', 0

    def propagate():
        nonlocal trail
        changed = True
        while changed:
            changed = False
            for c in clauses:
                status, lit = get_status(c)
                if status == 'empty':
                    return False
                if status == 'unit':
                    v = abs(lit)
                    assignment[v] = 1 if lit > 0 else -1
                    trail.append(v)
                    changed = True
        return True

    def dpll():
        nonlocal deadline
        
        if time.time() > deadline:
            return None
            
        if not propagate():
            return False
            
        for c in clauses:
            status, _ = get_status(c)
            if status == 'satisfied':
                continue
            if status == 'empty':
                return False
            if status == 'other':
                # Pick a variable from this clause
                branch_var = -1
                for l in c:
                    v = abs(l)
                    if assignment[v] == 0:
                        branch_var = v
                        break
                
                if branch_var == -1:
                    continue # Should not happen for 'other'
                
                trail_lim.append(len(trail))
                
                # Try True
                assignment[branch_var] = 1
                trail.append(branch_var)
                
                res = dpll()
                
                if res is True:
                    return True
                if res is None:
                    return None
                
                # Backtrack
                while len(trail) > trail_lim[-1]:
                    var = trail.pop()
                    assignment[var] = 0
                trail_lim.pop()
                
                # Try False
                assignment[branch_var] = -1
                trail.append(branch_var)
                
                res = dpll()
                
                if res is True:
                    return True
                if res is None:
                    return None
                
                # Backtrack
                while len(trail) > trail_lim[-1]:
                    var = trail.pop()
                    assignment[var] = 0
                trail_lim.pop()
                
                return False
        
        return True

    result = dpll()
    
    if result is True:
        final_assign = [False] * n
        for i in range(1, n + 1):
            final_assign[i-1] = (assignment[i] == 1)
        return final_assign
    else:
        # Fallback
        import random
        rng = random.Random(42)
        return [rng.random() < 0.5 for _ in range(n)]
