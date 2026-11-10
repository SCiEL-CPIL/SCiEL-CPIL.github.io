import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    total_points = q * q
    deadline = time.time() + 580.0

    # Field arithmetic setup: F_49 = F_7[alpha], alpha^2 = 3
    k = 3

    def mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua*va + ub*vb*k) % p, (ua*vb + ub*va) % p)
        
    def add(u, v):
        return ((u[0]+v[0])%p, (u[1]+v[1])%p)
        
    def get_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Precompute lines
    lines = []
    line_membership = [[] for _ in range(total_points)]
    
    # v = (0, 1) vertical lines
    for xa in range(p):
        for xb in range(p):
            pts = []
            for ya in range(p):
                for yb in range(p):
                    idx = get_idx(xa, xb, ya, yb)
                    pts.append(idx)
            lines.append(pts)
            for pt in pts:
                line_membership[pt].append(len(lines) - 1)
                
    # v = (1, m)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            seen_lines = set()
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            pts = []
                            for ta in range(p):
                                for tb in range(p):
                                    t = (ta, tb)
                                    x = add((xa, xb), t)
                                    y = add((ya, yb), mul(m, t))
                                    idx = get_idx(x[0], x[1], y[0], y[1])
                                    pts.append(idx)
                            line_set = frozenset(pts)
                            if line_set not in seen_lines:
                                seen_lines.add(line_set)
                                lines.append(list(line_set))
                                for pt in line_set:
                                    line_membership[pt].append(len(lines) - 1)

    num_lines = len(lines)
    
    # Initialize S based on the unital seed
    squares = {0, 1, 2, 4}
    S = set()
    in_S = np.zeros(total_points, dtype=bool)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa + ya*ya - yb*yb*k) % p == 0:
                        if ya not in squares:
                            idx = get_idx(xa, xb, ya, yb)
                            S.add(idx)
                            in_S[idx] = True
                            
    # Initialize line_count
    line_count = np.zeros(num_lines, dtype=np.int32)
    for x in S:
        for l_idx in line_membership[x]:
            line_count[l_idx] += 1
            
    # Initialize empty_lines_count and singleton_lines_count
    empty_lines_count = np.zeros(total_points, dtype=np.int32)
    singleton_lines_count = np.zeros(total_points, dtype=np.int32)
    
    for x in range(total_points):
        for l_idx in line_membership[x]:
            c = line_count[l_idx]
            if c == 0:
                empty_lines_count[x] += 1
            elif c == 1:
                singleton_lines_count[x] += 1

    # Check validity
    def is_valid():
        for x in range(total_points):
            if in_S[x]:
                if singleton_lines_count[x] == 0:
                    return False
            else:
                if empty_lines_count[x] == 0:
                    return False
        return True

    # Greedy expansion
    best_S_size = len(S)
    best_S = set(S)
    best_empty = empty_lines_count.copy()
    best_singleton = singleton_lines_count.copy()
    best_line_count = line_count.copy()
    best_in_S = in_S.copy()
    
    num_restarts = 50 # Increase restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Reset to best found so far
        S = set(best_S)
        in_S = best_in_S.copy()
        line_count = best_line_count.copy()
        empty_lines_count = best_empty.copy()
        singleton_lines_count = best_singleton.copy()
        
        if not is_valid():
            continue
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = [x for x in range(total_points) if not in_S[x] and empty_lines_count[x] > 0]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                
                # Try adding x
                lines_through_x = line_membership[x]
                affected_points = set()
                for l_idx in lines_through_x:
                    affected_points.update(lines[l_idx])
                
                # Save state
                old_empty = {pt: empty_lines_count[pt] for pt in affected_points}
                old_singleton = {pt: singleton_lines_count[pt] for pt in affected_points}
                old_line_count = {l_idx: line_count[l_idx] for l_idx in lines_through_x}
                
                # Apply change
                in_S[x] = True
                S.add(x)
                
                for l_idx in lines_through_x:
                    old_c = old_line_count[l_idx]
                    line_count[l_idx] += 1
                    new_c = old_c + 1
                    
                    # Update counts for points on this line
                    for pt in lines[l_idx]:
                        if old_c == 0:
                            empty_lines_count[pt] -= 1
                            if pt == x:
                                singleton_lines_count[pt] += 1
                        elif old_c == 1:
                            # The point in S on this line loses a singleton line
                            # Find that point
                            for y in lines[l_idx]:
                                if in_S[y] and y != x: # x was not in S before
                                    singleton_lines_count[y] -= 1
                                    break
                            # x gains a singleton line? No, x was not in S.
                            # x is now in S. The line had 1 point y. Now it has x, y.
                            # For x: this line is NOT singleton (has 2 points).
                            # So x does not gain a singleton count from this line.
                        # else old_c >= 2: no change to empty/singleton counts
                        # Note: if pt == x and old_c == 1, x is now on a 2-point line.
                        # x was not in S, so it didn't have singleton count for this line.
                        # Now x is in S. Does it have singleton count? No, count is 2.
                        # So no change for x in this case.
                        
                # Recalculate counts for x specifically?
                # x's empty count decreases by (number of empty lines through x).
                # x's singleton count increases by (number of empty lines through x).
                # Wait, the loop above handles this.
                # If old_c == 0, empty[pt]--, and if pt==x, singleton[x]++.
                # If old_c == 1, singleton[y]--. x is not y.
                # So x's singleton count increases only from lines that were empty.
                # This is correct.
                
                # Check validity of affected points
                valid = True
                for pt in affected_points:
                    if in_S[pt]:
                        if singleton_lines_count[pt] == 0:
                            valid = False
                            break
                    else:
                        if empty_lines_count[pt] == 0:
                            valid = False
                            break
                            
                if valid:
                    improved = True
                    if len(S) > best_S_size:
                        best_S_size = len(S)
                        best_S = set(S)
                        best_empty = empty_lines_count.copy()
                        best_singleton = singleton_lines_count.copy()
                        best_line_count = line_count.copy()
                        best_in_S = in_S.copy()
                else:
                    # Rollback
                    in_S[x] = False
                    S.remove(x)
                    for l_idx in lines_through_x:
                        line_count[l_idx] = old_line_count[l_idx]
                    for pt in affected_points:
                        empty_lines_count[pt] = old_empty[pt]
                        singleton_lines_count[pt] = old_singleton[pt]

    # Return N = F_q^2 \ best_S
    points = []
    for x in range(total_points):
        if not best_in_S[x]:
            yb = x % p
            ya = (x // p) % p
            xb = (x // (p*p)) % p
            xa = (x // (p*p*p)) % p
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
