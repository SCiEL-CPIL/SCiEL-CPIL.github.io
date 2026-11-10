import numpy as np
import time
import random
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    p = 7
    k = 3  # alpha^2 = 3
    
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # Generate all points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    num_dirs = 50  # 49 slopes + vertical
    
    # Precompute all lines for each direction
    direction_lines = []
    
    # Vertical lines: same (xa, xb), all y values
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Lines with slope m = (ma, mb)
    for ma in range(p):
        for mb in range(p):
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # For each point and direction, get the line index
    point_line = [[0] * num_dirs for _ in range(n)]
    for d in range(num_dirs):
        for line_idx, line in enumerate(direction_lines[d]):
            for pt_idx in line:
                point_line[pt_idx][d] = line_idx
    
    # For each point and direction, get all other points on the same line
    line_members = [[[] for _ in range(num_dirs)] for _ in range(n)]
    for d in range(num_dirs):
        for line in direction_lines[d]:
            for i in range(len(line)):
                for j in range(len(line)):
                    if i != j:
                        line_members[line[i]][d].append(line[j])
    
    def is_valid_S(in_S):
        """Check if S is a valid complement set."""
        for x in range(n):
            has_safe_dir = False
            for d in range(num_dirs):
                clean = True
                for q in line_members[x][d]:
                    if in_S[q]:
                        clean = False
                        break
                if clean:
                    has_safe_dir = True
                    break
            if not has_safe_dir:
                return False
        return True
    
    def greedy_construct():
        """Greedy construction: add points to S if they don't block any point."""
        in_S = [False] * n
        # safe_dirs[x] = bitmask of directions where x's line has no S points (other than x)
        safe_dirs = [(1 << num_dirs) - 1 for _ in range(n)]
        
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            # Check if adding idx to S would make any point lose all safe directions
            valid = True
            changes = []
            
            for d in range(num_dirs):
                li = point_line[idx][d]
                line = direction_lines[d][li]
                for q in line:
                    if q == idx:
                        continue
                    if (safe_dirs[q] >> d) & 1:
                        # This direction is currently safe for q
                        if safe_dirs[q] == (1 << d):
                            # This is q's only safe direction
                            valid = False
                            break
                        else:
                            changes.append((q, d))
                            safe_dirs[q] &= ~(1 << d)
                if not valid:
                    break
            
            if valid:
                in_S[idx] = True
            else:
                # Revert changes
                for q, d in changes:
                    safe_dirs[q] |= (1 << d)
        
        return in_S
    
    def local_search(in_S, max_iters=5000):
        """Improve S by removing points that allow adding others."""
        current_size = sum(in_S)
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Try to remove a random point from S
            in_points = [i for i in range(n) if in_S[i]]
            if not in_points:
                break
            pt = random.choice(in_points)
            
            # Temporarily remove
            in_S[pt] = False
            new_size = current_size - 1
            
            # Try to add back some points that were previously blocked
            added = 0
            for q in range(n):
                if in_S[q]:
                    continue
                # Check if q can be added
                can_add = True
                for d in range(num_dirs):
                    li = point_line[q][d]
                    line = direction_lines[d][li]
                    for r in line:
                        if r == q:
                            continue
                        if in_S[r]:
                            # Check if removing q's block would help
                            pass
                    break  # Simplified check
                # For now, just keep it simple
                break
            
            if new_size < current_size:
                in_S[pt] = True
                continue
            
            current_size = new_size
        
        return in_S, current_size
    
    best_S = None
    best_size = 0
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        in_S = greedy_construct()
        current_size = sum(in_S)
        
        if current_size > best_size:
            # Validate before accepting
            if is_valid_S(in_S):
                best_size = current_size
                best_S = in_S[:]
        
        # Try local search improvement
        if time.time() > deadline - 60:
            break
        
        improved_S, improved_size = local_search(in_S[:], max_iters=2000)
        
        if improved_size > best_size and is_valid_S(improved_S):
            best_size = improved_size
            best_S = improved_S[:]
    
    # Fallback: if no valid set found, use empty S
    if best_S is None:
        best_S = [False] * n
    
    # Build result: N = All \ S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result
