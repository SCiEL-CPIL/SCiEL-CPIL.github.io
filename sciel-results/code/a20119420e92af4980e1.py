"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import itertools
import math

# Precomputation
# cover[a][b] is a set of points c such that (a,b,c) is isosceles.
# Points are indices 0 to N*N-1.
# index = y * N + x

def _get_idx(x, y):
    return y * N + x

def _get_coords(idx):
    return (idx % N, idx // N)

# Precompute cover sets
# cover[i][j] -> set of k such that triangle(i, j, k) is isosceles
cover = [[set() for _ in range(N*N)] for _ in range(N*N)]

for i in range(N*N):
    xi, yi = _get_coords(i)
    for j in range(N*N):
        if i == j: continue
        xj, yj = _get_coords(j)
        d2_ij = (xi - xj)**2 + (yi - yj)**2
        for k in range(N*N):
            if k == i or k == j: continue
            xk, yk = _get_coords(k)
            d2_ik = (xi - xk)**2 + (yi - yk)**2
            d2_jk = (xj - xk)**2 + (yj - yk)**2
            if d2_ij == d2_ik or d2_ij == d2_jk or d2_ik == d2_jk:
                cover[i][j].add(k)
                # Symmetry: cover[j][i] is same set? 
                # Triangle(i,j,k) is iso. Triangle(j,i,k) is same triangle.
                # So cover[i][j] == cover[j][i]
                cover[j][i].add(k)

# Helper to check if a set of indices is a valid No-Iso set
def _is_no_iso(S):
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            a = S_list[i]
            b = S_list[j]
            # Check if any other point in S is in cover[a][b]
            for c in S_list:
                if c != a and c != b:
                    if c in cover[a][b]:
                        return False
    return True

# Helper to check if a set is Maximal
def _is_maximal(S):
    S_set = set(S)
    # For every point q not in S, there must exist a,b in S such that q in cover[a][b]
    for q in range(N*N):
        if q in S_set:
            continue
        blocked = False
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if q in cover[S_list[i]][S_list[j]]:
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            return False
    return True

def _construct_greedy():
    """Construct a maximal no-isosceles set using greedy."""
    S = set()
    # Random order
    order = list(range(N*N))
    random.shuffle(order)
    
    for p in order:
        # Check if p can be added to S without creating an isosceles triangle
        ok = True
        S_list = list(S)
        for a in S_list:
            if p in cover[p][a]:
                # This means there exists c in S such that (p,a,c) is iso?
                # No, cover[p][a] contains points c such that (p,a,c) is iso.
                # If p is in cover[p][a], that's a self-reference error in my logic.
                # cover[p][a] contains c. We are adding p.
                # We need to check if adding p creates an iso with existing pairs.
                # Pair (a, b) in S. If p in cover[a][b], then (a,b,p) is iso.
                # So check: for all a,b in S, p not in cover[a][b].
                pass
        
        # Correct check:
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if p in cover[S_list[i]][S_list[j]]:
                    ok = False
                    break
            if not ok:
                break
        
        if ok:
            S.add(p)
            
    # Now S is No-Iso. Make it Maximal by adding points?
    # Greedy addition doesn't guarantee maximality in the sense of "cannot add any more".
    # Actually, the greedy loop above adds p if it doesn't break No-Iso.
    # If we process ALL points, the resulting S is a Maximal No-Iso set.
    # Why? Because if there was a point q not in S that could be added, it would have been considered.
    # When it was considered, if it could be added (no iso with existing), it would have been added.
    # If it couldn't be added, it is blocked.
    # So the final S is Maximal.
    
    return S

def _calc_penalty(S, blocked_by):
    """
    S: set of indices
    blocked_by: dict or list of counts for each q not in S
    Returns: size(S) + 10 * holes + 100 * isos
    """
    # Count Holes
    holes = 0
    for q in range(N*N):
        if q not in S and blocked_by[q] == 0:
            holes += 1
            
    # Count Isos (should be 0 if we only make valid moves, but for SA we might allow invalid)
    # For efficiency, we assume moves maintain No-Iso or we check separately.
    # Let's just use holes and size.
    return len(S) + 10 * holes

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # 1. Get initial solution
    S = _construct_greedy()
    best_S = set(S)
    best_size = len(S)
    
    # Precompute initial blocked_by
    blocked_by = [0] * (N*N)
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            a = S_list[i]
            b = S_list[j]
            for q in cover[a][b]:
                if q not in S:
                    blocked_by[q] += 1
    
    current_penalty = _calc_penalty(S, blocked_by)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Choose a move: Add, Remove, or Swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_S = set(S)
        new_blocked = blocked_by[:] # Copy
        
        if move_type == 'add':
            candidates = [q for q in range(N*N) if q not in S]
            if not candidates: continue
            p = random.choice(candidates)
            
            # Check if p can be added (No-Iso constraint)
            valid = True
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    if p in cover[S_list[i]][S_list[j]]:
                        valid = False
                        break
                if not valid: break
            
            if not valid:
                # In SA, we might still accept if penalty decreases? 
                # No, adding a point that creates an iso is bad.
                # Let's skip invalid moves to keep search focused.
                continue
                
            new_S.add(p)
            
            # Update blocked_by
            for a in S:
                for q in cover[p][a]:
                    if q not in new_S: # q != p
                        new_blocked[q] += 1
                        # Note: if q was in S, it's not "blocked" in the sense of being outside, 
                        # but it's part of the set. The definition of maximal only cares about q not in S.
                        # However, if q is in S, it doesn't matter if it's covered.
            
        elif move_type == 'remove':
            if not S: continue
            p = random.choice(list(S))
            new_S.remove(p)
            
            # Update blocked_by
            for a in S:
                if a == p: continue
                for q in cover[p][a]:
                    if q not in new_S:
                        new_blocked[q] -= 1
                        if new_blocked[q] < 0: new_blocked[q] = 0 # Safety

        else: # swap
            if not S: continue
            p = random.choice(list(S))
            candidates = [q for q in range(N*N) if q not in S]
            if not candidates: continue
            q_new = random.choice(candidates)
            
            # Check if q_new can replace p
            # 1. q_new must not form iso with S \ {p}
            valid = True
            S_minus_p = list(S)
            S_minus_p.remove(p)
            for i in range(len(S_minus_p)):
                for j in range(i+1, len(S_minus_p)):
                    if q_new in cover[S_minus_p[i]][S_minus_p[j]]:
                        valid = False
                        break
                if not valid: break
            
            # 2. Also check q_new with single elements? No, iso requires 3 points.
            # Wait, if S_minus_p has only 1 point, no iso possible.
            # If S_minus_p has 0 points, no iso possible.
            
            if not valid:
                continue
                
            new_S.remove(p)
            new_S.add(q_new)
            
            # Update blocked_by
            # Remove contributions from pairs involving p
            for a in S:
                if a == p: continue
                for q in cover[p][a]:
                    if q not in new_S:
                        new_blocked[q] -= 1
                        if new_blocked[q] < 0: new_blocked[q] = 0
            
            # Add contributions from pairs involving q_new
            for a in S_minus_p:
                for q in cover[q_new][a]:
                    if q not in new_S:
                        new_blocked[q] += 1
        
        new_penalty = _calc_penalty(new_S, new_blocked)
        
        # Acceptance criterion
        delta = new_penalty - current_penalty
        if delta < 0 or random.random() < math.exp(-delta / T):
            S = new_S
            blocked_by = new_blocked
            current_penalty = new_penalty
            
            if len(S) < best_size:
                best_S = set(S)
                best_size = len(S)
        
        T *= cooling
        
    return list(best_S)
# EVOLVE-BLOCK-END
