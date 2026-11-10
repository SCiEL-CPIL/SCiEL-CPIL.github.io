import numpy as np
import time
import random
import heapq

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 10.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    
    # Precompute field operations
    mul_table = np.zeros((q, q), dtype=np.int32)
    add_table = np.zeros((q, q), dtype=np.int32)
    sub_table = np.zeros((q, q), dtype=np.int32)
    inv_table = np.zeros(q, dtype=np.int32)
    
    for i in range(q):
        for j in range(q):
            a1, b1 = i // p, i % p
            a2, b2 = j // p, j % p
            ra = (a1 * a2 + 3 * b1 * b2) % p
            rb = (a1 * b2 + a2 * b1) % p
            mul_table[i, j] = ra * p + rb
            
            ra = (a1 + a2) % p
            rb = (b1 + b2) % p
            add_table[i, j] = ra * p + rb
            
            ra = (a1 - a2) % p
            rb = (b1 - b2) % p
            sub_table[i, j] = ra * p + rb
    
    for i in range(1, q):
        a, b = i // p, i % p
        denom = (a * a - 3 * b * b) % p
        denom_inv = pow(denom, 5, 7)
        ia = (a * denom_inv) % p
        ib = (-b * denom_inv) % p
        inv_table[i] = ia * p + ib
    inv_table[0] = 0
    
    # Build all lines
    # Non-vertical: m in 0..48, c in 0..48 -> line_id = m*49 + c (total 2401)
    # Vertical: x0 in 0..48 -> line_id = 2401 + x0 (total 49)
    # Total: 2450 lines
    point_lines = [[] for _ in range(q2)]
    line_points = [set() for _ in range(2450)]
    
    for xi in range(q):
        for yi in range(q):
            pi = xi * q + yi
            vl = 2401 + xi
            point_lines[pi].append(vl)
            line_points[vl].add(pi)
            
            for m in range(q):
                mc = sub_table[yi, mul_table[m, xi]]
                nl = m * q + mc
                point_lines[pi].append(nl)
                line_points[nl].add(pi)
    
    line_points_list = [list(s) for s in line_points]
    # Convert line_points_list to numpy arrays for faster summing if needed, 
    # but we will use a specific "line occupancy" vector approach.
    
    def init_empty():
        return np.zeros(q2, dtype=bool)
        
    def init_parabola():
        in_C = np.zeros(q2, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                ya = (xa * xa + 3 * xb * xb) % p
                yb = (2 * xa * xb) % p
                x_idx = xa * p + xb
                y_idx = ya * p + yb
                pi = x_idx * q + y_idx
                in_C[pi] = True
        return in_C

    def init_random(density=0.1):
        in_C = np.zeros(q2, dtype=bool)
        n_points = int(q2 * density)
        indices = random.sample(range(q2), n_points)
        in_C[indices] = True
        return in_C

    def compute_line_occupancy(in_C):
        """
        Returns an array `line_occ` of length 2450, where line_occ[l] is the number of points in C on line l.
        """
        line_occ = np.zeros(2450, dtype=np.int32)
        for i in range(q2):
            if in_C[i]:
                for lid in point_lines[i]:
                    line_occ[lid] += 1
        return line_occ

    def get_valid_points(in_C, line_occ):
        """
        Returns a boolean array `valid` where valid[i] is True if point i has at least one witness line.
        A line L is a witness for i if line_occ[L] == 1 and i is the point on L in C, 
        OR line_occ[L] == 0.
        Wait, if i is NOT in C, witness L requires line_occ[L] == 0.
        If i IS in C, witness L requires line_occ[L] == 1 (and it's i).
        
        Actually, the condition is:
        For point x, there exists a line L containing x such that (L \ {x}) is disjoint from C.
        This is equivalent to:
        If x in C: sum(C on L) <= 1 (and if 1, it is x). Since x in C, sum(C on L) must be 1.
        If x not in C: sum(C on L) == 0.
        
        So:
        valid[i] = True if:
          (in_C[i] and any(line_occ[l] == 1 for l in point_lines[i])) 
          or 
          (not in_C[i] and any(line_occ[l] == 0 for l in point_lines[i]))
        """
        valid = np.zeros(q2, dtype=bool)
        for i in range(q2):
            if in_C[i]:
                # Need a line with occ 1
                for lid in point_lines[i]:
                    if line_occ[lid] == 1:
                        valid[i] = True
                        break
            else:
                # Need a line with occ 0
                for lid in point_lines[i]:
                    if line_occ[lid] == 0:
                        valid[i] = True
                        break
        return valid

    def is_valid(in_C):
        line_occ = compute_line_occupancy(in_C)
        valid = get_valid_points(in_C, line_occ)
        return np.all(valid)

    def greedy_repair(in_C, rng):
        """
        Adds points to in_C if they don't break validity.
        Returns new in_C, line_occ, valid.
        """
        line_occ = compute_line_occupancy(in_C)
        valid = get_valid_points(in_C, line_occ)
        
        if not np.all(valid):
            return in_C, line_occ, valid

        candidates = [i for i in range(q2) if not in_C[i]]
        
        while candidates and time.time() < _DEADLINE - _MARGIN:
            # Find a safe candidate
            # Damage calculation:
            # Adding p to C changes line_occ for lines containing p.
            # For each line L containing p, occ[L] increases by 1.
            # This might invalidate points on L.
            
            best_p = -1
            best_damage = float('inf')
            
            # Sample candidates to save time
            sample_size = min(len(candidates), 50)
            sample = rng.sample(candidates, sample_size)
            
            for p_idx in sample:
                if in_C[p_idx]:
                    continue
                
                damage = 0
                # Check validity impact
                # We need to check if adding p_idx keeps all points valid.
                # Instead of full re-eval, we can check "damage".
                # A point x becomes invalid if ALL its lines are no longer witnesses.
                
                # Simulate line_occ update
                new_line_occ = line_occ.copy()
                for lid in point_lines[p_idx]:
                    new_line_occ[lid] += 1
                
                # Check if p_idx itself is valid
                # p_idx is in C now. Needs a line with occ 1.
                p_idx_valid = False
                for lid in point_lines[p_idx]:
                    if new_line_occ[lid] == 1:
                        p_idx_valid = True
                        break
                if not p_idx_valid:
                    continue # Adding p_idx makes p_idx invalid? 
                             # Wait, if p_idx is added, it contributes to occ.
                             # If all lines through p_idx have occ > 1, then p_idx has no witness.
                             # So we skip.
                
                # Check other points
                for lid in point_lines[p_idx]:
                    for x_idx in line_points_list[lid]:
                        if x_idx == p_idx:
                            continue
                        # Check if x_idx is still valid with new_line_occ
                        # x_idx valid if:
                        # (in_C[x_idx] and any(new_line_occ[l] == 1 for l in point_lines[x_idx]))
                        # or
                        # (not in_C[x_idx] and any(new_line_occ[l] == 0 for l in point_lines[x_idx]))
                        
                        x_valid = False
                        if in_C[x_idx]:
                            for l2 in point_lines[x_idx]:
                                if new_line_occ[l2] == 1:
                                    x_valid = True
                                    break
                        else:
                            for l2 in point_lines[x_idx]:
                                if new_line_occ[l2] == 0:
                                    x_valid = True
                                    break
                            
                        if not x_valid:
                            damage += 1
                            if damage > best_damage:
                                break
                    if damage > best_damage:
                        break
                
                if damage < best_damage:
                    best_damage = damage
                    best_p = p_idx
                
                if best_damage == 0:
                    break

            if best_p == -1 or best_damage > 0:
                break
            
            # Add best_p
            in_C[best_p] = True
            candidates.remove(best_p)
            
            for lid in point_lines[best_p]:
                line_occ[lid] += 1
            
            # Update valid? We can recompute at the end or track.
            # For simplicity, recompute valid periodically or just rely on damage==0 logic.
            # Actually, if damage==0, all points remain valid.
        
        # Final validation
        line_occ = compute_line_occupancy(in_C)
        valid = get_valid_points(in_C, line_occ)
        
        return in_C, line_occ, valid

    best_in_C = None
    best_size = 0
    
    rng = random.Random(42)
    
    # ILS Loop
    for trial in range(200):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        rng.seed(42 + trial)
        
        # Initialize
        if trial % 3 == 0:
            in_C = init_empty()
        elif trial % 3 == 1:
            in_C = init_parabola()
        else:
            in_C = init_random(0.05)
        
        # Repair (Greedy Add)
        in_C, _, valid = greedy_repair(in_C, rng)
        
        if not np.all(valid):
            continue
            
        cur_size = int(np.sum(in_C))
        
        if cur_size > best_size:
            best_size = cur_size
            best_in_C = in_C.copy()
            
        # Perturb: Remove random points from best_in_C (or current in_C)
        # Let's perturb the current in_C
        if cur_size > 0:
            # Remove 5% of points
            n_remove = max(1, int(cur_size * 0.05))
            c_indices = np.where(in_C)[0]
            if len(c_indices) >= n_remove:
                to_remove = rng.sample(list(c_indices), n_remove)
                in_C[to_remove] = False
                # Re-repair
                in_C, _, valid = greedy_repair(in_C, rng)
                if np.all(valid):
                    new_size = int(np.sum(in_C))
                    if new_size > best_size:
                        best_size = new_size
                        best_in_C = in_C.copy()

    # If no valid solution found, fall back to empty set (N = all points)
    if best_in_C is None:
        best_in_C = init_empty()
    
    # Construct N = complement of C
    in_N = ~best_in_C
    
    N = []
    for i in range(q2):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
    
    return np.array(N, dtype=np.int64)
