import numpy as np
import time
import random

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 620.0
_MARGIN = 5.0

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
        if denom == 0:
            # Should not happen for F_7[alpha] with alpha^2=3 since 3 is not a square in F_7
            # 0^2=0, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1
            # Squares: 0,1,2,4. 3 is not a square. So a^2 - 3b^2 = 0 implies a=b=0.
            inv_table[i] = 0
        else:
            denom_inv = pow(denom, 5, 7)
            ia = (a * denom_inv) % p
            ib = (-b * denom_inv) % p
            inv_table[i] = ia * p + ib
    inv_table[0] = 0
    
    # Build lines
    # Line ID: m*49 + c for non-vertical, 2401 + x0 for vertical
    # Total lines: 49*49 + 49 = 2450
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
    
    def init_C_parabola():
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
    
    def compute_W(in_C):
        W = [set() for _ in range(q2)]
        for i in range(q2):
            for lid in point_lines[i]:
                lp = line_points_list[lid]
                valid = True
                for q_idx in lp:
                    if q_idx != i and in_C[q_idx]:
                        valid = False
                        break
                if valid:
                    W[i].add(lid)
        return W
    
    def try_greedy_add(in_C, W, rng, max_steps=50000):
        in_C = in_C.copy()
        W = [set(s) for s in W]
        wcount = np.array([len(s) for s in W], dtype=np.int32)
        
        # Precompute for each point p, the list of (x, lid) such that lid in W[x] and p in line_points[lid] and p != x
        # This is expensive, so we compute on the fly but optimize the inner loop.
        
        # For faster access: for each line lid, store the set of points x for which lid is in W[x]
        # witness_lines_for_x: W[x] is the set of lines.
        # Let's invert: for each line lid, which x have lid in W[x]?
        # Actually, W[x] contains lid iff lid is a witness for x.
        # Let's create a mapping: lid -> list of x such that lid in W[x].
        lid_to_witnesses = [[] for _ in range(2450)]
        for x in range(q2):
            for lid in W[x]:
                lid_to_witnesses[lid].append(x)
        
        steps = 0
        while steps < max_steps and time.time() < _DEADLINE - _MARGIN:
            steps += 1
            
            # Find the point p that minimizes damage
            # Damage of adding p: number of x such that p is on a witness line of x, and wcount[x] == 1
            # If we add p, then for any x where p is on a witness line L of x, L is no longer a witness for x.
            # So wcount[x] decreases by 1 for each such L.
            # If wcount[x] was 1, it becomes 0, which is invalid.
            
            # To speed up, we can maintain a list of "critical" points: those with wcount[x] == 1.
            # For each critical x, we know its unique witness line L_x.
            # Adding p is safe iff p is not on L_x for any critical x (where x != p, and p is not already in C).
            # Wait, if p is added to C, then for any x, if p is on a witness line L of x, L is no longer a witness.
            # So if x has wcount[x] == 1, and its unique witness line L contains p (p != x), then x becomes invalid.
            # So p can be added iff for all x with wcount[x] == 1, p is not on the unique witness line of x (unless x == p, but p is not in C yet).
            
            # Let's maintain a set of "blocked" points: points p that cannot be added because they lie on the unique witness line of some critical x.
            # Initially, blocked = union of (line_points[L_x] \ {x}) for all x with wcount[x] == 1.
            
            # For simplicity and speed, let's just scan all points not in C and compute damage.
            # Optimize: precompute for each point p, the set of lines through p.
            # For each p, damage = sum over lines L through p of (number of x with wcount[x]==1 and L in W[x] and x != p).
            
            # Let's maintain a count: for each line L, how many critical x have L as their unique witness?
            # critical_count_on_line[L] = number of x with wcount[x]==1 and L in W[x].
            # Then damage(p) = sum over L in point_lines[p] of critical_count_on_line[L].
            # But we must exclude x == p. If p is critical and its unique witness is L, and L is in point_lines[p], then we count it, but x=p is not affected by adding p (since p is added to C, and the condition for x=p is about lines through p avoiding C \ {p}).
            # Wait, if p is added to C, then for x=p, the witness line L must avoid C \ {p}. Since p is in C, L \ {p} must avoid C.
            # So if L was a witness for p, and we add p to C, L is still a witness for p if L \ {p} avoids C.
            # But we are adding p to C, so L \ {p} must avoid the new C. Since p was not in C before, L \ {p} avoided the old C. The new C is old C + {p}. So L \ {p} still avoids the new C.
            # So adding p to C does not affect the witness status of p itself.
            # So damage(p) = sum over L in point_lines[p] of (number of critical x != p with L in W[x]).
            
            # Let's maintain critical_count_on_line[L] as the number of critical x with L in W[x].
            # And for each L, the set of critical x with L in W[x].
            
            # For now, let's just compute damage on the fly but efficiently.
            
            best_p = -1
            best_damage = float('inf')
            
            # Sample a subset of candidates for speed
            candidates = [i for i in range(q2) if not in_C[i]]
            sample_size = min(len(candidates), 500)
            if sample_size < len(candidates):
                sample = rng.sample(candidates, sample_size)
            else:
                sample = candidates
            
            for p_idx in sample:
                if in_C[p_idx]:
                    continue
                damage = 0
                for lid in point_lines[p_idx]:
                    for x_idx in lid_to_witnesses[lid]:
                        if x_idx == p_idx:
                            continue
                        if wcount[x_idx] == 1:
                            damage += 1
                if damage < best_damage:
                    best_damage = damage
                    best_p = p_idx
            
            if best_p == -1 or best_damage > 0:
                break
            
            in_C[best_p] = True
            if best_p in candidates:
                candidates.remove(best_p)
            
            # Update W and wcount
            for lid in point_lines[best_p]:
                for x_idx in line_points_list[lid]:
                    if x_idx == best_p:
                        continue
                    if lid in W[x_idx]:
                        W[x_idx].discard(lid)
                        wcount[x_idx] -= 1
                        if wcount[x_idx] == 0:
                            # This should not happen if best_damage was 0
                            pass
                        # Update lid_to_witnesses
                        if x_idx in lid_to_witnesses[lid]:
                            lid_to_witnesses[lid].remove(x_idx)
        
        return in_C, W, wcount
    
    best_in_C = init_C_parabola()
    best_size = int(np.sum(best_in_C))
    
    rng = random.Random(42)
    
    for trial in range(200):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        in_C = init_C_parabola()
        W = compute_W(in_C)
        wcount = np.array([len(s) for s in W], dtype=np.int32)
        
        in_C, W, wcount = try_greedy_add(in_C, W, rng, max_steps=50000)
        
        cur_size = int(np.sum(in_C))
        if cur_size > best_size:
            best_size = cur_size
            best_in_C = in_C.copy()
    
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
