# EVOLVE-BLOCK-START
import numpy as np
import time

_P = 7
_Q = 49
_Q2 = 2401
_T0 = time.time()
_DEADLINE = _T0 + 600.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - time.time() - _MARGIN


def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    k = 3  # alpha^2 = 3

    # Field operations using precomputed tables
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for x in range(q):
        xa, xb = x // p, x % p
        for y in range(q):
            ya, yb = y // p, y % p
            add_table[x, y] = ((xa + ya) % p) * p + ((xb + yb) % p)
            mul_table[x, y] = ((xa * ya + xb * yb * k) % p) * p + ((xa * yb + xb * ya) % p)

    def pt_idx(x, y):
        return x * q + y

    def pt_xy(idx):
        return idx // q, idx % q

    # Build all lines
    line_points = []

    # Vertical lines: 49 lines
    for x in range(q):
        pts = np.array([pt_idx(x, y) for y in range(q)], dtype=np.int64)
        line_points.append(pts)

    # Slanted lines: 49*49 = 2401 lines
    for m in range(q):
        for c in range(q):
            pts = np.array([pt_idx(t, int(add_table[mul_table[m, t], c])) for t in range(q)], dtype=np.int64)
            line_points.append(pts)

    n_lines = len(line_points)  # 2450

    # For each point, list of lines containing it
    point_lines = [[] for _ in range(q2)]
    for li in range(n_lines):
        for pi in line_points[li]:
            point_lines[pi].append(li)

    best_S = np.zeros(q2, dtype=bool)
    best_size = 0

    rng_master = np.random.default_rng(42)
    
    max_restarts = 200
    
    for r in range(max_restarts):
        if _time_left() < 10.0:
            break
            
        S = np.zeros(q2, dtype=bool)
        line_S_count = np.zeros(n_lines, dtype=np.int64)
        good_line_count = np.full(q2, 50, dtype=np.int64)

        def try_add(z):
            if S[z]:
                return False
            
            valid = True
            for li in point_lines[z]:
                old_count = line_S_count[li]
                for w in line_points[li]:
                    if w == z:
                        continue
                    if S[w]:
                        if old_count == 1:
                            if good_line_count[w] == 1:
                                valid = False
                                break
                    else:
                        if old_count == 0:
                            if good_line_count[w] == 1:
                                valid = False
                                break
                if not valid:
                    break
            
            if not valid:
                return False
            
            S[z] = True
            for li in point_lines[z]:
                old_count = line_S_count[li]
                new_count = old_count + 1
                line_S_count[li] = new_count
                for w in line_points[li]:
                    if w == z:
                        continue
                    if S[w]:
                        if old_count == 1:
                            good_line_count[w] -= 1
                    else:
                        if old_count == 0:
                            good_line_count[w] -= 1
            
            good_line_count[z] = 50
            return True

        def try_swap(u, v):
            if not S[u] or S[v]:
                return False
            
            # Remove u
            S[u] = False
            for li in point_lines[u]:
                old_count = line_S_count[li]
                new_count = old_count - 1
                line_S_count[li] = new_count
                for w in line_points[li]:
                    if w == u:
                        continue
                    if S[w]:
                        if old_count == 1:
                            good_line_count[w] -= 1
                        elif new_count == 1:
                            good_line_count[w] += 1
                    else:
                        if old_count == 0:
                            good_line_count[w] -= 1
                        elif new_count == 0:
                            good_line_count[w] += 1
            u_good = 0
            for li in point_lines[u]:
                if line_S_count[li] == 0:
                    u_good += 1
            good_line_count[u] = u_good

            # Add v
            S[v] = True
            for li in point_lines[v]:
                old_count = line_S_count[li]
                new_count = old_count + 1
                line_S_count[li] = new_count
                for w in line_points[li]:
                    if w == v:
                        continue
                    if S[w]:
                        if old_count == 1:
                            good_line_count[w] -= 1
                        elif new_count == 1:
                            good_line_count[w] += 1
                    else:
                        if old_count == 0:
                            good_line_count[w] -= 1
                        elif new_count == 0:
                            good_line_count[w] += 1
            v_good = 0
            for li in point_lines[v]:
                if line_S_count[li] == 1:
                    v_good += 1
            good_line_count[v] = v_good

            if np.all(good_line_count >= 1):
                return True
            else:
                # Revert Add v
                S[v] = False
                for li in point_lines[v]:
                    old_count = line_S_count[li]
                    new_count = old_count - 1
                    line_S_count[li] = new_count
                    for w in line_points[li]:
                        if w == v:
                            continue
                        if S[w]:
                            if new_count == 1:
                                good_line_count[w] += 1
                            elif old_count == 1:
                                good_line_count[w] -= 1
                        else:
                            if new_count == 0:
                                good_line_count[w] += 1
                            elif old_count == 0:
                                good_line_count[w] -= 1
                v_good = 0
                for li in point_lines[v]:
                    if line_S_count[li] == 0:
                        v_good += 1
                good_line_count[v] = v_good

                # Revert Remove u
                S[u] = True
                for li in point_lines[u]:
                    old_count = line_S_count[li]
                    new_count = old_count + 1
                    line_S_count[li] = new_count
                    for w in line_points[li]:
                        if w == u:
                            continue
                        if S[w]:
                            if old_count == 1:
                                good_line_count[w] -= 1
                            elif new_count == 1:
                                good_line_count[w] += 1
                        else:
                            if old_count == 0:
                                good_line_count[w] -= 1
                            elif new_count == 0:
                                good_line_count[w] += 1
                u_good = 0
                for li in point_lines[u]:
                    if line_S_count[li] == 1:
                        u_good += 1
                good_line_count[u] = u_good
                return False

        # Greedy Phase
        rng = np.random.default_rng(r)
        order = np.arange(q2)
        rng.shuffle(order)
        
        for z in order:
            if _time_left() < 5.0:
                break
            if not S[z]:
                if try_add(z):
                    pass
        
        cur_size = int(np.sum(S))
        if cur_size > best_size:
            best_size = cur_size
            best_S = S.copy()
        
        # Local Search
        if _time_left() < 10.0:
            continue
            
        s_in = np.where(S)[0]
        s_not = np.where(~S)[0]
        
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        max_iter = 50000
        rng_sa = np.random.default_rng(r + 1000)
        
        for it in range(max_iter):
            if _time_left() < 5.0:
                break
            if T < min_T:
                break
            
            if len(s_in) == 0 or len(s_not) == 0:
                break
                
            u = s_in[rng_sa.integers(0, len(s_in))]
            v = s_not[rng_sa.integers(0, len(s_not))]
            
            if u == v:
                continue
                
            if try_swap(u, v):
                s_in = s_in[s_in != u]
                s_in = np.append(s_in, v)
                s_not = s_not[s_not != v]
                s_not = np.append(s_not, u)
                
                cur_size = int(np.sum(S))
                if cur_size > best_size:
                    best_size = cur_size
                    best_S = S.copy()
            
            T *= cooling

    # Build output
    pts_list = []
    for z in range(q2):
        if not best_S[z]:
            x, y = pt_xy(z)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
