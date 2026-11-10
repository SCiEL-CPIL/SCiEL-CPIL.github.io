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
    
    # Build lines
    # Line ID: m*49 + c for non-vertical, 2401 + x0 for vertical
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
    
    # Convert line_points to lists for faster iteration
    line_points_list = [list(s) for s in line_points]

    def calculate_cost(p_idx, W):
        """Calculate the number of witness lines that p_idx would invalidate."""
        cost = 0
        for lid in point_lines[p_idx]:
            for x_idx in line_points_list[lid]:
                if x_idx != p_idx:
                    if lid in W[x_idx]:
                        cost += 1
        return cost

    def is_safe(p_idx, W):
        """Check if adding p_idx to C leaves all points with at least one witness."""
        for lid in point_lines[p_idx]:
            for x_idx in line_points_list[lid]:
                if x_idx != p_idx:
                    if lid in W[x_idx]:
                        # If this is the only witness for x_idx, it's unsafe
                        if len(W[x_idx]) == 1:
                            return False
        return True

    def add_point_to_C(p_idx, in_C, W):
        """Add p_idx to C and update W."""
        in_C[p_idx] = True
        for lid in point_lines[p_idx]:
            for x_idx in line_points_list[lid]:
                if x_idx != p_idx:
                    if lid in W[x_idx]:
                        W[x_idx].discard(lid)

    best_in_C = None
    best_size = -1
    
    rng = random.Random(42)
    
    # Try multiple restarts with different initial sets or just one good run?
    # The parabola is a good seed.
    
    for trial in range(50):
        if time.time() > _DEADLINE - _MARGIN:
            break
            
        # Initialize C with Parabola
        in_C = np.zeros(q2, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                ya = (xa * xa + 3 * xb * xb) % p
                yb = (2 * xa * xb) % p
                x_idx = xa * p + xb
                y_idx = ya * p + yb
                pi = x_idx * q + y_idx
                in_C[pi] = True
                
        # Initialize W
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
                    
        # Build Heap
        heap = []
        for i in range(q2):
            if not in_C[i]:
                c = calculate_cost(i, W)
                heapq.heappush(heap, (c, i))
                
        # Greedy Expansion with Priority Queue
        while heap:
            if time.time() > _DEADLINE - _MARGIN:
                break
                
            cost, p_idx = heapq.heappop(heap)
            
            if in_C[p_idx]:
                continue
                
            # Verify cost is still accurate (Lazy Update)
            current_cost = calculate_cost(p_idx, W)
            if current_cost != cost:
                heapq.heappush(heap, (current_cost, p_idx))
                continue
                
            # Check safety
            if is_safe(p_idx, W):
                add_point_to_C(p_idx, in_C, W)
                
                # Update heap for affected points?
                # We don't strictly need to update all, lazy verification handles it.
                # But we can push back neighbors to speed up convergence.
                # Points on lines with p_idx are affected.
                for lid in point_lines[p_idx]:
                    for x_idx in line_points_list[lid]:
                        if x_idx != p_idx and not in_C[x_idx]:
                            # Only push if it's a candidate
                            # We can just push it, lazy check will filter
                            new_c = calculate_cost(x_idx, W)
                            heapq.heappush(heap, (new_c, x_idx))
            
        current_size = np.sum(in_C)
        if current_size > best_size:
            best_size = current_size
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
