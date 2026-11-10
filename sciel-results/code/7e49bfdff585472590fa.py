import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n = q * q  # 2401
    
    start_time = time.time()
    deadline = start_time + 570.0
    
    # Field operations in F_49 = F_7[alpha] where alpha^2 = 3
    # Represent elements as (a, b) meaning a + b*alpha, a,b in 0..6
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0*b0 + a0*b1*alpha + a1*b0*alpha + a1*b1*alpha^2
        # = a0*b0 + a1*b1*3 + (a0*b1 + a1*b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*3) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    def field_inv(a):
        # For a + b*alpha, inverse is (a - b*alpha) / (a^2 - 3*b^2)
        # = (a - b*alpha) / (a^2 - 3*b^2)
        denom = (a[0]*a[0] - 3*a[1]*a[1]) % p
        if denom == 0:
            return (0, 0)
        # Inverse of denom in F_7
        denom_inv = pow(denom, p-2, p)
        # (a - b*alpha) * denom_inv
        return (a[0]*denom_inv % p, (-a[1]*denom_inv) % p)
    
    # Precompute all field elements as tuples
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Map tuple to index
    elem_to_idx = {}
    for i, elem in enumerate(field_elements):
        elem_to_idx[elem] = i
    
    # Precompute addition and multiplication tables
    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    neg_table = np.zeros(q, dtype=np.int32)
    inv_table = np.zeros(q, dtype=np.int32)
    
    for i, a in enumerate(field_elements):
        neg_table[i] = elem_to_idx[field_neg(a)]
        for j, b in enumerate(field_elements):
            add_table[i, j] = elem_to_idx[field_add(a, b)]
            mul_table[i, j] = elem_to_idx[field_mul(a, b)]
    
    # Precompute inverses
    for i, a in enumerate(field_elements):
        if a == (0, 0):
            inv_table[i] = 0
        else:
            inv_table[i] = elem_to_idx[field_inv(a)]
    
    # Point index: i = x_idx * q + y_idx, where x_idx, y_idx in 0..48
    # Point (x, y) in F_49^2
    
    # Generate all lines
    # Direction (1, m) for m in F_49, and direction (0, 1)
    # For each direction, there are q parallel lines
    
    lines = []  # list of sets of point indices
    lines_through = [set() for _ in range(n)]  # lines_through[i] = set of line indices containing point i
    
    def add_line(pts):
        idx = len(lines)
        lines.append(set(pts))
        for pt in pts:
            lines_through[pt].add(idx)
    
    # Direction (0, 1): vertical lines x = c
    for c in range(q):
        pts = []
        for y in range(q):
            pts.append(c * q + y)
        add_line(pts)
    
    # Direction (1, m): y = m*x + c
    for m in range(q):
        for c in range(q):
            pts = []
            for x in range(q):
                # y = m*x + c
                y = add_table[mul_table[m, x], c]
                pts.append(x * q + y)
            add_line(pts)
    
    total_lines = len(lines)
    
    # Greedy construction of complement set C
    # We want to find a large set C such that for every point x, there exists a line L through x with |L ∩ C| ≤ 1
    # If x in C: need a line through x with exactly 1 point in C (itself)
    # If x not in C: need a line through x with 0 points in C
    
    # Track:
    # line_cnt[L] = |L ∩ C|
    # good_count[x] = number of lines L through x with line_cnt[L] ≤ 1
    
    C = np.zeros(n, dtype=bool)
    line_cnt = np.zeros(total_lines, dtype=np.int32)
    good_count = np.full(n, total_lines, dtype=np.int32)  # initially all lines have cnt=0
    
    def is_valid():
        return np.all(good_count > 0)
    
    # Greedy: try to add points to C in a random order
    # When adding x to C:
    #   For each line L through x:
    #     old_cnt = line_cnt[L]
    #     line_cnt[L] = old_cnt + 1
    #     If old_cnt == 1: L becomes bad for all y in L (was good with cnt=1, now cnt=2)
    #       For y in L: good_count[y] -= 1
    #     If old_cnt == 0: L stays good (cnt goes 0->1, still ≤1)
    
    # When removing x from C (revert):
    #   For each line L through x:
    #     new_cnt = line_cnt[L]
    #     line_cnt[L] = new_cnt - 1
    #     If new_cnt == 2: L becomes good for all y in L (was bad with cnt=2, now cnt=1)
    #       For y in L: good_count[y] += 1
    #     If new_cnt == 1: L stays good
    
    best_C = np.zeros(n, dtype=bool)
    best_size = 0
    
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 60:
            break
        
        C = np.zeros(n, dtype=bool)
        line_cnt = np.zeros(total_lines, dtype=np.int32)
        good_count = np.full(n, total_lines, dtype=np.int32)
        
        # Shuffle points
        candidates = np.random.permutation(n)
        
        for x in candidates:
            if time.time() > deadline - 60:
                break
            
            if C[x]:
                continue
            
            # Try adding x to C
            C[x] = True
            valid = True
            
            # Update lines through x
            for L in lines_through[x]:
                old_cnt = line_cnt[L]
                line_cnt[L] = old_cnt + 1
                
                if old_cnt == 1:
                    # L becomes bad for all points on L
                    for y in lines[L]:
                        good_count[y] -= 1
                        if good_count[y] == 0:
                            valid = False
                
                elif old_cnt == 0:
                    # L stays good (0->1, still ≤1)
                    pass
                
                else:
                    # old_cnt >= 2, L was already bad, stays bad
                    pass
            
            if valid:
                # Keep x in C
                if np.sum(C) > best_size:
                    best_size = np.sum(C)
                    best_C = C.copy()
            else:
                # Revert: remove x from C
                C[x] = False
                for L in lines_through[x]:
                    new_cnt = line_cnt[L]
                    line_cnt[L] = new_cnt - 1
                    
                    if new_cnt == 2:
                        # L becomes good for all points on L
                        for y in lines[L]:
                            good_count[y] += 1
                    
                    elif new_cnt == 1:
                        # L stays good
                        pass
    
    # N = complement of best_C
    N = []
    for i in range(n):
        if not best_C[i]:
            x_idx = i // q
            y_idx = i % q
            x = field_elements[x_idx]
            y = field_elements[y_idx]
            N.append([[x[0], x[1]], [y[0], y[1]]])
    
    if len(N) == 0:
        # Should not happen, but safety
        N.append([[0, 0], [0, 0]])
    
    result = np.array(N, dtype=np.int64)
    # Reduce mod 7
    result = result % 7
    
    return result
