# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using integer field representation
    and a priority-based greedy algorithm for maximizing the complement.
    """
    p = 7
    q = p * p  # 49
    deadline = time.time() + 570  # Safety margin
    
    # Field F_49 = F_7[alpha] / (alpha^2 - 3)
    # Represent element a + b*alpha as integer a + b*7 (so 0..48)
    k_const = 3
    
    # Precompute addition and multiplication tables
    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    
    for a in range(q):
        a0, a1 = a % p, a // p
        for b in range(q):
            b0, b1 = b % p, b // p
            s0 = (a0 + b0) % p
            s1 = (a1 + b1) % p
            add_table[a, b] = s0 + s1 * p
            
            m0 = (a0 * b0 + a1 * b1 * k_const) % p
            m1 = (a0 * b1 + a1 * b0) % p
            mul_table[a, b] = m0 + m1 * p
    
    n_points = q * q  # 2401
    
    # Directions: v = (1, m) for m in F_49 (49 dirs) + v = (0, 1) (1 dir) = 50 dirs
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    n_dirs = len(directions)
    
    # Precompute line_others[di][p] = array of q-1 point indices on line through p with direction di
    line_others = np.zeros((n_dirs, n_points, q - 1), dtype=np.int32)
    for di, (vx, vy) in enumerate(directions):
        for px in range(q):
            for py in range(q):
                p_idx = px + py * q
                ti = 0
                for t in range(q):
                    if t == 0:
                        continue
                    tx = int(mul_table[t, vx])
                    ty = int(mul_table[t, vy])
                    nx = int(add_table[px, tx])
                    ny = int(add_table[py, ty])
                    line_others[di, p_idx, ti] = nx + ny * q
                    ti += 1
    
    # Start with C = a conic or unital to have a valid starting complement
    # Use: points (x, y) where y = x^3 (cubic curve) - has 49 points
    C = np.zeros(n_points, dtype=bool)
    for x in range(q):
        y = int(mul_table[x, x])  # x^2
        y = int(mul_table[y, x])  # x^3
        C[x + y * q] = True
    
    def can_add(p, C):
        for di in range(n_dirs):
            found = False
            others = line_others[di, p]
            for idx in others:
                if C[idx]:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Greedy: multiple passes
    changed = True
    while changed:
        changed = False
        for p in range(n_points):
            if C[p]:
                continue
            if time.time() > deadline:
                break
            if can_add(p, C):
                C[p] = True
                changed = True
        if time.time() > deadline:
            break
    
    # Try random restarts to improve
    remaining_time = deadline - time.time()
    if remaining_time > 60:
        best_C = C.copy()
        best_count = C.sum()
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            C_trial = best_C.copy()
            order = np.random.permutation(n_points)
            changed = True
            while changed:
                changed = False
                for p in order:
                    if C_trial[p]:
                        continue
                    if time.time() > deadline - 10:
                        break
                    if can_add(p, C_trial):
                        C_trial[p] = True
                        changed = True
                if time.time() > deadline - 10:
                    break
            if C_trial.sum() > best_count:
                best_count = C_trial.sum()
                best_C = C_trial.copy()
        C = best_C
    
    # Build N = all points not in C
    N_indices = np.where(~C)[0]
    
    k = len(N_indices)
    points = np.zeros((k, 2, 2), dtype=np.int64)
    for i, p in enumerate(N_indices):
        px = p % q
        py = p // q
        points[i, 0, 0] = px % p
        points[i, 0, 1] = px // p
        points[i, 1, 0] = py % p
        points[i, 1, 1] = py // p
    
    return points

# EVOLVE-BLOCK-END
