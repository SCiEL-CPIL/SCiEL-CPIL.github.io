import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    DEADLINE = 600
    p = 7
    q = 49
    k = 3

    # Precompute field tables
    add_tbl = np.zeros((q, q), dtype=np.int64)
    sub_tbl = np.zeros((q, q), dtype=np.int64)
    mul_tbl = np.zeros((q, q), dtype=np.int64)

    for i in range(q):
        a0, a1 = i // p, i % p
        for j in range(q):
            b0, b1 = j // p, j % p
            add_tbl[i, j] = ((a0 + b0) % p) * p + ((a1 + b1) % p)
            sub_tbl[i, j] = ((a0 - b0) % p) * p + ((a1 - b1) % p)
            r0 = (a0 * b0 + a1 * b1 * k) % p
            r1 = (a0 * b1 + a1 * b0) % p
            mul_tbl[i, j] = r0 * p + r1

    total_points = q * q

    # Construct Unital U
    U = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        x_idx = xa * p + xb
                        y_idx = ya * p + yb
                        U.add(x_idx * q + y_idx)

    all_points_set = set(range(total_points))
    outside_U = all_points_set - U

    # Build all lines
    lines = []
    for c in range(q):
        pts = [c * q + y for y in range(q)]
        lines.append(pts)

    for m in range(q):
        for b in range(q):
            pts = []
            for x in range(q):
                y = add_tbl[mul_tbl[m, x], b]
                pts.append(x * q + y)
            lines.append(pts)

    num_lines = len(lines)

    # Precompute line properties
    line_U_intersections = []
    line_outside_coverage = []
    line_point_sets = []

    for L in lines:
        I = set(pt for pt in L if pt in U)
        cov = set(pt for pt in L if pt in outside_U)
        line_U_intersections.append(I)
        line_outside_coverage.append(cov)
        line_point_sets.append(set(L))

    def run_greedy(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = set(outside_U)
        H = set()
        chosen_lines = []
        
        max_iterations = 10000
        iteration = 0
        
        while uncovered and iteration < max_iterations:
            if time.time() - start_time > DEADLINE - 100:
                break
                
            best_line_idx = -1
            best_score = -1
            
            # Add small random noise to break ties and explore different solutions
            noise = random.uniform(0, 0.1) if random_seed is not None else 0
            
            for i in range(num_lines):
                if not line_outside_coverage[i]:
                    continue

                actual_new_cov = 0
                for pt in line_outside_coverage[i]:
                    if pt in uncovered:
                        actual_new_cov += 1

                if actual_new_cov == 0:
                    continue

                I_L = line_U_intersections[i]
                new_H_cost = 0
                for pt in I_L:
                    if pt not in H:
                        new_H_cost += 1

                score = actual_new_cov / (new_H_cost + 1) + noise
                # Actually, noise should be small.
                
                if score > best_score:
                    best_score = score
                    best_line_idx = i

            if best_line_idx == -1:
                break

            I_L = line_U_intersections[best_line_idx]
            H.update(I_L)
            chosen_lines.append(best_line_idx)

            for pt in lines[best_line_idx]:
                if pt in uncovered:
                    uncovered.remove(pt)

            iteration += 1
            
        return H, chosen_lines

    def local_search(H, chosen_lines):
        # Try to remove lines
        improved = True
        while improved:
            improved = False
            for i in range(len(chosen_lines)):
                if time.time() - start_time > DEADLINE - 50:
                    return H, chosen_lines
                
                removed_line = chosen_lines[i]
                removed_pts = set(lines[removed_line])
                
                # Check if all points in removed_pts are still covered
                still_covered = True
                for pt in removed_pts:
                    if pt in outside_U:
                        covered = False
                        for j in range(len(chosen_lines)):
                            if j != i and pt in line_point_sets[chosen_lines[j]]:
                                covered = True
                                break
                        if not covered:
                            still_covered = False
                            break
                
                if still_covered:
                    # Remove line
                    chosen_lines.pop(i)
                    # Update H
                    removed_U = set(line_U_intersections[removed_line])
                    for pt in removed_U:
                        in_other = False
                        for j in range(len(chosen_lines)):
                            if pt in line_U_intersections[chosen_lines[j]]:
                                in_other = True
                                break
                        if not in_other:
                            H.discard(pt)
                    improved = True
                    break
        return H, chosen_lines

    best_H = None
    best_size = float('inf')
    
    # Run multiple greedy passes
    num_runs = 10
    for r in range(num_runs):
        if time.time() - start_time > DEADLINE - 150:
            break
            
        H, chosen_lines = run_greedy(r)
        if len(H) < best_size:
            best_H = set(H)
            best_chosen_lines = list(chosen_lines)
            best_size = len(H)
            
    if best_H is None:
        # Fallback
        H, chosen_lines = run_greedy()
        best_H = set(H)
        best_chosen_lines = list(chosen_lines)
        best_size = len(H)

    # Local search on best
    best_H, best_chosen_lines = local_search(best_H, best_chosen_lines)
    
    S = U - best_H
    N = all_points_set - S

    points = []
    for pt in N:
        x = pt // q
        y = pt % q
        xa, xb = x // p, x % p
        ya, yb = y // p, y % p
        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
