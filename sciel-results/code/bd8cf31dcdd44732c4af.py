import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 58.0  # Safety margin before 90s limit

    def elapsed():
        return time.time() - t0

    def get_max_area_and_counter(pts):
        """
        Compute the maximum multiplicity of any non-zero doubled area and the full counter.
        Optimized using vectorized numpy operations.
        """
        n = len(pts)
        if n < 3:
            return 0, {}

        P = np.array(pts, dtype=np.int64)
        
        # Generate all pairs (i, j) with i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx] # shape (M, 2)
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)
        
        # We need to compute cross products for all triples (i, j, k) with i < j < k.
        # This is tricky to do in one vectorized shot for all triples directly from pairs.
        # However, we can iterate over the "base" point i and vectorize over j, k for fixed i?
        # Or iterate over pairs (i, j) and vectorize over k > j?
        
        # Let's stick to the loop over u (base vector index in the pair list) but optimize the inner loop.
        # Actually, the previous method was O(M^2) in worst case if not careful, but M is ~435.
        # M^2 is ~190k operations per score. 
        # The bottleneck in the previous code was likely the python loop overhead and dict updates.
        
        # Let's try a fully vectorized approach for small N.
        # For N=30, total triples is 4060.
        # We can generate all triples indices.
        
        # Indices for triples i < j < k
        # This creates arrays of length C(30,3) = 4060
        # This is very small. We can just compute all areas in one vectorized op.
        
        # Generate all combinations of 3 indices
        # itertools.combinations is slow in python loop.
        # Let's use numpy to generate them? Or just use the pair difference method but optimized.
        
        # Method:
        # For each pair (i, j), we have vector v = P[j] - P[i].
        # For each k > j, area = 0.5 * |cross(v, P[k] - P[i])| = 0.5 * |cross(v, P[k] - P[j])|? No.
        # Area = 0.5 * | (P[j]-P[i]) x (P[k]-P[i]) |
        
        # Let's just compute all 4060 areas directly. It's fast enough.
        # P is (30, 2).
        # We need indices i, j, k.
        
        # Creating indices for all triples
        # i in 0..27, j in i+1..28, k in j+1..29
        
        # This might be slower to generate indices than to loop.
        # Let's try the loop over i, vectorizing j, k.
        
        area_counts = defaultdict(int)
        
        for i in range(n - 2):
            # Points j, k > i
            Pi = P[i]
            Pj = P[i+1:]
            # We need pairs (j, k) from Pj such that j_idx < k_idx in original array
            # Pj has indices i+1 .. n-1
            # Let's just get the submatrix of points > i
            # And compute cross products of (Pj - Pi) and (Pk - Pi) for all pairs in Pj
            
            # This is getting complicated. Let's revert to the pair-difference method but make it as fast as possible.
            # The previous method iterated M times. M=435.
            # Inside, it did vector ops on arrays of size M-u.
            # Total ops ~ M^2/2 ~ 100k.
            # The overhead was python loops.
            
            pass
        
        # Alternative: Precompute all pairwise cross products? No, cross product depends on the third point.
        
        # Let's use the fact that N is small (30).
        # We can generate all triples using a list comprehension and numpy vectorization on the result?
        # No, let's just use a tight loop with numpy slicing.
        
        area_counts = defaultdict(int)
        
        # Precompute P as int64
        # Loop over i
        for i in range(n - 2):
            Pi = P[i]
            # Points from i+1 to end
            rest = P[i+1:]
            # We need pairs (j, k) from rest.
            # Let's just iterate j over rest
            m = len(rest)
            for j in range(m - 1):
                Pj = rest[j]
                v1 = Pj - Pi
                # Pk for k > j in rest
                Pks = rest[j+1:]
                v2s = Pks - Pi
                # Cross product: v1 x v2s
                # v1 is (2,), v2s is (m-j-1, 2)
                cross = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
                abs_cross = np.abs(cross)
                nz = abs_cross[abs_cross != 0]
                if len(nz) > 0:
                    vals, cnts = np.unique(nz, return_counts=True)
                    for v, c in zip(vals, cnts):
                        area_counts[int(v)] += int(c)
                        
        if not area_counts:
            return 0, {}
        
        max_count = max(area_counts.values())
        return max_count, dict(area_counts)

    def local_search_for_A(pts, A, time_budget):
        """
        Local search maximizing count of triangles with area exactly A.
        Returns (best_points, best_count_A, best_max_area).
        """
        current = list(pts)
        current_max_area, current_counter = get_max_area_and_counter(current)
        current_count_A = current_counter.get(A, 0)
        best = list(current)
        best_count_A = current_count_A
        best_max_area = current_max_area

        t_start = time.time()
        no_improve = 0
        step_size = 3
        random.seed(int(A) % 10000 + 42)

        while time.time() - t_start < time_budget:
            idx = random.randint(0, 29)
            old_x, old_y = current[idx]

            r = random.random()
            if r < 0.05:
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
            elif r < 0.2:
                new_x = old_x + random.randint(-5, 5)
                new_y = old_y + random.randint(-5, 5)
            else:
                new_x = old_x + random.randint(-step_size, step_size)
                new_y = old_y + random.randint(-step_size, step_size)

            if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                continue
            new_pt = (new_x, new_y)
            if new_pt == (old_x, old_y):
                continue
            is_valid = True
            for j in range(30):
                if j != idx and current[j] == new_pt:
                    is_valid = False
                    break
            if not is_valid:
                continue

            current[idx] = new_pt
            new_max_area, new_counter = get_max_area_and_counter(current)
            new_count_A = new_counter.get(A, 0)

            # Acceptance criteria:
            # 1. Max area increased
            # 2. Target area A count increased
            # 3. Small probability of accepting neutral/slightly worse to escape local minima
            if new_max_area > current_max_area or new_count_A > current_count_A:
                current_max_area = new_max_area
                current_count_A = new_count_A
                current_counter = new_counter
                no_improve = 0
                
                if new_max_area > best_max_area or (new_max_area == best_max_area and new_count_A > best_count_A):
                    best_max_area = new_max_area
                    best_count_A = new_count_A
                    best = list(current)
                
                if step_size > 1 and random.random() < 0.2:
                    step_size = max(1, step_size - 1)
            else:
                current[idx] = (old_x, old_y)
                no_improve += 1

            if no_improve > 1000:
                current = list(best)
                current_max_area = best_max_area
                current_count_A = best_count_A
                no_improve = 0
                step_size = random.randint(4, 10)

            if random.random() < 0.005:
                step_size = random.randint(1, 6)

        return best, best_count_A, best_max_area

    # Generate diverse initial candidates
    candidates = []

    # Candidate 1: Two horizontal lines
    c1 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    candidates.append(c1)

    # Candidate 2: Grid 6x5
    c2 = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(c2)

    # Candidate 3: Three lines
    c3 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]
    candidates.append(c3)

    # Candidate 4: Random
    random.seed(42)
    c4 = []
    used = set()
    while len(c4) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in used:
            used.add((x, y))
            c4.append((x, y))
    candidates.append(c4)

    # Candidate 5: Five lines, 6 each
    c5 = [(x, 0) for x in range(6)] + [(x, 1) for x in range(6)] + \
         [(x, 2) for x in range(6)] + [(x, 3) for x in range(6)] + \
         [(x, 4) for x in range(6)]
    candidates.append(c5)

    # Candidate 6: Two lines offset
    c6 = [(x, 0) for x in range(15)] + [(x, 2) for x in range(15)]
    candidates.append(c6)

    # Candidate 7: Three lines with spacing
    c7 = [(x, 0) for x in range(10)] + [(x, 2) for x in range(10)] + [(x, 4) for x in range(10)]
    candidates.append(c7)

    # Candidate 8: Four lines
    c8 = [(x, 0) for x in range(8)] + [(x, 1) for x in range(8)] + \
         [(x, 3) for x in range(7)] + [(x, 5) for x in range(7)]
    candidates.append(c8)

    # Evaluate all candidates
    best_points = None
    best_score = 0
    best_area_A = 0

    for cand in candidates:
        s, counter = get_max_area_and_counter(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            if counter:
                best_area_A = max(counter, key=counter.get)

    # Phase 1: Local search from best candidate (maximizing max area)
    current_points = list(best_points)
    current_score = best_score

    t_phase1_start = time.time()
    phase1_budget = min(15.0, DEADLINE - elapsed() - 20.0)
    if phase1_budget < 2.0:
        phase1_budget = 2.0

    random.seed(123)
    no_improve = 0
    step_size = 3

    while time.time() - t_phase1_start < phase1_budget and elapsed() < DEADLINE - 15.0:
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]

        r = random.random()
        if r < 0.03:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif r < 0.15:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            new_x = old_x + random.randint(-step_size, step_size)
            new_y = old_y + random.randint(-step_size, step_size)

        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
        new_pt = (new_x, new_y)
        if new_pt == (old_x, old_y):
            continue
        is_valid = True
        for j in range(30):
            if j != idx and current_points[j] == new_pt:
                is_valid = False
                break
        if not is_valid:
            continue

        current_points[idx] = new_pt
        new_score, new_counter = get_max_area_and_counter(current_points)

        if new_score > current_score:
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
                if new_counter:
                    best_area_A = max(new_counter, key=new_counter.get)
            if step_size > 1 and random.random() < 0.2:
                step_size = max(1, step_size - 1)
        else:
            current_points[idx] = (old_x, old_y)
            no_improve += 1

        if no_improve > 1500:
            current_points = list(best_points)
            current_score = best_score
            no_improve = 0
            step_size = random.randint(4, 10)

        if random.random() < 0.003:
            step_size = random.randint(1, 6)

    # Phase 2: Model-and-optimize - try multiple target areas A
    # Use remaining time for targeted area optimization
    remaining_time = DEADLINE - elapsed() - 2.0
    if remaining_time > 3.0:
        # Get top candidate areas from current best
        _, counter = get_max_area_and_counter(best_points)
        if counter:
            # Sort areas by count, try top few
            sorted_areas = sorted(counter.items(), key=lambda x: -x[1])
            # Also add some small areas that might be promising
            candidate_areas = [a for a, c in sorted_areas[:5]]
            # Add some small values
            for a in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
                if a not in candidate_areas:
                    candidate_areas.append(a)
            candidate_areas = candidate_areas[:12]  # Limit number of areas to try

            time_per_area = max(1.0, remaining_time / len(candidate_areas))

            for A in candidate_areas:
                if elapsed() > DEADLINE - 2.0:
                    break
                t_area_start = time.time()
                area_budget = min(time_per_area, DEADLINE - elapsed() - 2.0)
                if area_budget < 0.5:
                    break

                # Start from current best and do local search for area A
                pts_for_A = list(best_points)
                result_pts, result_count_A, result_max = local_search_for_A(
                    pts_for_A, A, area_budget
                )

                if result_max > best_score:
                    best_score = result_max
                    best_points = result_pts

    # Final check
    if elapsed() < DEADLINE - 1.0:
        final_score, _ = get_max_area_and_counter(best_points)
        if final_score > best_score:
            best_score = final_score

    return best_points
