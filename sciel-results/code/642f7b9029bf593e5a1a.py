# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    deadline = time.time() + 570  # 30s safety margin before 600s limit
    
    # Field arithmetic: F_49 = F_7[alpha], alpha^2 = 3
    # Element represented as (a, b) for a + b*alpha
    # Addition: component-wise mod 7
    # Multiplication: (a,b)*(c,d) = (ac + 3bd, ad + bc) mod 7
    # Inverse: (a,b)^(-1) = (a, -b)/(a^2 - 3b^2) = (a, -b) * inv(a^2 - 3b^2)
    # where inv is mod 7 inverse
    
    k_const = 3  # alpha^2 = 3
    
    # Precompute field multiplication and inverse tables
    # Index: a * p + b
    def idx(a, b):
        return a * p + b
    
    def add(a1, b1, a2, b2):
        return ((a1 + a2) % p, (b1 + b2) % p)
    
    def mul(a1, b1, a2, b2):
        return ((a1*a2 + k_const*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    def norm(a, b):
        return (a*a - k_const*b*b) % p
    
    def inv(a, b):
        n = norm(a, b)
        if n == 0:
            return None
        n_inv = pow(n, -1, p)
        return ((a * n_inv) % p, ((p - b) * n_inv) % p)
    
    # All field elements as (a, b) pairs
    field_elems = [(a, b) for a in range(p) for b in range(p)]
    
    # Map from (a,b) to index
    elem_to_idx = {}
    for i, (a, b) in enumerate(field_elems):
        elem_to_idx[(a, b)] = i
    
    # Precompute multiplication table
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i, (a1, b1) in enumerate(field_elems):
        for j, (a2, b2) in enumerate(field_elems):
            r_a, r_b = mul(a1, b1, a2, b2)
            mul_table[i, j] = idx(r_a, r_b)
    
    # Precompute inverse
    inv_table = np.full(q, -1, dtype=np.int64)
    for i, (a, b) in enumerate(field_elems):
        r = inv(a, b)
        if r is not None:
            inv_table[i] = elem_to_idx[r]
    
    # Points in F_q^2: each point is (x_idx, y_idx) where x_idx, y_idx in [0, q)
    # Total points: q^2 = 2401
    # A point can be indexed as x_idx * q + y_idx
    point_count = q * q
    
    # Directions: (1, m) for m in F_q, and (0, 1)
    # m values: all q field elements
    # So total directions: q + 1 = 50
    # direction 0: (0, 1)
    # direction 1..q: (1, m) where m = field_elems[d-1]
    
    num_dirs = q + 1  # 50
    
    # For each direction, precompute the line membership:
    # line_dir[d][point] = line_id (or -1 if not on this line structure)
    # Actually, for a given direction, each point belongs to exactly one line.
    # A line in direction d through point p contains q points.
    # line_id for direction d: p - (p mod q) ... no.
    # For direction (1, m): line through (x, y) is {(x + t, y + t*m) : t in F_q}
    # The "base point" of a line: when x = 0, y = y0 where y0 = y - x*m
    # So line_id = elem_to_idx[y0]
    # For direction (0, 1): line through (x, y) is {(x, y + t) : t in F_q}
    # line_id = x_idx
    
    # Precompute for each direction and each point, the line_id
    # line_id[d][point_idx] where point_idx = x_idx * q + y_idx
    # And also, for each line, the list of points on it (excluding the base for counting)
    
    # Actually, let's think about the Nikodym condition more carefully.
    # N is a Nikodym set if for every point x in F_q^2 (in N or not),
    # there exists a direction v (from the allowed set) such that
    # {x + t*v : t in F_q, t != 0} is fully contained in N.
    
    # We want to maximize |complement| = 2401 - |N|.
    # Let E = complement of N (the excluded set).
    # For each point x, there must exist a direction d such that
    # all other points on the line through x in direction d are in N (i.e., not in E).
    # Equivalently: for each point x, there exists a direction d such that
    # E contains at most one point on the line through x in direction d (namely possibly x itself).
    
    # So the condition is: for every point x, there is at least one direction d
    # such that |E ∩ L_d(x)| <= 1, where L_d(x) is the line through x in direction d.
    
    # We want to maximize |E| subject to this constraint.
    
    # This is a combinatorial optimization problem. Let's use a greedy approach:
    # Start with E = empty, and try to add points to E one by one,
    # maintaining the invariant that for every point x, at least one direction d
    # has |E ∩ L_d(x)| <= 1.
    
    # For each point x and direction d, track count_d(x) = |E ∩ L_d(x)|.
    # A point x is "satisfied" if min_d count_d(x) <= 1.
    # When we add a point y to E, for each direction d, all points x on L_d(y)
    # get their count_d(x) incremented by 1.
    # We need to ensure that no point x becomes unsatisfied (all its counts > 1).
    
    # Let's implement this.
    
    # Precompute lines: for each direction d, group points into lines.
    # lines[d][line_id] = list of point indices on that line.
    
    lines = []
    for d in range(num_dirs):
        if d == 0:
            # Direction (0, 1): lines are vertical, one for each x
            # line_id = x_idx
            line_map = {}
            for x_idx in range(q):
                for y_idx in range(q):
                    point_idx = x_idx * q + y_idx
                    if x_idx not in line_map:
                        line_map[x_idx] = []
                    line_map[x_idx].append(point_idx)
            lines.append(list(line_map.values()))
        else:
            # Direction (1, m) where m = field_elems[d-1]
            m_idx = d - 1
            m_a, m_b = field_elems[m_idx]
            # Line through (x, y): base point is (0, y - x*m)
            # line_id = elem_to_idx[y - x*m]
            line_map = {}
            for x_idx in range(q):
                x_a, x_b = field_elems[x_idx]
                for y_idx in range(q):
                    point_idx = x_idx * q + y_idx
                    y_a, y_b = field_elems[y_idx]
                    # Compute y - x*m
                    # y + (-1)*x*m
                    # (-1, 0) * (x, m) = (-x, 0) ... wait
                    # In F_49, -1 = (6, 0)
                    # neg_x = (p - x_a, p - x_b) if not both 0
                    if x_a == 0 and x_b == 0:
                        neg_x = (0, 0)
                    else:
                        neg_x = ((p - x_a) % p, (p - x_b) % p)
                    # neg_x * m
                    nx_a, nx_b = neg_x
                    nm_a, nm_b = field_elems[m_idx]
                    prod_a, prod_b = mul(nx_a, nx_b, nm_a, nm_b)
                    # y + prod
                    sum_a = (y_a + prod_a) % p
                    sum_b = (y_b + prod_b) % p
                    line_id = idx(sum_a, sum_b)
                    if line_id not in line_map:
                        line_map[line_id] = []
                    line_map[line_id].append(point_idx)
            lines.append(list(line_map.values()))
    
    # For each point and direction, precompute the line_id
    point_line_id = np.zeros((point_count, num_dirs), dtype=np.int64)
    for d in range(num_dirs):
        if d == 0:
            for x_idx in range(q):
                for y_idx in range(q):
                    point_idx = x_idx * q + y_idx
                    point_line_id[point_idx, d] = x_idx
        else:
            m_idx = d - 1
            m_a, m_b = field_elems[m_idx]
            for x_idx in range(q):
                x_a, x_b = field_elems[x_idx]
                for y_idx in range(q):
                    point_idx = x_idx * q + y_idx
                    y_a, y_b = field_elems[y_idx]
                    if x_a == 0 and x_b == 0:
                        neg_x = (0, 0)
                    else:
                        neg_x = ((p - x_a) % p, (p - x_b) % p)
                    nx_a, nx_b = neg_x
                    nm_a, nm_b = field_elems[m_idx]
                    prod_a, prod_b = mul(nx_a, nx_b, nm_a, nm_b)
                    sum_a = (y_a + prod_a) % p
                    sum_b = (y_b + prod_b) % p
                    line_id = idx(sum_a, sum_b)
                    point_line_id[point_idx, d] = line_id
    
    # Now, for each direction d and line_id, we have the list of points.
    # We'll use a greedy approach to build E.
    
    # counts[d][line_id] = number of points in E on that line
    # For each point x, we track how many directions d have counts[d][line_id(x,d)] <= 1
    # sat_count[x] = number of directions d where counts[d][point_line_id[x,d]] <= 1
    # Point x is satisfied iff sat_count[x] > 0
    
    # Initialize
    counts = np.zeros((num_dirs, q), dtype=np.int64)  # num_dirs x max_line_id
    # For d=0, line_ids are 0..q-1 (x_idx)
    # For d>0, line_ids are 0..q-1 (field element index)
    # So all line_ids are in [0, q)
    
    sat_count = np.zeros(point_count, dtype=np.int64)
    for x in range(point_count):
        for d in range(num_dirs):
            lid = point_line_id[x, d]
            if counts[d, lid] <= 1:
                sat_count[x] += 1
    
    in_E = np.zeros(point_count, dtype=bool)
    
    # Greedy: try to add points to E in some order
    # We'll try random orderings and pick the best result
    
    best_E = np.zeros(point_count, dtype=bool)
    best_size = 0
    
    start_time = time.time()
    
    # Try multiple random permutations
    num_restarts = 0
    while time.time() < deadline - 60:  # leave 60s margin
        rng = np.random.RandomState(num_restarts * 12345 + 678)
        order = rng.permutation(point_count)
        
        # Reset
        counts.fill(0)
        in_E.fill(False)
        for x in range(point_count):
            sat_count[x] = num_dirs  # all directions have count 0 <= 1
        
        e_size = 0
        for point_idx in order:
            # Check if we can add point_idx to E
            # For each direction d, the line through point_idx in direction d
            # will have its count incremented.
            # We need to check that no point x on any of these lines becomes unsatisfied.
            
            # For direction d, line_id = point_line_id[point_idx, d]
            # Points on this line: lines[d][line_id]
            # After adding, counts[d, line_id] increases by 1.
            # For each point x on this line, sat_count[x] may decrease
            # if counts[d, line_id] was 0 (becomes 1, still <= 1, no change in sat)
            # or was 1 (becomes 2, now > 1, sat_count[x] decreases by 1)
            # or was >= 2 (becomes >= 3, no change in sat)
            
            # So we only need to check points x where counts[d, point_line_id[x,d]] == 1
            # and x is on the line through point_idx in direction d.
            
            can_add = True
            # We need to track which points would become unsatisfied
            # and whether they already have other satisfied directions.
            
            # For efficiency, let's check each direction d:
            # lid = point_line_id[point_idx, d]
            # if counts[d, lid] == 0:
            #   After: counts becomes 1. No point's sat_count changes (1 <= 1 still).
            #   Actually wait: sat_count counts directions where count <= 1.
            #   count 0 -> 1: still <= 1, so no change.
            # if counts[d, lid] == 1:
            #   After: counts becomes 2. For each x on this line,
            #   direction d no longer satisfies x. sat_count[x] -= 1.
            #   If sat_count[x] becomes 0, x is unsatisfied -> can't add.
            # if counts[d, lid] >= 2:
            #   No change in sat_count.
            
            # So only directions where counts[d, lid] == 1 matter.
            
            for d in range(num_dirs):
                lid = point_line_id[point_idx, d]
                if counts[d, lid] == 1:
                    # Check all points x on this line
                    for x in lines[d][lid]:
                        if sat_count[x] == 1:
                            # This point would become unsatisfied
                            can_add = False
                            break
                    if not can_add:
                        break
            
            if can_add:
                # Add point_idx to E
                in_E[point_idx] = True
                e_size += 1
                for d in range(num_dirs):
                    lid = point_line_id[point_idx, d]
                    if counts[d, lid] == 1:
                        for x in lines[d][lid]:
                            sat_count[x] -= 1
                    counts[d, lid] += 1
        
        if e_size > best_size:
            best_size = e_size
            best_E = in_E.copy()
        
        num_restarts += 1
    
    # Build N = complement of E
    n_points = []
    for point_idx in range(point_count):
        if not best_E[point_idx]:
            x_idx = point_idx // q
            y_idx = point_idx % q
            x_a, x_b = field_elems[x_idx]
            y_a, y_b = field_elems[y_idx]
            n_points.append([[x_a, x_b], [y_a, y_b]])
    
    return np.array(n_points, dtype=np.int64)


# EVOLVE-BLOCK-END
