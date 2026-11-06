import numpy as np
import itertools
import random
import time
from scipy.optimize import milp, LinearConstraint, Bounds
import sys

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)
POINT_TO_INDEX = {p: i for i, p in enumerate(GRID_POINTS)}

# Precompute squared distances
coords = np.array(GRID_POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diffs * diffs, axis=-1)

def get_isosceles_triples():
    """
    Returns a list of tuples (a, b, c) representing isosceles triangles.
    We identify them by apex.
    """
    triples = []
    for c in range(NUM_POINTS):
        # Get distances from c to all other points
        dists = dist_sq[c, :].copy()
        dists[c] = -1 # Exclude self
        
        # Group points by distance
        dist_to_points = {}
        for p in range(NUM_POINTS):
            if p == c:
                continue
            d = dists[p]
            if d not in dist_to_points:
                dist_to_points[d] = []
            dist_to_points[d].append(p)
        
        # For each distance group with > 1 point, generate pairs
        for d, points in dist_to_points.items():
            if len(points) > 1:
                for a, b in itertools.combinations(points, 2):
                    # Triple is (a, b, c) with c as apex
                    triples.append((a, b, c))
                    
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # 1. Generate isosceles triples
    triples = get_isosceles_triples()
    
    # 2. Formulate ILP
    # Variables:
    # x[0..N-1]: binary, 1 if point i is in S
    # w[0..W-1]: binary, auxiliary variables for maximality constraints
    
    # Map each pair {a,b} for a given q to a variable index
    # We need to iterate over q, find all pairs {a,b} such that {a,b,q} is isosceles.
    
    # Pre-compute for each q, the list of (a,b) pairs
    q_pairs = {q: [] for q in range(NUM_POINTS)}
    
    # We have triples (a,b,c) where c is apex.
    # This means {a,b,c} is isosceles with c as apex.
    # So for q=c, the pair is (a,b).
    # Also, {a,b,c} is isosceles. Does it have other apices?
    # We established no equilateral triangles.
    # So c is the UNIQUE apex.
    # However, the set {a,b,c} is isosceles.
    # If we consider q=a, is there a pair {b,c} such that {a,b,c} is isosceles with a as apex?
    # Only if d(a,b)=d(a,c). But d(a,b) != d(a,c) because c is the unique apex (d(c,a)=d(c,b) != d(a,b)).
    # Wait. If c is apex, d(c,a)=d(c,b).
    # Is it possible that d(a,b)=d(a,c)?
    # If so, a would also be an apex. That would mean d(a,b)=d(a,c)=d(c,a) -> Equilateral.
    # So no.
    # So for the triple {a,b,c} with unique apex c:
    # - For q=c, the pair is (a,b).
    # - For q=a, the pair is (b,c)? No.
    #   The condition for q=a to be blocked is: exists b,c in S such that {a,b,c} is isosceles.
    #   We know {a,b,c} is isosceles.
    #   So if b,c are in S, a is blocked.
    #   So for q=a, the pair is (b,c).
    #   Similarly for q=b, the pair is (a,c).
    
    # So we must add entries to q_pairs for ALL vertices of the isosceles triangle.
    for a, b, c in triples:
        # c is apex
        q_pairs[c].append((a, b))
        # a is base vertex
        q_pairs[a].append((b, c))
        # b is base vertex
        q_pairs[b].append((a, c))

    # Create variable indices for w
    # We need a unique ID for each (q, a, b) combination?
    # Actually, w is defined for each (q, pair).
    # Let's create a list of all w variables.
    w_vars = []
    w_index_map = {} # (q, a, b) -> index in w_vars
    
    for q in range(NUM_POINTS):
        for a, b in q_pairs[q]:
            key = (q, min(a,b), max(a,b))
            if key not in w_index_map:
                w_index_map[key] = len(w_vars)
                w_vars.append(key)
                
    num_w = len(w_vars)
    num_x = NUM_POINTS
    total_vars = num_x + num_w
    
    # Objective: Minimize sum(x)
    c_obj = np.zeros(total_vars)
    c_obj[:num_x] = 1.0
    
    # Constraints
    # 1. No Isosceles: x_a + x_b + x_c <= 2
    A_rows = []
    b_rows = []
    
    for a, b, c in triples:
        row = np.zeros(total_vars)
        row[a] = 1
        row[b] = 1
        row[c] = 1
        A_rows.append(row)
        b_rows.append(2)
        
    # 2. Maximality: For each q, sum(w_{q, a, b}) >= 1 - x_q
    # Rearranged: x_q + sum(w_{q, a, b}) >= 1
    # Wait, the constraint is: if x_q=0, sum(w) >= 1.
    # If x_q=1, sum(w) >= 0 (trivially true).
    # So: x_q + sum(w_{q, a, b}) >= 1 is INCORRECT.
    # If x_q=1, 1 + 0 >= 1 is True.
    # If x_q=0, 0 + sum(w) >= 1 is True iff sum(w) >= 1.
    # Yes, this linear constraint works!
    
    for q in range(NUM_POINTS):
        row = np.zeros(total_vars)
        row[q] = 1 # x_q
        
        for a, b in q_pairs[q]:
            key = (q, min(a,b), max(a,b))
            if key in w_index_map:
                idx = w_index_map[key]
                row[num_x + idx] = 1 # w variable
                
        A_rows.append(row)
        b_rows.append(1) # Lower bound
        
    # 3. Linearization of w_{q,a,b} = x_a * x_b
    # w <= x_a
    # w <= x_b
    # w >= x_a + x_b - 1
    # These are:
    # -x_a + w <= 0
    # -x_b + w <= 0
    # x_a + x_b - w <= 1
    
    for q in range(NUM_POINTS):
        for a, b in q_pairs[q]:
            key = (q, min(a,b), max(a,b))
            if key not in w_index_map:
                continue
            idx = w_index_map[key]
            w_idx = num_x + idx
            
            # -x_a + w <= 0
            row1 = np.zeros(total_vars)
            row1[a] = -1
            row1[w_idx] = 1
            A_rows.append(row1)
            b_rows.append(0)
            
            # -x_b + w <= 0
            row2 = np.zeros(total_vars)
            row2[b] = -1
            row2[w_idx] = 1
            A_rows.append(row2)
            b_rows.append(0)
            
            # x_a + x_b - w <= 1
            row3 = np.zeros(total_vars)
            row3[a] = 1
            row3[b] = 1
            row3[w_idx] = -1
            A_rows.append(row3)
            b_rows.append(1)
            
    # Build sparse matrix for efficiency? 
    # HiGHS accepts dense or sparse. Given size, dense might be okay, but sparse is better.
    # Let's use scipy.sparse
    from scipy.sparse import csr_matrix
    
    # Convert A_rows to sparse
    # We have many rows.
    # Let's collect non-zeros
    # Actually, building it row by row and concatenating is slow.
    # Let's use the list of rows.
    
    A_sparse = csr_matrix(np.array(A_rows))
    b_sparse = np.array(b_rows)
    
    # Constraints:
    # Rows 0 to len(triples)-1: A x <= b (2)
    # Rows len(triples) to ...: A x >= b (1)
    # Rows ... to end: A x <= b (0 or 1)
    
    num_no_iso = len(triples)
    num_max = NUM_POINTS
    
    # LinearConstraint takes (lb, ub)
    # We can split into two constraints or use mixed.
    # HiGHS supports mixed via separate constraints? 
    # scipy.optimize.milp takes a single LinearConstraint object.
    # We can combine them if we adjust bounds.
    # But directions are different.
    # We can create multiple LinearConstraints? No, milp takes one.
    # We can convert >= to <= by negating.
    
    # Let's just create a big matrix with all <= constraints.
    # 1. No Iso: <= 2
    # 2. Max: x_q + sum(w) >= 1  =>  -x_q - sum(w) <= -1
    # 3. Lin: <= 0, <= 0, <= 1
    
    A_le_rows = []
    b_le_rows = []
    
    # 1. No Iso
    for i in range(num_no_iso):
        A_le_rows.append(A_rows[i])
        b_le_rows.append(2)
        
    # 2. Max (Negated)
    for i in range(num_no_iso, num_no_iso + num_max):
        A_le_rows.append(-A_rows[i])
        b_le_rows.append(-1)
        
    # 3. Linearization
    for i in range(num_no_iso + num_max, len(A_rows)):
        A_le_rows.append(A_rows[i])
        b_le_rows.append(b_rows[i])
        
    A_final = csr_matrix(np.array(A_le_rows))
    b_final = np.array(b_le_rows)
    
    # Integrality
    integrality = np.ones(total_vars)
    
    # Bounds
    bounds = Bounds(lb=0, ub=1)
    
    # Solve
    try:
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A_final, -np.inf, b_final),
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 250} # Leave 30s for overhead
        )
        
        if res.status == 0: # Optimal
            x_sol = res.x[:num_x]
            S_indices = [i for i in range(num_x) if x_sol[i] > 0.5]
            return [GRID_POINTS[i] for i in S_indices]
        else:
            # If not optimal, check if feasible
            if res.status == 1: # Feasible
                x_sol = res.x[:num_x]
                S_indices = [i for i in range(num_x) if x_sol[i] > 0.5]
                return [GRID_POINTS[i] for i in S_indices]
            else:
                # Infeasible or other error, fallback
                pass
                
    except Exception as e:
        print(f"ILP failed: {e}", file=sys.stderr)
        
    # Fallback: Heuristic
    # Re-use the logic from the original program?
    # To keep the edit small, I will just return a simple greedy result if ILP fails.
    # But the original program had a good heuristic.
    # I will include a minimal version of the heuristic as fallback.
    
    def greedy_construct(random_order):
        S = []
        S_set = set()
        dist_sets = {}
        
        for p in random_order:
            if p in S_set:
                continue
            
            # Check can_add
            can_add = True
            dists_from_p = dist_sq[p, S]
            if len(S) > 0 and len(np.unique(dists_from_p)) != len(dists_from_p):
                can_add = False
            
            if can_add:
                for a in S:
                    if dist_sq[p, a] in dist_sets.get(a, set()):
                        can_add = False
                        break
            
            if can_add:
                S.append(p)
                S_set.add(p)
                
                for a in S[:-1]:
                    d = dist_sq[p, a]
                    if a not in dist_sets:
                        dist_sets[a] = set()
                    dist_sets[a].add(d)
                
                if S[:-1]:
                    dist_sets[p] = set(dist_sq[p, S[:-1]])
                else:
                    dist_sets[p] = set()
        
        return S

    random_order = list(range(NUM_POINTS))
    random.shuffle(random_order)
    S = greedy_construct(random_order)
    return [GRID_POINTS[i] for i in S]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
