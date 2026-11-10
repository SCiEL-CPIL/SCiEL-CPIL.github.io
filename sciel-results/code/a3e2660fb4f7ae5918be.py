import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # Build field operations
    def field_add(x, y):
        return ((x[0]+y[0])%p, (x[1]+y[1])%p)
    
    def field_mul(x, y):
        return ((x[0]*y[0] + k*x[1]*y[1])%p, (x[0]*y[1]+x[1]*y[0])%p)
    
    def field_neg(x):
        return ((-x[0])%p, (-x[1])%p)
    
    def field_sub(x, y):
        return field_add(x, field_neg(y))
    
    # Enumerate all field elements
    field_elements = [(a,b) for a in range(p) for b in range(p)]
    
    # All points in F_49^2
    all_points = [((xa,xb),(ya,yb)) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    n_points = 2401
    
    # Precompute all lines
    lines = []  # list of sets of point indices
    
    # Vertical lines: direction (0,1)
    for xa in range(p):
        for xb in range(p):
            line_pts = set()
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa,xb),(ya,yb))
                    line_pts.add(point_to_idx[pt])
            lines.append(line_pts)
    
    # Slanted lines: direction (1,m)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_pts = set()
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = field_mul(m, x)
                            y = field_add(mx, c)
                            pt = (x, y)
                            line_pts.add(point_to_idx[pt])
                    lines.append(line_pts)
    
    # For each point, which lines pass through it
    point_lines = [[] for _ in range(n_points)]
    for li, line in enumerate(lines):
        for pt_idx in line:
            point_lines[pt_idx].append(li)
    
    # Count how many complement points are on each line (for fast checking)
    # complement_count[line_idx] = number of points in C on that line
    
    def verify_nikodym(C_set):
        """Check if N = all_points - C_set is a Nikodym set."""
        C_frozen = frozenset(C_set)
        for pt_idx in range(n_points):
            found = False
            for li in point_lines[pt_idx]:
                line = lines[li]
                valid = True
                for other in line:
                    if other != pt_idx and other in C_frozen:
                        valid = False
                        break
                if valid:
                    found = True
                    break
            if not found:
                return False
        return True
    
    def count_complement_on_line(line, C_set):
        """Count points in C_set that are on this line."""
        cnt = 0
        for pt in line:
            if pt in C_set:
                cnt += 1
        return cnt
    
    # Try multiple seed constructions
    best_C = None
    best_size = 0
    
    # Seed 1: Hermitian unital with ya not in squares
    squares_mod7 = {0, 1, 2, 4}
    C1 = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2*xa + ya*ya - k*yb*yb) % p == 0
                    if in_unital and ya not in squares_mod7:
                        C1.add(point_to_idx[((xa,xb),(ya,yb))])
    
    # Seed 2: Hermitian unital with ya not in {0}
    C2 = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2*xa + ya*ya - k*yb*yb) % p == 0
                    if in_unital and ya != 0:
                        C2.add(point_to_idx[((xa,xb),(ya,yb))])
    
    # Seed 3: Random subset
    random.seed(42)
    C3 = set(random.sample(range(n_points), 300))
    
    # Seed 4: Points with ya=0
    C4 = set()
    for xa in range(p):
        for xb in range(p):
            for yb in range(p):
                C4.add(point_to_idx[((xa,xb),(0,yb))])
    
    # Seed 5: Points with ya in {0,1}
    C5 = set()
    for xa in range(p):
        for xb in range(p):
            for ya in [0, 1]:
                for yb in range(p):
                    C5.add(point_to_idx[((xa,xb),(ya,yb))])
    
    seeds = [C1, C2, C3, C4, C5]
    
    for C_seed in seeds:
        if not verify_nikodym(C_seed):
            continue
        
        C = set(C_seed)
        
        # Swap-based local search: try to remove points from C (add to N) 
        # while maintaining Nikodym property, OR add points to C (remove from N)
        # to enable further removals
        
        # Actually, we want to MINIMIZE |N| = MAXIMIZE |C|
        # So we want to ADD points to C while maintaining validity
        
        # But also try swaps: remove a point from C and add a different one
        
        improved = True
        while improved and (time.time() - start_time) < deadline - 10:
            improved = False
            
            # Try adding each point to C
            for pt_idx in range(n_points):
                if pt_idx in C:
                    continue
                
                # Check if adding pt_idx to C maintains Nikodym property
                C_test = set(C)
                C_test.add(pt_idx)
                
                # Quick check: for each point, is there still a valid line?
                # A point x is "covered" if there's a line through x with no C points other than possibly x
                # Adding pt_idx to C breaks coverage for points that only had lines through pt_idx
                # But it also might break coverage for pt_idx itself and others
                
                # Full verification is expensive, so do a quick check first
                # For pt_idx to be addable, we need:
                # 1. For every point x != pt_idx, if all valid lines through x contained pt_idx, 
                #    then x must have another valid line
                # 2. pt_idx itself must still have a valid line (which it will, since 
                #    any line through pt_idx with no other C points works, but now pt_idx is in C)
                
                # Actually, let's just do full verification for simplicity
                if verify_nikodym(C_test):
                    C = C_test
                    improved = True
                    break
            
            if not improved and (time.time() - start_time) < deadline - 10:
                # Try swaps: remove one point from C, add another
                # This is more expensive, so do it less frequently
                for _ in range(50):
                    if (time.time() - start_time) > deadline - 10:
                        break
                    
                    # Pick a random point to remove from C
                    if not C:
                        break
                    remove_pt = random.choice(list(C))
                    
                    # Try adding a random point not in C
                    available = [i for i in range(n_points) if i not in C]
                    if not available:
                        break
                    add_pt = random.choice(available)
                    
                    C_test = set(C)
                    C_test.discard(remove_pt)
                    C_test.add(add_pt)
                    
                    if len(C_test) > len(C):
                        continue  # No improvement
                    
                    if verify_nikodym(C_test):
                        C = C_test
                        improved = True
                        break
        
        if len(C) > best_size:
            best_C = set(C)
            best_size = len(C)
    
    # If no valid seed was found, use a fallback
    if best_C is None:
        # Fallback: use the entire plane minus one point (trivially valid)
        best_C = set(range(n_points - 1))
        # Actually, let's just use a large valid set
        # The whole plane is not valid, but almost all points should work
        # Let's use C = all points except those on a specific line
        # Actually, simplest valid N: N = all points. Then C is empty.
        # But we want to maximize C.
        # Let's just return N = all points (C = empty) as fallback
        best_C = set()
    
    # Convert best_C to array
    N_points = []
    for i, pt in enumerate(all_points):
        if i not in best_C:
            xa, xb = pt[0]
            ya, yb = pt[1]
            N_points.append([[xa, xb], [ya, yb]])
    
    return np.array(N_points, dtype=np.int64)
