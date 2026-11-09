import random
import time
import sys
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin against 120s limit

    # --- Geometry Helpers ---
    
    def norm_dir(dx, dy):
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0)
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        return (dx, dy)

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        a, b = norm_dir(dx, dy)
        c = a * x1 + b * y1
        return (a, b, c)

    def has_four_collinear(points):
        n = len(points)
        line_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
        
        for pts in line_points.values():
            if len(pts) >= 4:
                return True
        return False

    # --- Exact Score Calculator (Branch and Bound) ---
    
    def calc_exact_score(points):
        """
        Computes the largest subset with no three collinear.
        Uses bit-parallel B&B.
        """
        n = len(points)
        
        # Precompute for each point i, the bitmask of points j such that 
        # adding i to a set containing j would require checking for a third?
        # Actually, the constraint is: if we pick i and j, we cannot pick any k 
        # such that i,j,k are collinear.
        
        # Let's precompute `conflict[i][j]` = bitmask of points k that are collinear with i and j.
        # If we have a set S, and we consider adding i, we must ensure that for all j in S, 
        # (conflict[i][j] & S) is empty? No, that's for independent sets in graphs.
        
        # For 3-collinear:
        # A set S is valid iff for all i, j, k in S, they are not collinear.
        # Equivalent: For any pair (i, j) in S, no k in S is on line(i, j).
        
        # B&B State:
        # available: bitmask of points we can still add.
        # current_set: bitmask of points already picked.
        # 
        # If we pick point i from available:
        # The new available set must exclude any point k such that there exists j in current_set 
        # where i, j, k are collinear.
        # Also, if we pick i, we must ensure i doesn't form a triple with two existing points?
        # Yes. If i, j1, j2 are in current_set, that's invalid. So we must check validity before adding.
        
        # Better formulation:
        # Let's define `bad[i]` as the set of points that cannot coexist with i? No, that's pairwise.
        # The constraint is hypergraph.
        
        # Let's use this logic:
        # `forbidden_by_current`: bitmask of points that cannot be added to `current_set` 
        # because they would form a triple with two points in `current_set`?
        # No, if we add point `i` to `current_set`, we must ensure:
        # 1. No two points in `current_set` are collinear with `i`.
        # 2. `i` is not already in `current_set`.
        
        # So, for a candidate `i`, it is valid to add if:
        # For all pairs (j, k) in `current_set`, i is NOT on line(j, k).
        
        # This check is O(|S|^2) per candidate. With N=20, max |S| ~ 10-12, this is ~100 ops.
        # Total nodes in B&B tree might be large, but pruning is strong.
        
        # Let's optimize the check:
        # Precompute `line_mask[i][j]`: bitmask of points k collinear with i and j.
        # Note: line_mask[i][j] includes i and j.
        
        line_mask = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                key = get_line_key(points[i], points[j])
                mask = 1 << i | (1 << j)
                for k in range(n):
                    if k == i or k == j: continue
                    if get_line_key(points[i], points[k]) == key:
                        mask |= (1 << k)
                line_mask[i][j] = mask
                line_mask[j][i] = mask

        # B&B
        best_size = 0
        
        def dfs(available, current_mask, current_size):
            nonlocal best_size
            
            # Pruning: even if we pick all available, can we beat best?
            if current_size + bin(available).count('1') <= best_size:
                return
            
            if available == 0:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Get candidates from available
            # To improve pruning, pick the point with highest "conflict" or just iterate.
            # Let's iterate through available bits.
            
            # Optimization: If we pick a point, we must remove from `available` 
            # any point that would form a triple with it and any point in `current_mask`.
            
            # Let's find a good candidate to branch on.
            # Heuristic: pick a point that is in many lines with current set?
            # Or just pick the first available?
            
            # Let's try picking the point that maximizes the reduction of available set?
            # That's expensive. Let's just iterate.
            
            # To avoid symmetry/permutation issues, we can fix the order.
            
            # Iterate through available points
            avail = available
            while avail:
                # Pick the lowest set bit
                lsb = avail & (-avail)
                i = lsb.bit_length() - 1
                avail ^= lsb
                
                # Check if i can be added to current_mask
                # i is valid if for all j in current_mask, there is no k in current_mask 
                # such that i,j,k collinear.
                # Actually, simpler: 
                # If we add i, the new current set is C' = C U {i}.
                # C' is valid iff C is valid (assumed) AND no triple in C' involves i.
                # Triples involving i: {i, j, k} where j,k in C.
                # So we need to check if there exist j, k in C such that i is on line(j, k).
                
                valid = True
                c = current_mask
                while c:
                    lsb_j = c & (-c)
                    j = lsb_j.bit_length() - 1
                    c ^= lsb_j
                    
                    # Check if i is on line(j, k) for any k in C?
                    # No, we need to check if there is ANY k in C such that i, j, k collinear.
                    # We can precompute for each pair (i, j) the mask of k.
                    # If (line_mask[i][j] & current_mask) has any bit set other than i and j?
                    # line_mask[i][j] contains i, j, and all k on the line.
                    # If (line_mask[i][j] & current_mask) != (1<<i | 1<<j), then there is a k in C.
                    # But i is not in current_mask yet.
                    # So we check: (line_mask[i][j] & current_mask) != (1<<j)?
                    # Wait, line_mask[i][j] includes i and j.
                    # current_mask does not include i.
                    # So (line_mask[i][j] & current_mask) should be exactly {j} if valid.
                    # If it contains any other bit k, then i, j, k are collinear.
                    
                    common = line_mask[i][j] & current_mask
                    if common != (1 << j):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # If valid, we can add i.
                # New current_mask: current_mask | (1 << i)
                # New available: 
                # We must remove from `available` any point m such that 
                # adding m to (C U {i}) would be invalid? 
                # No, `available` is just the set of points we *can* consider next.
                # However, if we add i, we permanently exclude any m that forms a triple 
                # with i and some j in C? 
                # Actually, if m forms a triple with i and j (j in C), then m cannot be in the final set.
                # Also, if m forms a triple with i and another point added later, that's handled later.
                # But if m forms a triple with i and j (j in C), m is forbidden.
                
                new_current = current_mask | (1 << i)
                
                # Calculate forbidden points due to i
                # For each j in current_mask, any k in line_mask[i][j] is forbidden.
                forbidden = 0
                c = current_mask
                while c:
                    lsb_j = c & (-c)
                    j = lsb_j.bit_length() - 1
                    c ^= lsb_j
                    forbidden |= line_mask[i][j]
                
                # Also, if we add i, we can't add i again (already handled by removing i from avail)
                forbidden |= (1 << i)
                
                new_available = available & ~forbidden
                
                # Recursive call
                dfs(new_available, new_current, current_size + 1)

        # Start B&B
        # Initial available is all points
        initial_available = (1 << n) - 1
        dfs(initial_available, 0, 0)
        
        return best_size

    # --- Main Search Loop (Simulated Annealing) ---
    
    def random_valid_points():
        while True:
            pts = []
            used = set()
            while len(pts) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in used:
                    used.add((x, y))
                    pts.append((x, y))
            if not has_four_collinear(pts):
                return pts

    # Initialize
    best_points = None
    best_score = 20  # Max possible
    current_points = None
    current_score = 20
    
    # Find a good starting point
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        pts = random_valid_points()
        score = calc_exact_score(pts)
        if best_points is None or score < best_score:
            best_points = pts
            best_score = score
            current_points = pts
            current_score = score

    if current_points is None:
        # Fallback: grid-like structure with perturbations
        pts = []
        for i in range(20):
            x = (i % 5) * 20
            y = (i // 5) * 20
            pts.append((x, y))
        # This has 4 collinear in rows/cols. Need to break.
        # Just return random valid if this fails, but we already tried random.
        return [[p[0], p[1]] for p in random_valid_points()]

    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    while time.time() < deadline - 5 and T > T_min:
        T *= cooling_rate
        
        # Generate neighbor
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        
        # Strategy for new point:
        # 1. Random
        # 2. On a line with 2 other points (to create triple)
        
        if random.random() < 0.5:
            # Random
            while True:
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                if (nx, ny) not in current_points:
                    break
            new_p = (nx, ny)
        else:
            # Targeted: pick 2 other points, put new point on their line
            j, k = random.sample([i for i in range(20) if i != idx], 2)
            p_j = current_points[j]
            p_k = current_points[k]
            
            # Find integer points on line(p_j, p_k)
            x1, y1 = p_j
            x2, y2 = p_k
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
            else:
                step_x = dx // g
                step_y = dy // g
                
                # Find a valid t
                # We want a point not in current_points
                # Try t in [-10, 10] excluding 0, 1
                found = False
                for t in range(-10, 11):
                    if t == 0 or t == 1:
                        continue
                    nx = x1 + t * step_x
                    ny = y1 + t * step_y
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        if (nx, ny) not in current_points:
                            new_p = (nx, ny)
                            found = True
                            break
                if not found:
                    new_p = (random.randint(-100, 100), random.randint(-100, 100))

        # Check validity of new configuration
        new_points = current_points.copy()
        new_points[idx] = new_p
        
        if has_four_collinear(new_points):
            # Reject invalid moves
            continue
        
        # Calculate new score
        # Optimization: if the move didn't change the "structure" much, maybe skip exact calc?
        # But we need exact score for SA.
        # To speed up, we can first check if it's clearly worse/better using a fast proxy?
        # No, just compute. N=20 B&B is fast if pruned well.
        
        new_score = calc_exact_score(new_points)
        
        # Acceptance probability
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            current_points = new_points
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_points = current_points.copy()

    return [[p[0], p[1]] for p in best_points]

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Count:", len(pts))
    print("Has 4 collinear:", has_four_collinear(pts))
    print("Exact Score:", calc_exact_score(pts))

