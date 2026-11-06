import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 150s limit

    def elapsed():
        return time.time() - start_time

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        # Canonical form
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def is_valid(pts):
        """Check if no 4 points are collinear."""
        n = len(pts)
        if n < 4:
            return True
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None:
                    continue
                line_counts[key] = line_counts.get(key, 0) + 1
                if line_counts[key] > 1:
                    # We have at least 2 pairs on this line.
                    # If we have 2 pairs, we have at least 3 points.
                    # If we have a 3rd pair, we have 4 points?
                    # Pairs: (1,2), (1,3) -> 3 points.
                    # Pairs: (1,2), (3,4) -> 4 points distinct? No, same line.
                    # If count > 1, it means there are at least 3 points on this line.
                    # We need to check if there are 4.
                    # Let's just count points on the line directly if count > 1.
                    A, B, C = key
                    count = 0
                    for k in range(n):
                        x3, y3 = pts[k]
                        if A * x3 + B * y3 + C == 0:
                            count += 1
                    if count >= 4:
                        return False
        return True

    def calculate_score(pts):
        """
        Exact Branch and Bound for Maximum Independent Set in 3-uniform hypergraph.
        Vertices: 0..19
        Edges: Triples (i, j, k) that are collinear.
        """
        n = len(pts)
        if n < 3:
            return n

        # Precompute triples
        triples = []
        # Map point index to list of triples containing it
        point_triples = [[] for _ in range(n)]
        
        # To speed up triple detection, use line keys
        line_to_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in line_to_points:
                    line_to_points[key] = []
                line_to_points[key].append(i)
                line_to_points[key].append(j)
                # Note: This adds duplicates. Better to build lines properly.
        
        # Re-do line building properly
        line_to_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in line_to_points:
                    line_to_points[key] = set()
                line_to_points[key].add(i)
                line_to_points[key].add(j)
                
        # Generate triples from lines with >= 3 points
        for key, points_on_line in line_to_points.items():
            if len(points_on_line) >= 3:
                pts_list = list(points_on_line)
                # Generate all triples from this line
                for a in range(len(pts_list)):
                    for b in range(a + 1, len(pts_list)):
                        for c in range(b + 1, len(pts_list)):
                            t = (pts_list[a], pts_list[b], pts_list[c])
                            # Avoid duplicate triples? 
                            # Since each line is unique, triples from different lines are distinct.
                            # Triples from same line are distinct combinations.
                            triples.append(t)
                            point_triples[pts_list[a]].append(t)
                            point_triples[pts_list[b]].append(t)
                            point_triples[pts_list[c]].append(t)
        
        # Remove duplicates in triples (just in case, though lines are disjoint in terms of defining the triple set? 
        # Actually, a triple belongs to exactly one line. So no duplicates.)
        
        # Order vertices by degree (number of triples) descending
        degrees = [len(point_triples[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: degrees[i], reverse=True)
        
        # Map original index to position in order? 
        # No, we iterate through 'order' in the B&B.
        # We need to check if a point is in the current mask.
        
        best_score = 0
        
        # Precompute for fast conflict checking:
        # For each vertex v, and each other vertex u, do they share a triple with any w?
        # Actually, in B&B, when we add v, we need to know which other vertices u are "blocked" 
        # because there exists w in current_mask such that (v, u, w) is a triple.
        
        # Let's store triples as sets of indices for fast lookup?
        # Or, for each v, store a list of (u, w) pairs?
        # Better: For each v, store a dictionary: u -> list of w's such that (v, u, w) is a triple.
        # When v is added, for each u, if any w in current_mask is in the list, u is blocked.
        
        # Structure: conflict_map[v][u] = set of w's
        conflict_map = [dict() for _ in range(n)]
        for (i, j, k) in triples:
            conflict_map[i][j] = conflict_map[i].get(j, set())
            conflict_map[i][j].add(k)
            conflict_map[i][k] = conflict_map[i].get(k, set())
            conflict_map[i][k].add(j)
            
            conflict_map[j][i] = conflict_map[j].get(i, set())
            conflict_map[j][i].add(k)
            conflict_map[j][k] = conflict_map[j].get(k, set())
            conflict_map[j][k].add(i)
            
            conflict_map[k][i] = conflict_map[k].get(i, set())
            conflict_map[k][i].add(j)
            conflict_map[k][j] = conflict_map[k].get(j, set())
            conflict_map[k][j].add(i)

        def bbb(step, current_mask, current_size, blocked_mask):
            nonlocal best_score
            if elapsed() > deadline - 5:
                return
            
            if current_size > best_score:
                best_score = current_size
                
            if step == n:
                return
                
            # Pruning: Even if we pick all remaining unblocked points, can we beat best?
            # Remaining points are order[step:]
            # But some might be blocked.
            # Simple bound: current_size + (n - step) <= best_score
            if current_size + (n - step) <= best_score:
                return
                
            idx = order[step]
            
            # Option 1: Exclude idx
            bbb(step + 1, current_mask, current_size, blocked_mask)
            
            # Option 2: Include idx
            # Check if idx is already blocked
            if blocked_mask & (1 << idx):
                return
                
            # Check if idx conflicts with current_mask
            # This is handled by blocked_mask? 
            # Wait, blocked_mask is calculated *before* this call?
            # No, blocked_mask should represent vertices that cannot be added 
            # due to conflicts with vertices *already in current_mask*.
            
            # If idx is not blocked, we can add it.
            new_mask = current_mask | (1 << idx)
            
            # Calculate new blocked_mask for the next step
            # Vertices that conflict with idx AND have a partner in current_mask 
            # (or idx itself? No, idx is in the set now)
            # A vertex u is blocked if there exists v in current_mask such that (idx, u, v) is a triple.
            # Note: current_mask does NOT include idx yet in the check, but will in the next state.
            # Actually, if we add idx, then for any u, if there is a triple (idx, u, v) where v is in current_mask, 
            # then u cannot be added later.
            
            new_blocked = blocked_mask
            # Also, if there is a triple (idx, u, v) where v is in current_mask, u is blocked.
            for u, vs in conflict_map[idx].items():
                # If any v in vs is in current_mask, u is blocked
                # Check intersection of vs and current_mask
                # vs is a set of indices. Convert to mask?
                # For speed, precompute masks for vs? 
                # Since n=20, iterating is fast.
                for v in vs:
                    if current_mask & (1 << v):
                        new_blocked |= (1 << u)
                        break
                        
            # Also, idx itself is now in the set, so it's "used". 
            # The blocked_mask logic usually prevents adding vertices that form a hyperedge with 2 existing vertices.
            # It does not prevent adding a vertex that forms a hyperedge with 1 existing vertex and 1 future vertex.
            # That's correct.
            
            bbb(step + 1, new_mask, current_size + 1, new_blocked)

        bbb(0, 0, 0, 0)
        return best_score

    def generate_random_valid():
        """Generate a random set of 20 points with no 4 collinear."""
        while True:
            pts = []
            while len(pts) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = (x, y)
                if p not in pts:
                    pts.append(p)
            if is_valid(pts):
                return pts

    def local_search(pts, max_iters=5000):
        """
        Simulated Annealing / Hill Climbing.
        """
        current_pts = [tuple(p) for p in pts]
        current_score = calculate_score(current_pts)
        best_pts = current_pts[:]
        best_score = current_score
        
        T = 10.0
        cool = 0.999
        
        for i in range(max_iters):
            if elapsed() > deadline - 10:
                break
                
            # Generate move
            idx = random.randint(0, 19)
            old_p = current_pts[idx]
            
            # 70% local move, 30% global move
            if random.random() < 0.7:
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                new_p = (old_p[0] + dx, old_p[1] + dy)
            else:
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
                
            if new_p in current_pts:
                continue
            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                continue
                
            new_pts = current_pts[:]
            new_pts[idx] = new_p
            
            if not is_valid(new_pts):
                continue
                
            new_score = calculate_score(new_pts)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_pts = new_pts
                current_score = new_score
                
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts[:]
            
            T *= cool
            
        return best_pts, best_score

    # Main loop
    best_points = None
    best_val = float('inf')
    
    # Initial candidate
    pts = generate_random_valid()
    best_points, best_val = local_search(pts, max_iters=2000)
    
    # Multiple restarts
    restarts = 0
    while elapsed() < deadline - 20:
        pts = generate_random_valid()
        res_pts, res_score = local_search(pts, max_iters=1000)
        
        if res_score < best_val:
            best_val = res_score
            best_points = res_pts
        
        restarts += 1
        if restarts % 10 == 0:
            # Maybe perturb the best found so far?
            pass

    return [[x, y] for x, y in best_points]

if __name__ == "__main__":
    pts = construct()
    print(pts)
