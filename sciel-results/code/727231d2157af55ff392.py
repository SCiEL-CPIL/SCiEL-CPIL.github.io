import time
import random
import sys
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    N = 20
    COORD_MIN = -100
    COORD_MAX = 100

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def build_line_map(points):
        n = len(points)
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map

    def is_valid(points):
        line_map = build_line_map(points)
        for line_pts in line_map.values():
            if len(line_pts) >= 4:
                return False
        return True

    def get_collinear_triples(points):
        n = len(points)
        line_map = build_line_map(points)
        triples = []
        for line_pts in line_map.values():
            if len(line_pts) >= 3:
                pts = sorted(line_pts)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        for k in range(j + 1, len(pts)):
                            triples.append((pts[i], pts[j], pts[k]))
        return triples

    def heuristic_score(points):
        return len(get_collinear_triples(points))

    def calculate_exact_score(points):
        n = len(points)
        triples = get_collinear_triples(points)
        
        if not triples:
            return n

        # Build conflict graph: two vertices conflict if they are in a triple
        # We want the largest subset with NO three collinear.
        # This is equivalent to finding the maximum independent set in the 3-uniform hypergraph?
        # No, "no three collinear" means we cannot pick a set of 3 points that are collinear.
        # This is a 3-uniform hypergraph independence problem.
        # However, we can model this as a standard graph problem if we consider the "forbidden" triples.
        # Actually, the standard formulation is: Find max subset S such that no triple in S is collinear.
        
        # Branch and bound on the hypergraph is hard.
        # But note: if we pick 2 points, we forbid the 3rd point of any collinear triple containing them.
        # Let's use a recursive backtracking with pruning.
        
        # Precompute for each pair (i,j), the set of points k such that (i,j,k) is collinear.
        # If we include i and j, we cannot include any k in conflict[i][j].
        
        conflict_pairs = [[set() for _ in range(n)] for _ in range(n)]
        for i, j, k in triples:
            conflict_pairs[i][j].add(k)
            conflict_pairs[j][i].add(k)
            conflict_pairs[i][k].add(j)
            conflict_pairs[k][i].add(j)
            conflict_pairs[j][k].add(i)
            conflict_pairs[k][j].add(i)

        best = [0]
        
        def backtrack(available_mask, current_set, current_size):
            if current_size > best[0]:
                best[0] = current_size
            
            # Upper bound: even if we add all remaining, can we beat best?
            if current_size + available_mask.bit_count() <= best[0]:
                return
            
            # If available is empty, return
            if available_mask == 0:
                return
                
            # Pruning heuristic:
            # If there is a triple (i,j,k) in available, we can't take all 3.
            # A simple bound: the size of a maximal independent set in the conflict graph of triples?
            # Let's use a greedy coloring bound on the "conflict graph" where edges exist between any pair in a triple?
            # No, that's too strong.
            # Let's just do simple branching.
            
            # Pick a vertex to branch on.
            # Heuristic: pick a vertex involved in many triples with other available vertices.
            
            # To speed up, we can use the fact that if we pick a vertex v, we remove v and any vertex that forms a triple with v and another already picked vertex?
            # No, the constraint is global: NO THREE in the final set are collinear.
            # This means for ANY pair (i,j) in the final set, NO k in the final set can be collinear with i and j.
            
            # This is equivalent to: The set of points is a 2-independent set in the hypergraph of collinear triples.
            
            # Let's branch on the first available point.
            v_bit = available_mask & -available_mask
            v = v_bit.bit_length() - 1
            
            # Option 1: Exclude v
            backtrack(available_mask ^ v_bit, current_set, current_size)
            
            # Option 2: Include v
            # If we include v, we must be careful.
            # Actually, including v doesn't immediately forbid others, unless we already have another point u in current_set.
            # If u is in current_set, then any k collinear with u and v is forbidden.
            
            new_available = available_mask ^ v_bit
            for u_bit in (current_set & new_available): # Wait, current_set is a mask?
                 # Let's keep current_set as a mask for efficiency.
                 pass
            
            # Let's rewrite backtrack to use masks for current_set too.
            # backtrack(available_mask, current_mask, current_size)
            
            # If we include v:
            # For every u in current_mask, any k in new_available that is collinear with u and v must be removed.
            # Also, for any pair (u, w) in current_mask, they are already valid.
            # The new constraint is: for any u in current_mask, remove k such that (u,v,k) collinear.
            
            # But wait, if we add v, we might create a triple with two existing points? No, we check when adding.
            # If we add v, we must ensure no triple is formed.
            # A triple is formed if there are u, w in current_mask such that (u, w, v) is collinear.
            # If such u, w exist, we CANNOT add v.
            
            # So, before branching "Include v", we must check if v forms a triple with any pair in current_mask.
            # If it does, "Include v" is invalid.
            
            # Let's precompute for each v, and for each pair (u,w), whether they are collinear.
            # Or simpler: for each v, store the set of pairs (u,w) such that (u,v,w) collinear.
            # If any such pair is a subset of current_mask, v cannot be added.
            
            # Let's precompute:
            # triples_with_v[v] = list of (u, w) such that (u, v, w) is a triple.
            
            # For now, let's do the check inside the recursion for correctness, then optimize.
            
            # Check if v can be added:
            # For all pairs (u, w) in current_mask, is (u, w, v) collinear?
            # If yes, skip "Include v".
            
            can_add = True
            # Iterate through pairs in current_mask
            cm = current_mask
            # Extract list of indices in current_mask
            # Since N=20, we can just loop
            for u in range(n):
                if not (current_mask & (1 << u)):
                    continue
                for w in range(u + 1, n):
                    if not (current_mask & (1 << w)):
                        continue
                    if (u, w, v) in triple_set or (u, v, w) in triple_set or (v, u, w) in triple_set:
                         # Check if collinear
                         # We have triple_set of sorted tuples
                         t = tuple(sorted((u, w, v)))
                         if t in triple_set:
                             can_add = False
                             break
                if not can_add:
                    break
            
            if can_add:
                # Remove any points k that would form a triple with v and some u in current_mask?
                # No, the constraint is on the FINAL set.
                # If we add v, the new set is current_mask | {v}.
                # We must ensure that NO triple is fully contained in new_set.
                # We already checked that no triple is formed with v and 2 from current_mask.
                # What about triples formed with v and 1 from current_mask and 1 from new_available?
                # Those will be handled when we pick the next vertex.
                # So we just add v and recurse.
                backtrack(new_available, current_mask | v_bit, current_size + 1)

        # Precompute triple_set for fast lookup
        triple_set = set()
        for i, j, k in triples:
            triple_set.add(tuple(sorted((i, j, k))))
        
        # We need to pass triple_set to backtrack. In Python, we can use a nonlocal or global, or closure.
        # Let's use a closure variable.
        
        def backtrack_inner(available_mask, current_mask, current_size):
            if current_size > best[0]:
                best[0] = current_size
            
            if current_size + available_mask.bit_count() <= best[0]:
                return
            
            if available_mask == 0:
                return
                
            # Pick a vertex
            # Heuristic: pick vertex with highest degree in the "conflict" sense?
            # Let's just pick the lowest index for simplicity, or random?
            # Random might help escape symmetries.
            # But for exactness, any order works.
            # Let's pick the one that is part of the most triples with other available points?
            # That would be expensive.
            # Let's just pick the first one.
            v_bit = available_mask & -available_mask
            v = v_bit.bit_length() - 1
            
            # Option 1: Exclude v
            backtrack_inner(available_mask ^ v_bit, current_mask, current_size)
            
            # Option 2: Include v
            can_add = True
            # Check if v forms a triple with any pair in current_mask
            for u in range(n):
                if not (current_mask & (1 << u)):
                    continue
                for w in range(u + 1, n):
                    if not (current_mask & (1 << w)):
                        continue
                    t = tuple(sorted((u, w, v)))
                    if t in triple_set:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                backtrack_inner(available_mask ^ v_bit, current_mask | v_bit, current_size + 1)

        backtrack_inner((1 << n) - 1, 0, 0)
        return best[0]

    def generate_random_points():
        pts = []
        while len(pts) < N:
            x = random.randint(COORD_MIN, COORD_MAX)
            y = random.randint(COORD_MIN, COORD_MAX)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        return pts

    # Initial Configuration
    best_config = None
    best_score = 20
    best_heuristic = -1
    
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        pts = generate_random_points()
        if is_valid(pts):
            best_config = pts
            break
            
    if best_config is None:
        return [[x, x * x] for x in range(-10, 10)]

    current_pts = [p[:] for p in best_config]
    current_heuristic = heuristic_score(current_pts)
    
    temp = 1.0
    cooling = 0.995
    step_count = 0
    
    while time.time() < deadline - 5:
        step_count += 1
        idx = random.randint(0, N - 1)
        old_x, old_y = current_pts[idx]
        
        new_x = random.randint(COORD_MIN, COORD_MAX)
        new_y = random.randint(COORD_MIN, COORD_MAX)
        
        if new_x == old_x and new_y == old_y:
            continue
            
        if any(current_pts[i] == [new_x, new_y] for i in range(N) if i != idx):
            continue
            
        current_pts[idx] = [new_x, new_y]
        
        # Check validity
        if not is_valid(current_pts):
            current_pts[idx] = [old_x, old_y]
            continue
            
        new_heuristic = heuristic_score(current_pts)
        
        if new_heuristic > current_heuristic:
            current_heuristic = new_heuristic
        else:
            delta = new_heuristic - current_heuristic
            if random.random() < pow(2.718, delta / temp):
                current_heuristic = new_heuristic
            else:
                current_pts[idx] = [old_x, old_y]
                continue
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
        if current_heuristic > best_heuristic:
            best_heuristic = current_heuristic
            exact = calculate_exact_score(current_pts)
            if exact < best_score:
                best_score = exact
                best_config = [p[:] for p in current_pts]
        elif step_count % 500 == 0:
            exact = calculate_exact_score(current_pts)
            if exact < best_score:
                best_score = exact
                best_config = [p[:] for p in current_pts]
        
    if best_config:
        final_score = calculate_exact_score(best_config)
        if final_score < best_score:
            best_score = final_score
            
    return best_config

# Test
if __name__ == "__main__":
    res = construct()
    print(res)
