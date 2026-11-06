import time
import random
import sys
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 135  # Safety margin for 150s limit

    # Helper: normalize direction vector (dx, dy)
    def norm_dir(dx, dy):
        if dx == 0 and dy == 0:
            return (0, 0)
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        return (dx, dy)

    # Check validity: no 4 collinear
    def is_valid(points):
        n = len(points)
        # For each pair, count points on the line
        # O(n^3) but with small constant. n=20 -> 20*19/2 * 18 ~ 3420 ops
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        count += 1
                        if count >= 4:
                            return False
        return True

    # Get all collinear triples
    def get_triples(points):
        n = len(points)
        triples = set()
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                line_pts = [i, j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        line_pts.append(k)
                if len(line_pts) == 3:
                    triples.add(tuple(sorted(line_pts)))
        return triples

    # Calculate score: Largest subset with no 3 collinear
    # This is equivalent to: N - (Minimum Vertex Cover of the 3-uniform hypergraph of collinear triples)
    # Let's use a Branch and Bound for Minimum Vertex Cover.
    # Vertices: 0..19. Edges: Triples.
    
    def calc_score(points):
        triples = get_triples(points)
        if not triples:
            return 20
        
        # Convert to bitmask representation for speed
        # Edge is a tuple of 3 indices
        # We want to find min k such that there is a set of k vertices covering all edges.
        # This is NP-hard, but N=20 is small enough for B&B with good heuristics.
        
        # Preprocess: Remove vertices with degree 0? No, they don't affect cover size directly 
        # except that they are free to be in the independent set.
        # Actually, we want to MAXIMIZE the independent set.
        # Let's use a recursive search to find the max independent set.
        # Or min vertex cover.
        
        # Let's use a heuristic-guided branch and bound for Min Vertex Cover.
        # State: remaining edges, current cover size.
        # We can use a simple greedy lower bound: 
        # While edges remain, pick vertex with max degree in remaining graph, add to cover, remove incident edges.
        # This gives a valid cover size, which is an upper bound on the optimal cover size.
        # Wait, we want to MINIMIZE cover size. So we need a lower bound to prune.
        # Lower bound: ceil(num_remaining_edges / max_degree_in_remaining_graph)?
        # Or simply: num_remaining_edges / 3 (since each vertex covers at most 3 edges? No, degree can be high).
        
        # Let's try a different approach: Branch on a vertex.
        # For a specific edge (u, v, w), at least one of u, v, w must be in the cover.
        # So we branch into 3 cases: u in cover, v in cover, w in cover.
        
        best_cover = 20  # Upper bound
        
        # Sort vertices by degree for better branching?
        # Let's just use the edges directly.
        
        def solve(edges, current_size):
            nonlocal best_cover
            if current_size >= best_cover:
                return
            
            if not edges:
                if current_size < best_cover:
                    best_cover = current_size
                return
            
            # Heuristic: Pick the edge with the "most constrained" vertices?
            # Or just pick the first edge.
            # To improve, we can sort edges by the sum of degrees of their vertices in the remaining graph.
            # Let's compute degrees for the current subgraph.
            deg = {i: 0 for i in range(20)}
            for e in edges:
                for v in e:
                    deg[v] += 1
            
            # Pick edge with max sum of degrees
            edge = max(edges, key=lambda e: deg[e[0]] + deg[e[1]] + deg[e[2]])
            u, v, w = edge
            
            # Branch 1: Include u
            new_edges_1 = set()
            for e in edges:
                if u not in e:
                    new_edges_1.add(e)
            solve(new_edges_1, current_size + 1)
            
            # Branch 2: Include v
            new_edges_2 = set()
            for e in edges:
                if v not in e:
                    new_edges_2.add(e)
            solve(new_edges_2, current_size + 1)
            
            # Branch 3: Include w
            new_edges_3 = set()
            for e in edges:
                if w not in e:
                    new_edges_3.add(e)
            solve(new_edges_3, current_size + 1)
            
        solve(triples, 0)
        return 20 - best_cover

    def evaluate(points):
        if not is_valid(points):
            return 100
        return calc_score(points)

    # Initial Configuration
    # Strategy: Start with a "twisted" grid or random, then refine.
    # A known good structure for no 4-collinear is a parabola y = x^2, but we need 20 points.
    # x from 0 to 19, y = x^2. Max y = 361 > 100.
    # We can scale: x in [-5, 5] (11 pts), y = x^2.
    # We need 20 points.
    
    def get_initial_points():
        points = []
        # Try to place points on a few lines to create many triples?
        # No, we want to MINIMIZE the largest no-3-collinear subset.
        # This means we want to MAXIMIZE the number of collinear triples.
        # So we want a configuration with many lines of exactly 3 points.
        
        # Random start
        while len(points) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if [x, y] not in points:
                points.append([x, y])
        return points

    # Local Search
    best_points = get_initial_points()
    best_score = evaluate(best_points)
    
    # Ensure valid start
    while not is_valid(best_points):
        best_points = get_initial_points()
    best_score = evaluate(best_points)
    
    # Pre-compute some candidate moves?
    # Just random moves.
    
    while time.time() - start_time < timeout:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = best_points[idx]
        
        improved = False
        # Try multiple random moves for this point
        for _ in range(20):
            # Bias towards "interesting" locations?
            # Maybe near other points to form triples?
            # For now, uniform random in [-100, 100]
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            if new_x == old_x and new_y == old_y:
                continue
            if [new_x, new_y] in best_points:
                continue
            
            candidate = [p[:] for p in best_points]
            candidate[idx] = [new_x, new_y]
            
            if not is_valid(candidate):
                continue
            
            score = evaluate(candidate)
            if score < best_score:
                best_score = score
                best_points = candidate
                improved = True
                break
            elif score == best_score and random.random() < 0.05:
                # Simulated annealing style acceptance
                best_points = candidate
                improved = True
                break
        
        if not improved and random.random() < 0.01:
            # Restart
            new_points = get_initial_points()
            while not is_valid(new_points):
                new_points = get_initial_points()
            new_score = evaluate(new_points)
            if new_score < best_score:
                best_points = new_points
                best_score = new_score

    return best_points

if __name__ == "__main__":
    construct()
