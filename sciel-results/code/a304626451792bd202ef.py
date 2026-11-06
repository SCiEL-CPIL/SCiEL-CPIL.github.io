"""Rewritten program: minimize largest no-three-collinear subset of 20 points."""
import random
import itertools
from collections import defaultdict

def construct():
    """
    Construct 20 distinct integer points with |x|,|y| <= 100, no four collinear,
    minimizing the largest subset with no three collinear.
    
    Strategy:
    1. Efficient line counting and triple generation.
    2. Bitmask-based Branch and Bound for Maximum Independent Set.
    3. Simulated Annealing local search to minimize the score.
    """
    random.seed(42)
    
    def has_four_collinear(points):
        """Check if any 4 points are collinear using line hashing."""
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = points[i], points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Normalize direction vector
                g = __import__('math').gcd(abs(dx), abs(dy))
                if g != 0:
                    dx //= g
                    dy //= g
                else:
                    # Handle zero vector (should not happen for distinct points)
                    continue
                # Normalize to canonical form
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                # Line is determined by direction and one point on it.
                # To canonicalize the point, we can use the cross product with the direction.
                # Or simply store (dx, dy, p1) and ensure p1 is the "lowest" point on the line in some order.
                # Easier: For each pair, calculate the line equation ax+by=c.
                # a = dy, b = -dx, c = -(dy*x1 - dx*y1)
                a = dy
                b = -dx
                c = -(dy * p1[0] - dx * p1[1])
                # Normalize (a, b, c)
                g2 = __import__('math').gcd(a, __import__('math').gcd(b, c))
                if g2 != 0:
                    a //= g2
                    b //= g2
                    c //= g2
                if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                    a, b, c = -a, -b, -c
                line_counts[(a, b, c)] += 1
        
        for count in line_counts.values():
            if count >= 6: # C(4,2) = 6 pairs for 4 collinear points
                return True
        return False

    def count_collinear_triples(points):
        """Return list of collinear triples (as tuples of indices)."""
        n = len(points)
        triples = []
        # Map from line key to list of points on that line
        lines = defaultdict(list)
        
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = points[i], points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                a = dy
                b = -dx
                c = -(dy * p1[0] - dx * p1[1])
                # Normalize
                g2 = __import__('math').gcd(a, __import__('math').gcd(b, c))
                if g2 != 0:
                    a //= g2
                    b //= g2
                    c //= g2
                if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                    a, b, c = -a, -b, -c
                lines[(a, b, c)].append((i, j))
                
        # This approach is tricky because a line with k points has C(k,2) pairs.
        # We need to group points by line.
        # Let's restart the triple counting logic more robustly.
        
        triples_set = set()
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = points[i], points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Find all k > j such that p1, p2, pk are collinear
                for k in range(j+1, n):
                    p3 = points[k]
                    dx3 = p3[0] - p1[0]
                    dy3 = p3[1] - p1[1]
                    if dx * dy3 - dy * dx3 == 0:
                        triples_set.add((i, j, k))
        return list(triples_set)

    def max_independent_set_size(points, triples):
        """
        Exact branch and bound using bitmasks.
        """
        n = len(points)
        # Precompute for each vertex v, the set of vertices u such that {u, v, w} is a triple for some w?
        # No, we need: if we have included set S, and we want to add v,
        # we must ensure no triple {v, u, w} has u, w in S.
        
        # Let's precompute for each v, a list of (u, w) pairs such that {v, u, w} is a triple.
        # Actually, for a fixed v, and a fixed u, is there a w?
        # Let's store for each v, a dict: neighbor_pair_mask?
        
        # Simpler: For each v, store a list of triples containing v.
        # When checking if v can be added to S:
        # For each triple T in triples_with_v:
        #   if (T \ {v}) is subset of S: return False
        
        # To speed up subset check, represent S as a bitmask.
        # Precompute for each v, a list of masks M_v where M_v is the bitmask of the other 2 vertices in the triple.
        # Then check: if (S & M_v) == M_v: conflict.
        
        triple_masks = [[] for _ in range(n)]
        for t in triples:
            i, j, k = t
            mask_jk = (1 << j) | (1 << k)
            mask_ik = (1 << i) | (1 << k)
            mask_ij = (1 << i) | (1 << j)
            triple_masks[i].append(mask_jk)
            triple_masks[j].append(mask_ik)
            triple_masks[k].append(mask_ij)
            
        best = 0
        
        def dfs(available_mask, current_mask, current_size):
            nonlocal best
            
            # Pruning: if current_size + popcount(available_mask) <= best, return
            if current_size + bin(available_mask).count('1') <= best:
                return
                
            # Reduction: Remove vertices from available_mask that cannot be added to current_mask
            # A vertex v can be added if for all M in triple_masks[v], (current_mask & M) != M
            new_available = 0
            for v in range(n):
                if not (available_mask & (1 << v)):
                    continue
                can_add = True
                for M in triple_masks[v]:
                    if (current_mask & M) == M:
                        can_add = False
                        break
                if can_add:
                    new_available |= (1 << v)
            
            available_mask = new_available
            
            if available_mask == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Branch on a vertex
            # Heuristic: pick vertex with fewest options? Or just lowest index.
            # Let's pick the first available vertex.
            # Or pick one that appears in most triples among available?
            # For simplicity, pick lowest index.
            v_bit = available_mask & (-available_mask)
            v = v_bit.bit_length() - 1
            
            # Option 1: Exclude v
            dfs(available_mask & ~v_bit, current_mask, current_size)
            
            # Option 2: Include v
            # Since we already checked it can be added (it's in available_mask), we just add it
            dfs(available_mask & ~v_bit, current_mask | v_bit, current_size + 1)
            
        dfs((1 << n) - 1, 0, 0)
        return best

    def score(points):
        """Compute the score: largest subset with no three collinear."""
        triples = count_collinear_triples(points)
        return max_independent_set_size(points, triples)
    
    def is_valid(points):
        """Check no four collinear and all distinct, in bounds."""
        if len(set(points)) != len(points):
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        if has_four_collinear(points):
            return False
        return True
    
    # Generate initial valid configuration
    def generate_initial():
        while True:
            points = []
            seen = set()
            while len(points) < 20:
                # Use a smaller range to increase density of collinearities initially?
                # Or just random.
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                p = (x, y)
                if p not in seen:
                    seen.add(p)
                    points.append(p)
            if not has_four_collinear(points):
                return points

    points = generate_initial()
    current_score = score(points)
    
    best_points = list(points)
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    max_iterations = 20000
    no_improve = 0
    max_no_improve = 2000
    
    # Candidate pool
    candidates = set()
    for _ in range(500):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        candidates.add((x, y))
        
    for it in range(max_iterations):
        if no_improve > max_no_improve:
            break
            
        # Pick a random point to swap
        idx = random.randint(0, 19)
        old_point = points[idx]
        
        # Try swapping with a random candidate
        improved = False
        new_point = random.choice(list(candidates))
        if new_point == old_point:
            continue
            
        new_points = list(points)
        new_points[idx] = new_point
        
        if not is_valid(new_points):
            continue
            
        new_score = score(new_points)
        
        # Acceptance criterion
        if new_score < current_score:
            points = new_points
            current_score = new_score
            improved = True
            if current_score < best_score:
                best_score = current_score
                best_points = list(points)
            no_improve = 0
        else:
            # Accept with probability exp(-delta / T)
            delta = new_score - current_score
            if delta == 0:
                # Accept neutral moves to explore
                if random.random() < 0.5:
                    points = new_points
                    current_score = new_score
                    improved = True
            else:
                prob = __import__('math').exp(-delta / T)
                if random.random() < prob:
                    points = new_points
                    current_score = new_score
                    improved = True
            
            if not improved:
                no_improve += 1
                
        T *= alpha
        if T < T_min:
            T = T_min

    # Return the best points found
    result = [[p[0], p[1]] for p in best_points]
    return result
