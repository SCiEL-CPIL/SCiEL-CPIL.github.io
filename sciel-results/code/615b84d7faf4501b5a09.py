import time
import random
import math
from functools import lru_cache

DEADLINE = 115.0

def construct():
    start_time = time.time()
    random.seed(42)
    
    def are_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1])
    
    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            slopes = {}
            xi, yi = points[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                if slope in slopes:
                    slopes[slope] += 1
                    if slopes[slope] >= 3:
                        return True
                else:
                    slopes[slope] = 1
        return False
    
    def calc_score(points):
        """
        Fast B&B for Maximum Independent Set in 3-uniform hypergraph.
        Returns size of largest subset with no 3 collinear.
        """
        n = len(points)
        # Build hypergraph
        # edges: list of frozensets of vertices
        # adj: for each vertex, list of edge indices
        # We can represent the hypergraph by the set of forbidden triples.
        
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        
        if not triples:
            return n
            
        # Map vertex to list of triple indices it belongs to
        vertex_to_triples = [[] for _ in range(n)]
        for idx, (i, j, k) in enumerate(triples):
            vertex_to_triples[i].append(idx)
            vertex_to_triples[j].append(idx)
            vertex_to_triples[k].append(idx)
            
        best = [0]
        
        # State:
        # available: bitmask of vertices that can still be chosen
        # removed_triples: bitmask of triples that have been "satisfied" (broken)
        # Actually, easier:
        # We want to choose a set S such that no triple is fully contained in S.
        # This is equivalent to: For every triple {i,j,k}, at least one of i,j,k is NOT in S.
        # So S is an Independent Set.
        
        # Recursive search
        # State:
        # - mask: bitmask of vertices currently in S (chosen)
        # - active_mask: bitmask of vertices still available to choose
        # - We need to ensure that for any triple, not all 3 are in S.
        #   This is hard to check incrementally if we just add vertices.
        #   Alternative:
        #   When we add vertex v to S, we must remove all other vertices that form a triple with v and another vertex in S?
        #   No, if v, u, w are a triple, and we add v, we can still add u, but then we CANNOT add w.
        #   So if we have S, and we consider adding v:
        #   New S' = S U {v}.
        #   This is valid if no triple is fully contained in S'.
        #   This happens if there is no triple {v, u, w} such that u, w in S.
        #   So, v is "blocked" if there exists u, w in S such that {v, u, w} is a triple.
        
        # Precompute: for each pair (u, w), which vertices v make {v, u, w} a triple?
        # pair_to_vertex: dict[(u, w)] -> list of v
        # Since triples are unique, for a pair (u,w), there can be multiple v's.
        # Actually, if u,w are fixed, the line is fixed. Any other point on that line is a v.
        
        # Let's use a different B&B strategy:
        # Branch on a vertex v.
        # Option 1: Include v.
        #   This forbids any vertex x such that {v, x, y} is a triple for some y in S?
        #   No. If we include v, we just cannot include a pair (x, y) later if {v, x, y} is a triple.
        #   This dependency is complex.
        
        # Simpler B&B:
        # Pick a triple {i, j, k} that is "active" (none of its vertices are excluded, and not all included? No).
        # Actually, let's stick to the vertex branching but with a strong pruning rule.
        # Pruning Rule:
        # If we have chosen S, and available A.
        # Any vertex v in A is "dangerous" if it forms a triple with two vertices in S.
        # If v is dangerous, we CANNOT include v.
        # So we can remove all dangerous vertices from A immediately.
        
        # Let's implement this.
        
        # Precompute: for each pair (u, v), list of w such that {u,v,w} is a triple.
        # pair_to_thirds = {}
        # for (i,j,k) in triples:
        #    pair_to_thirds[(i,j)].append(k)
        #    ...
        
        # This might be slow to build if many triples, but n=20 is small.
        
        # Let's try a simpler approach first:
        # Use the existing B&B but with better ordering and a tighter bound.
        # The existing B&B in the prompt is actually okay, but maybe slow due to Python overhead.
        
        # Let's try to optimize the existing one or use a C-like loop structure if possible? No, pure Python.
        
        # Alternative: Use `networkx`? Not allowed.
        
        # Let's refine the existing B&B in the prompt.
        # The prompt's B&B:
        # 1. Sort vertices by triple count.
        # 2. Backtrack include/exclude.
        # 3. Prune if current_size + remaining <= best.
        
        # Improvement:
        # When we include a vertex v, we can immediately determine which other vertices are "blocked" from being included together with any other vertex?
        # No.
        
        # Let's try a different formulation: Minimum Vertex Cover in Hypergraph.
        # We want to find the smallest set C such that every triple intersects C.
        # Then Score = n - |C|.
        # Minimizing Score is Maximizing |C|? No.
        # Score = |W| where W is Max Independent Set.
        # We want to minimize Score.
        # So we want to find a config where Max Independent Set is small.
        
        # The B&B should find the Max Independent Set.
        
        # Let's optimize the B&B by using bitmasks for the "forbidden" checks.
        
        def max_independent_set(points):
            n = len(points)
            triples = []
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if are_collinear(points[i], points[j], points[k]):
                            triples.append((i, j, k))
            
            if not triples:
                return n
                
            # For each vertex, store bitmask of vertices that form a triple with it and another vertex?
            # Let's use the "blocked" logic.
            # State:
            # chosen_mask: vertices in S
            # avail_mask: vertices not yet decided
            # 
            # A vertex v in avail_mask is "blocked" if there exist u, w in chosen_mask such that {v, u, w} is a triple.
            # If v is blocked, we cannot add it to S.
            
            # Precompute:
            # triple_mask[v] : bitmask of vertices w such that there exists u with {v, w, u} triple?
            # No, we need pairs.
            
            # Let's precompute `conflict_pair_mask`:
            # For each pair (u, v), `conflict_pair_mask[(u,v)]` is the bitmask of all w such that {u,v,w} is a triple.
            # If u and v are in S, then NO w in conflict_pair_mask[(u,v)] can be in S.
            
            conflict_pair_mask = {}
            for (i, j, k) in triples:
                # Add k to mask for (i,j)
                if (i, j) not in conflict_pair_mask: conflict_pair_mask[(i, j)] = 0
                conflict_pair_mask[(i, j)] |= (1 << k)
                
                if (j, i) not in conflict_pair_mask: conflict_pair_mask[(j, i)] = 0
                conflict_pair_mask[(j, i)] |= (1 << k)
                
                # And so on for all 3 pairs
                if (i, k) not in conflict_pair_mask: conflict_pair_mask[(i, k)] = 0
                conflict_pair_mask[(i, k)] |= (1 << j)
                
                if (k, i) not in conflict_pair_mask: conflict_pair_mask[(k, i)] = 0
                conflict_pair_mask[(k, i)] |= (1 << j)
                
                if (j, k) not in conflict_pair_mask: conflict_pair_mask[(j, k)] = 0
                conflict_pair_mask[(j, k)] |= (1 << i)
                
                if (k, j) not in conflict_pair_mask: conflict_pair_mask[(k, j)] = 0
                conflict_pair_mask[(k, j)] |= (1 << i)

            best = [0]
            
            def backtrack(chosen_mask, avail_mask, current_size):
                # Prune
                if current_size + bin(avail_mask).count('1') <= best[0]:
                    return
                
                # If no available vertices, update best
                if avail_mask == 0:
                    if current_size > best[0]:
                        best[0] = current_size
                    return
                
                # Calculate blocked vertices
                # A vertex v is blocked if there is a pair (u, w) in chosen_mask such that v is in conflict_pair_mask[(u,w)]
                # We can iterate through pairs in chosen_mask? That's O(k^2).
                # Or, we can maintain a `blocked_mask` incrementally.
                # When we add v to chosen_mask, we update blocked_mask.
                # Let's pass blocked_mask as a parameter.
                
                # If blocked_mask is passed:
                # avail_mask &= ~blocked_mask
                # If avail_mask becomes 0, update best.
                
                # To keep it simple and avoid complex state passing, let's just compute it here.
                # n=20, so chosen_mask has at most 20 bits.
                # Number of pairs in chosen_mask is <= 190.
                # This is fast enough.
                
                blocked_mask = 0
                # Get list of vertices in chosen_mask
                chosen_vertices = [i for i in range(n) if (chosen_mask >> i) & 1]
                for i in range(len(chosen_vertices)):
                    u = chosen_vertices[i]
                    for j in range(i+1, len(chosen_vertices)):
                        v = chosen_vertices[j]
                        if (u, v) in conflict_pair_mask:
                            blocked_mask |= conflict_pair_mask[(u, v)]
                
                avail_mask &= ~blocked_mask
                
                if avail_mask == 0:
                    if current_size > best[0]:
                        best[0] = current_size
                    return
                
                # Pick a vertex to branch on
                # Heuristic: Pick vertex with highest degree in the remaining hypergraph?
                # Or just pick the lowest set bit for speed.
                # Let's pick the vertex that appears in the most triples among available?
                # Precomputed degrees?
                # For now, pick lowest bit.
                
                v_bit = avail_mask & -avail_mask
                v = (v_bit.bit_length() - 1)
                
                # Option 1: Include v
                new_chosen = chosen_mask | v_bit
                new_avail = avail_mask ^ v_bit
                backtrack(new_chosen, new_avail, current_size + 1)
                
                # Option 2: Exclude v
                backtrack(chosen_mask, new_avail, current_size)
                
            backtrack(0, (1 << n) - 1, 0)
            return best[0]

        return max_independent_set(points)

    def generate_structured(seed_val):
        random.seed(seed_val)
        patterns = []
        
        # Pattern 1: Perturbed Grid
        # 5x4 grid
        for offset in range(3):
            pts = []
            for i in range(5):
                for j in range(4):
                    x = -40 + i * 20
                    y = -30 + j * 20
                    # Perturb
                    x += random.randint(-3, 3)
                    y += random.randint(-3, 3)
                    x = max(-100, min(100, x))
                    y = max(-100, min(100, y))
                    pts.append((x, y))
            
            # Fix duplicates
            unique = list(dict.fromkeys(pts))
            while len(unique) < 20:
                unique.append((random.randint(-100, 100), random.randint(-100, 100)))
            pts = unique[:20]
            
            # Break 4-collinearities
            # Simple local search to remove 4-collinear
            changed = True
            while changed and has_four_collinear(pts):
                changed = False
                # Find a 4-collinear set
                # This is O(n^4) if done naively, but n=20.
                # We can just pick a random point and move it if it's in a 4-collinear
                # Or just move a random point
                idx = random.randint(0, 19)
                old_p = pts[idx]
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
                if new_p not in pts:
                    pts[idx] = new_p
                    changed = True
            patterns.append(pts)

        return patterns

    def count_triples(points):
        count = 0
        for i in range(20):
            slopes = {}
            xi, yi = points[i]
            for j in range(20):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                if slope in slopes:
                    slopes[slope] += 1
                else:
                    slopes[slope] = 1
            for k in slopes.values():
                if k >= 2:
                    count += k * (k - 1) // 2
        return count

    best_points = None
    best_score = 21
    best_triples = 0
    
    # Try structured initializations
    structured = generate_structured(42)
    for pts in structured:
        if time.time() - start_time > DEADLINE:
            break
        if not has_four_collinear(pts):
            score = calc_score(pts)
            triples = count_triples(pts)
            if score < best_score or (score == best_score and triples > best_triples):
                best_score = score
                best_triples = triples
                best_points = pts[:]
    
    # Also try random initializations
    for trial in range(20):
        if time.time() - start_time > DEADLINE:
            break
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append((x, y))
        if not has_four_collinear(pts):
            score = calc_score(pts)
            triples = count_triples(pts)
            if score < best_score or (score == best_score and triples > best_triples):
                best_score = score
                best_triples = triples
                best_points = pts[:]
    
    if best_points is None:
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
                best_points = pts
                best_score = calc_score(pts)
                best_triples = count_triples(pts)
                break
    
    # Local search
    current_points = best_points[:]
    current_triples = best_triples
    current_score = best_score
    
    T = 5.0
    cooling = 0.995
    
    while time.time() - start_time < DEADLINE:
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        
        if random.random() < 0.7:
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            x = max(-100, min(100, old_p[0] + dx))
            y = max(-100, min(100, old_p[1] + dy))
        else:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
        
        new_p = (x, y)
        if new_p == old_p:
            continue
        if new_p in current_points:
            continue
        
        current_points[idx] = new_p
        
        if has_four_collinear(current_points):
            current_points[idx] = old_p
            continue
        
        new_triples = count_triples(current_points)
        
        if new_triples > current_triples:
            current_triples = new_triples
            if random.random() < 0.1 or new_triples > best_triples:
                exact_score = calc_score(current_points)
                if exact_score < best_score or (exact_score == best_score and new_triples > best_triples):
                    best_score = exact_score
                    best_triples = new_triples
                    best_points = current_points[:]
        elif random.random() < math.exp((new_triples - current_triples) / T):
            current_triples = new_triples
        else:
            current_points[idx] = old_p
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    return [[p[0], p[1]] for p in best_points]
