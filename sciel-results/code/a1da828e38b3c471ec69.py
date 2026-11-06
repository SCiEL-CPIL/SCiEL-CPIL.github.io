import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0
    
    def time_left():
        return deadline - (time.time() - start_time)

    # Target n = 67, the smallest known valid size.
    # If we can't find it, we might try 68, 69, but 67 is the goal.
    # We will focus all effort on n=67.
    
    n = 67
    
    # Initialize random tournament
    # adj[i][j] = 1 if i -> j
    adj = [[0]*n for _ in range(n)]
    out_mask = [0] * n
    
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i][j] = 1
                out_mask[i] |= (1 << j)
            else:
                adj[j][i] = 1
                out_mask[j] |= (1 << i)
                
    # Precompute all 4-sets for faster iteration
    # Note: Storing all 4-sets takes memory but speeds up iteration
    all_quads = list(combinations(range(n), 4))
    
    def is_valid_quad(quad):
        """Check if the quad has a common dominator."""
        a, b, c, d = quad
        quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        for v in range(n):
            if (out_mask[v] & quad_mask) == quad_mask:
                return True
        return False

    def get_violations():
        """Get the set of all violating quads."""
        violations = set()
        for quad in all_quads:
            if not is_valid_quad(quad):
                violations.add(quad)
        return violations

    # Initial violations
    violations = get_violations()
    
    # Helper to update violations after a flip
    # Flipping edge (u, v) changes out_mask[u] and out_mask[v]
    # Affected quads are those containing u or v
    def get_affected_quads(u, v):
        affected = set()
        for quad in all_quads:
            if u in quad or v in quad:
                affected.add(quad)
        return affected

    # To speed up, we can precompute which quads contain which vertex
    quads_containing = {i: [] for i in range(n)}
    for quad in all_quads:
        for v in quad:
            quads_containing[v].append(quad)

    def update_violations_after_flip(u, v, old_out_u, old_out_v, new_out_u, new_out_v):
        """
        Update the global violations set after flipping edge (u, v).
        We must check all quads containing u or v.
        """
        # Get all quads containing u or v
        affected_quads = set(quads_containing[u])
        affected_quads.update(quads_containing[v])
        
        # We need to check these quads against the NEW state
        # But we must be careful: is_valid_quad uses the global out_mask.
        # We have already updated out_mask[u] and out_mask[v] to the new values.
        
        for quad in affected_quads:
            # Check if quad is currently valid
            valid = is_valid_quad(quad)
            if valid:
                violations.discard(quad)
            else:
                violations.add(quad)

    # Main Loop: Simulated Annealing / Greedy Repair
    # We will try to reduce the number of violations.
    
    best_violation_count = len(violations)
    best_state = None
    
    # Store best state (adj, out_mask, violations)
    # We'll just return the first valid one found.
    
    # To avoid getting stuck, we use a temperature-like mechanism or random restarts.
    # Given the time limit, we'll do a single long run with random perturbation if stuck.
    
    stuck_counter = 0
    
    while time_left() > 2.0:
        if not violations:
            break
            
        # Pick a random violation
        quad = random.choice(list(violations))
        
        # Find the best vertex to dominate this quad
        # Score: number of edges already pointing to the quad
        a, b, c, d = quad
        quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        
        best_v = -1
        best_score = -1
        # Check all vertices, or just candidates?
        # Checking all n vertices is O(n) = 67, which is cheap.
        for v in range(n):
            score = (out_mask[v] & quad_mask).bit_count()
            if score > best_score:
                best_score = score
                best_v = v
        
        # If the best vertex already dominates the quad, it's not a violation (contradiction)
        # But since it's in violations, best_score < 4.
        
        # Flip edges from best_v to quad to make best_v dominate quad
        # We need to flip edges (best_v, x) for x in quad if adj[best_v][x] is 0
        # Note: If best_v is IN the quad, we can't dominate itself.
        # A vertex cannot dominate itself. So if best_v is in quad, it can never be the dominator.
        # We must exclude vertices in the quad from being the dominator?
        # "Common dominator" usually implies an external vertex or any vertex?
        # In a tournament, a vertex cannot have an arc to itself.
        # So if the dominator is in the 4-set, it must have arcs to the OTHER 3.
        # Wait, the definition: "Every set of 4 vertices must have a vertex with arcs to all four."
        # If the vertex is IN the set, it has no arc to itself. So it cannot have arcs to "all four" (including itself).
        # Therefore, the dominator MUST be outside the 4-set?
        # Let's re-read: "common dominator". In graph theory, a dominator of a set S is a vertex v such that v->u for all u in S.
        # If v in S, v->v is false. So v cannot be a dominator of S if v in S.
        # So the dominator must be in V \ S.
        
        # Correction: The problem statement says "a vertex with arcs to all four".
        # This implies the vertex is distinct from the four?
        # Usually, "dominator of S" means v \notin S.
        # If v \in S, it can't dominate itself.
        # So we only consider v \notin quad.
        
        candidates = [v for v in range(n) if v not in quad]
        if not candidates:
            continue # Impossible for n>=5
            
        best_v = -1
        best_score = -1
        for v in candidates:
            score = (out_mask[v] & quad_mask).bit_count()
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v == -1:
            continue
            
        # Flip edges
        old_out_u = out_mask[best_v]
        old_out_v = 0 # dummy, we only change best_v's out_mask?
        # Wait, flipping (best_v, x) changes out_mask[best_v] AND out_mask[x].
        # So we must track all changed vertices.
        
        changed_vertices = set()
        changed_vertices.add(best_v)
        
        for x in quad:
            if not adj[best_v][x]:
                # Flip: best_v -> x
                adj[best_v][x] = 1
                adj[x][best_v] = 0
                out_mask[best_v] |= (1 << x)
                out_mask[x] &= ~(1 << best_v)
                changed_vertices.add(x)
            # If adj[best_v][x] is 1, it's already correct, no flip needed.
        
        if not changed_vertices:
            # No flips needed? Then it was already dominated?
            # Should not happen if quad is in violations.
            continue
            
        # Update violations
        # We need to check all quads containing any vertex in changed_vertices
        affected_quads = set()
        for v in changed_vertices:
            affected_quads.update(quads_containing[v])
            
        for quad in affected_quads:
            valid = is_valid_quad(quad)
            if valid:
                violations.discard(quad)
            else:
                violations.add(quad)
                
        # Track progress
        current_count = len(violations)
        if current_count < best_violation_count:
            best_violation_count = current_count
            stuck_counter = 0
        else:
            stuck_counter += 1
            
        # If stuck for too long, random perturbation
        if stuck_counter > 100:
            # Flip a random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                # Flip u->v
                if adj[u][v]:
                    adj[u][v] = 0
                    adj[v][u] = 1
                    out_mask[u] &= ~(1 << v)
                    out_mask[v] |= (1 << u)
                else:
                    adj[u][v] = 1
                    adj[v][u] = 0
                    out_mask[u] |= (1 << v)
                    out_mask[v] &= ~(1 << u)
                
                changed_vertices = {u, v}
                affected_quads = set()
                for v in changed_vertices:
                    affected_quads.update(quads_containing[v])
                    
                for quad in affected_quads:
                    valid = is_valid_quad(quad)
                    if valid:
                        violations.discard(quad)
                    else:
                        violations.add(quad)
                        
                stuck_counter = 0

    # Construct the orientation list
    if not violations:
        orientation = []
        for i in range(n):
            for j in range(i+1, n):
                orientation.append(0 if adj[i][j] == 1 else 1)
        return {'n': n, 'orientation': orientation}
    else:
        # Fallback to Paley 83 if we failed
        p = 83
        squares = set(x * x % p for x in range(1, p))
        adj_paley = [[0]*p for _ in range(p)]
        for i in range(p):
            for j in range(i+1, p):
                if (j - i) % p in squares:
                    adj_paley[i][j] = 1
                else:
                    adj_paley[j][i] = 1
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                orientation.append(0 if adj_paley[i][j] == 1 else 1)
        return {'n': p, 'orientation': orientation}
