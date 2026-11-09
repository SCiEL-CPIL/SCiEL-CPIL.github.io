"""
Triangular lattice packing of 32 unit pennies.
Goal: maximize number of unit-distance edges (target 76 via Harborth).

Strategy:
1. Generate a known optimal spiral configuration (hexagonal spiral).
2. Verify it achieves 76 edges.
3. If not, perform local search (swap-based) to improve.
4. Return the best configuration found.
"""
import math
import time
import random
from collections import deque

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Triangular lattice: point (a, b) represents a + b*omega, omega = exp(i*pi/3)
    # Distance squared between (a1,b1) and (a2,b2):
    # dx = a1-a2, dy = b1-b2
    # |dx + dy*omega|^2 = dx^2 + dy^2 + dx*dy  (since |omega|=1, Re(omega)=1/2)
    # Actually: (a1-a2 + (b1-b2)*omega) magnitude squared
    # = (a1-a2)^2 + (b1-b2)^2 + (a1-a2)*(b1-b2) * 2*Re(omega)
    # = (a1-a2)^2 + (b1-b2)^2 + (a1-a2)*(b1-b2)
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db
    
    def count_edges(points):
        """Count pairs at distance exactly 1."""
        n = len(points)
        edges = 0
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    edges += 1
        return edges
    
    def is_valid(points):
        """Check all pairs have distance >= 1."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) < 1 - 1e-9:
                    return False
        return True
    
    # Hexagonal spiral construction for 32 points
    # The hexagonal spiral goes: center, then ring 1 (6 pts), ring 2 (12 pts), ring 3 (18 pts)...
    # Ring k has 6k points.
    # Center: 1
    # Ring 1: 6 -> total 7
    # Ring 2: 12 -> total 19
    # Ring 3: 18 -> total 37 (we need 32, so take first 32)
    
    def hex_spiral(n):
        """Generate n points in hexagonal spiral order on triangular lattice."""
        if n <= 0:
            return []
        points = [[0, 0]]
        if n == 1:
            return points
        
        # Directions for the 6 sides of the hexagon
        # In axial coordinates (a, b) for triangular lattice:
        # The 6 neighbors of (0,0) are:
        # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        ring = 1
        while len(points) < n:
            # Start of ring: go in direction 0 for 'ring' steps from previous ring's start
            # The ring starts at (ring, 0)
            a, b = ring, 0
            for side in range(6):
                da, db = directions[side]
                for step in range(ring):
                    if len(points) >= n:
                        break
                    points.append([a, b])
                    a += da
                    b += db
                if len(points) >= n:
                    break
            ring += 1
        
        return points[:n]
    
    # Try hexagonal spiral
    best_points = hex_spiral(32)
    best_edges = count_edges(best_points)
    best_valid = is_valid(best_points)
    
    # Verify bounds
    if best_valid:
        for p in best_points:
            if abs(p[0]) > 64 or abs(p[1]) > 64:
                best_valid = False
                break
    
    # If spiral doesn't give 76, try local search
    if best_edges < 76:
        random.seed(42)
        
        def local_search(points, max_iters=50000):
            """Swap-based local search to maximize edges while maintaining validity."""
            points = [p[:] for p in points]
            n = len(points)
            current_edges = count_edges(points)
            current_valid = is_valid(points)
            
            # Precompute neighbor positions for potential swaps
            # For each point, consider moving to nearby lattice positions
            # and swapping with another point
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                improved = False
                # Try random swaps
                i, j = random.sample(range(n), 2)
                # Try moving point i to a random nearby position
                da = random.randint(-3, 3)
                db = random.randint(-3, 3)
                if da == 0 and db == 0:
                    continue
                new_pos = [points[i][0] + da, points[i][1] + db]
                
                if abs(new_pos[0]) > 64 or abs(new_pos[1]) > 64:
                    continue
                
                # Check if new_pos is already occupied
                occupied = set(tuple(p) for p in points)
                if tuple(new_pos) in occupied:
                    continue
                
                # Tentatively move point i to new_pos
                old_pos = points[i][:]
                points[i] = new_pos
                
                # Check validity
                valid = True
                for k in range(n):
                    if k == i:
                        continue
                    if dist_sq(points[i], points[k]) < 1 - 1e-9:
                        valid = False
                        break
                
                if not valid:
                    points[i] = old_pos
                    continue
                
                new_edges = count_edges(points)
                if new_edges > current_edges:
                    current_edges = new_edges
                    improved = True
                else:
                    points[i] = old_pos
                
                if not improved and it % 1000 == 0:
                    # Occasionally try a different strategy
                    pass
            
            return points, current_edges, is_valid(points)
        
        # Run local search with multiple restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Start from spiral with small random perturbations
            perturbed = [p[:] for p in best_points]
            for _ in range(restart * 2):
                idx = random.randint(0, 31)
                da = random.randint(-1, 1)
                db = random.randint(-1, 1)
                new_pos = [perturbed[idx][0] + da, perturbed[idx][1] + db]
                if abs(new_pos[0]) <= 64 and abs(new_pos[1]) <= 64:
                    occupied = set(tuple(p) for p in perturbed)
                    if tuple(new_pos) not in occupied:
                        # Check validity
                        valid = True
                        for k in range(32):
                            if k != idx and dist_sq(new_pos, perturbed[k]) < 1 - 1e-9:
                                valid = False
                                break
                        if valid:
                            perturbed[idx] = new_pos
            
            result, edges, valid = local_search(perturbed, max_iters=20000)
            if valid and edges > best_edges:
                best_edges = edges
                best_points = result
                best_valid = True
            
            if best_edges >= 76:
                break
    
    # Final check
    if not best_valid:
        # Fallback: use the simple strip
        best_points = [[a, b] for a in range(16) for b in range(2)]
        best_edges = count_edges(best_points)
    
    # Ensure exactly 32 distinct points
    seen = set()
    final = []
    for p in best_points:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            final.append([p[0], p[1]])
    
    # If we have fewer than 32, add more
    while len(final) < 32:
        for a in range(-64, 65):
            for b in range(-64, 65):
                t = (a, b)
                if t not in seen:
                    # Check distance to all existing points
                    valid = True
                    for p in final:
                        if dist_sq([a, b], p) < 1 - 1e-9:
                            valid = False
                            break
                    if valid:
                        final.append([a, b])
                        seen.add(t)
                        break
            if len(final) >= 32:
                break
    
    return final[:32]
