import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_all_vertices(pose):
        # pose: (N, 3) -> x, y, angle_deg
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(pose[i, 0], pose[i, 1], 1.0, pose[i, 2])
        return verts

    def required_outer_side(verts):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all hexagons given their vertices.
        """
        verts_flat = verts.reshape(-1, 2)
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        # Support functions for hexagon directions
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def check_validity_vectorized(verts):
        """
        Vectorized SAT check for N hexagons.
        Returns True if all pairs are disjoint.
        """
        # verts shape: (N, 6, 2)
        # For each pair (i, j), we need to check 12 axes (6 from i, 6 from j).
        # We can do this in a loop over pairs but vectorize the axis checks, 
        # or fully vectorize if memory permits. 
        # N=12, pairs=66. 66*12 axes. Small enough for a simple loop over pairs 
        # but vectorized axis projection.
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = verts[i]
                p2 = verts[j]
                
                # Edges of p1
                edges1 = np.roll(p1, -1, axis=0) - p1
                # Normals: (-dy, dx)
                normals1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
                norms1 = np.linalg.norm(normals1, axis=1)
                normals1 = normals1 / norms1[:, np.newaxis]
                
                # Project p1 and p2 onto normals1
                proj1_1 = p1 @ normals1.T  # (6, 6)
                proj2_1 = p2 @ normals1.T  # (6, 6)
                
                min1_1 = proj1_1.min(axis=0)
                max1_1 = proj1_1.max(axis=0)
                min2_1 = proj2_1.min(axis=0)
                max2_1 = proj2_1.max(axis=0)
                
                # Check if any axis separates them
                # Separated if max1 < min2 or max2 < min1
                separated = (max1_1 < min2_1 - 1e-5) | (max2_1 < min1_1 - 1e-5)
                
                if np.any(separated):
                    continue # Disjoint, check next pair
                
                # If not separated by p1's axes, check p2's axes
                edges2 = np.roll(p2, -1, axis=0) - p2
                normals2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
                norms2 = np.linalg.norm(normals2, axis=1)
                normals2 = normals2 / norms2[:, np.newaxis]
                
                proj1_2 = p1 @ normals2.T
                proj2_2 = p2 @ normals2.T
                
                min1_2 = proj1_2.min(axis=0)
                max1_2 = proj1_2.max(axis=0)
                min2_2 = proj2_2.min(axis=0)
                max2_2 = proj2_2.max(axis=0)
                
                separated2 = (max1_2 < min2_2 - 1e-5) | (max2_2 < min1_2 - 1e-5)
                
                if np.any(separated2):
                    continue # Disjoint
                
                # If we reach here, they overlap
                return False
        
        return True

    def evaluate(pose):
        """
        Returns (is_valid, side_length).
        """
        verts = get_all_vertices(pose)
        if not check_validity_vectorized(verts):
            return False, float('inf')
        side = required_outer_side(verts)
        return True, side

    def gen_hex_lattice():
        """
        Generate a valid initial hexagonal lattice packing.
        """
        centers = []
        # Hex grid coordinates (axial)
        for q in range(-3, 4):
            for r in range(-3, 4):
                x = np.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                centers.append((x, y))
        
        # Sort by distance to origin
        centers.sort(key=lambda c: c[0]**2 + c[1]**2)
        selected = centers[:12]
        
        pose = np.zeros((12, 3))
        for i, (x, y) in enumerate(selected):
            pose[i, 0] = x
            pose[i, 1] = y
            pose[i, 2] = 0.0
        return pose

    def gen_random():
        pose = np.zeros((12, 3))
        for i in range(12):
            angle = 2 * np.pi * np.random.random()
            r = np.random.uniform(0, 4.0)
            pose[i, 0] = r * np.cos(angle)
            pose[i, 1] = r * np.sin(angle)
            pose[i, 2] = np.random.uniform(0, 60)
        return pose

    def repair_collisions(pose, max_iters=20):
        """
        Simple repair: push overlapping pairs apart.
        """
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            verts = get_all_vertices(pose)
            if check_validity_vectorized(verts):
                return pose, True
            
            # Find first collision
            for i in range(N):
                for j in range(i + 1, N):
                    p1 = verts[i]
                    p2 = verts[j]
                    
                    # Quick check if colliding
                    colliding = True
                    # Re-use vectorized logic for single pair for speed? 
                    # Just use a simple distance check as a proxy for "needs push"
                    # If centers are too close, they likely overlap
                    c1 = np.mean(p1, axis=0)
                    c2 = np.mean(p2, axis=0)
                    dist = np.linalg.norm(c1 - c2)
                    
                    # Minimum distance for non-overlap is roughly 2 * inradius = sqrt(3) ~ 1.732
                    # But with rotation, it can be less. 
                    # Let's do a proper SAT check for this pair only if dist < 2.5
                    if dist < 2.5:
                         # Check SAT for this pair
                         edges1 = np.roll(p1, -1, axis=0) - p1
                         normals1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
                         norms1 = np.linalg.norm(normals1, axis=1)
                         normals1 = normals1 / norms1[:, np.newaxis]
                         proj1_1 = p1 @ normals1.T
                         proj2_1 = p2 @ normals1.T
                         min1_1 = proj1_1.min(axis=0)
                         max1_1 = proj1_1.max(axis=0)
                         min2_1 = proj2_1.min(axis=0)
                         max2_1 = proj2_1.max(axis=0)
                         separated = (max1_1 < min2_1 - 1e-5) | (max2_1 < min1_1 - 1e-5)
                         
                         if not np.any(separated):
                              # Check p2 axes
                              edges2 = np.roll(p2, -1, axis=0) - p2
                              normals2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
                              norms2 = np.linalg.norm(normals2, axis=1)
                              normals2 = normals2 / norms2[:, np.newaxis]
                              proj1_2 = p1 @ normals2.T
                              proj2_2 = p2 @ normals2.T
                              min1_2 = proj1_2.min(axis=0)
                              max1_2 = proj1_2.max(axis=0)
                              min2_2 = proj2_2.min(axis=0)
                              max2_2 = proj2_2.max(axis=0)
                              separated2 = (max1_2 < min2_2 - 1e-5) | (max2_2 < min1_2 - 1e-5)
                              
                              if not np.any(separated2):
                                  colliding = False
                         else:
                             colliding = False
                             
                    if not colliding:
                         # Push apart
                         diff = c2 - c1
                         dist = np.linalg.norm(diff)
                         if dist < 1e-6:
                             diff = np.array([1.0, 0.0])
                             dist = 1.0
                         push_dir = diff / dist
                         # Push by half the overlap estimate
                         pose[i, :2] -= push_dir * 0.1
                         pose[j, :2] += push_dir * 0.1
                         break
                if not colliding:
                    break
            else:
                # No collision found or loop finished
                break
        
        verts = get_all_vertices(pose)
        return pose, check_validity_vectorized(verts)

    def differential_evolution(pop_size=50, generations=100):
        # Initialize population
        pop = [gen_hex_lattice()]
        for _ in range(pop_size - 1):
            pop.append(gen_random())
        
        # Repair and evaluate initial pop
        valid_pops = []
        best_pose = None
        best_side = float('inf')
        
        for i in range(pop_size):
            if time.time() > deadline:
                break
            pose, valid = repair_collisions(pop[i])
            pop[i] = pose
            if valid:
                _, side = evaluate(pose)
                valid_pops.append(pose)
                if side < best_side:
                    best_side = side
                    best_pose = pose.copy()
            else:
                # If invalid, keep it in pop but mark as bad? 
                # DE requires all members to be valid for fitness. 
                # We will re-repair or replace invalid ones with randoms in next gen.
                pass
        
        if best_pose is None:
            # Fallback
            pose = np.zeros((12, 3))
            for i in range(12):
                pose[i, 0] = i * 2.5
                pose[i, 1] = 0
                pose[i, 2] = 0
            best_pose = pose
            best_side = 20.0

        for gen in range(generations):
            if time.time() > deadline:
                break
            
            new_pop = []
            # Elitism
            new_pop.append(best_pose.copy())
            
            # Ensure we have enough valid parents. If not, reuse best or randoms.
            # For simplicity, if valid_pops is small, we just do random mutation of best.
            
            for _ in range(pop_size - 1):
                if time.time() > deadline:
                    break
                
                # Select 3 distinct parents from current population (valid ones preferred)
                # If not enough valid, use any
                candidates = pop if len(valid_pops) < 3 else valid_pops
                if len(candidates) < 3:
                    candidates = [best_pose] * 3
                
                i1, i2, i3 = np.random.choice(len(candidates), 3, replace=False)
                p1 = candidates[i1]
                p2 = candidates[i2]
                p3 = candidates[i3]
                
                # Mutation: p1 + F * (p2 - p3)
                F = 0.8
                child = p1 + F * (p2 - p3)
                
                # Clamp positions to a reasonable box to prevent drift
                # Outer hex side ~ 5-6, so centers should be within ~ 4
                child[:, 0] = np.clip(child[:, 0], -5, 5)
                child[:, 1] = np.clip(child[:, 1], -5, 5)
                child[:, 2] = child[:, 2] % 60
                
                # Crossover: Binomial
                CR = 0.9
                j_rand = np.random.randint(3) # x, y, or angle
                mask = np.random.random((N, 3)) < CR
                mask[:, j_rand] = True # Ensure at least one gene is taken from mutant
                child = np.where(mask, child, p1)
                
                # Repair
                child, valid = repair_collisions(child)
                new_pop.append(child)
                
                if valid:
                    _, side = evaluate(child)
                    if side < best_side:
                        best_side = side
                        best_pose = child.copy()
            
            pop = new_pop
            
            # Update valid_pops list for next generation
            valid_pops = []
            for p in pop:
                # Quick validity check? We already repaired, but repair might fail.
                # Let's assume repaired ones are valid for selection, but verify best.
                valid_pops.append(p)
            
            # Inject diversity occasionally
            if gen % 10 == 0 and gen > 0:
                pop[-1] = gen_random()
                pose, valid = repair_collisions(pop[-1])
                pop[-1] = pose

        return best_pose, best_side

    def local_refinement(pose):
        """
        Use L-BFGS-B to minimize the outer side length while maintaining validity.
        We penalize collisions heavily.
        """
        def objective(x):
            pose = x.reshape(N, 3)
            verts = get_all_vertices(pose)
            
            # Check validity
            if not check_validity_vectorized(verts):
                # Calculate overlap penalty
                # Simple penalty: number of overlapping pairs * large number
                # For speed, just return large number
                return 1000.0
            
            side = required_outer_side(verts)
            return side
        
        x0 = pose.flatten()
        
        # Bounds
        bounds = [(-5, 5), (-5, 5), (0, 60)] * N
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5})
            
            if res.success or res.fun < 100.0:
                refined_pose = res.x.reshape(N, 3)
                refined_pose[:, 2] = refined_pose[:, 2] % 60
                # Final check
                if check_validity_vectorized(get_all_vertices(refined_pose)):
                    return refined_pose
        except Exception:
            pass
            
        return pose

    # Main Execution
    best_positions, best_side = differential_evolution()
    
    # Local refinement on the best found
    if time.time() < deadline - 5.0:
        refined = local_refinement(best_positions)
        refined, valid = repair_collisions(refined)
        if valid:
            _, refined_side = evaluate(refined)
            if refined_side < best_side:
                best_positions = refined
                best_side = refined_side

    # Final check
    if best_positions is None:
        best_positions = gen_hex_lattice()
        best_positions, _ = repair_collisions(best_positions)
        best_side = 20.0

    # Construct output
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_positions[i, 2])])
        
    # The outer hexagon is centered at (0,0) with rotation 0
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
