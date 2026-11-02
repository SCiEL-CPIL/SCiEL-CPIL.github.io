import numpy as np
import time
import math

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

    def check_validity(verts):
        """
        Check if all hexagons are pairwise disjoint using SAT.
        verts: (N, 6, 2)
        """
        for i in range(N):
            for j in range(i + 1, N):
                p1 = verts[i]
                p2 = verts[j]
                
                disjoint = False
                # Check edges of p1
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p1[j_sat] - p1[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                        disjoint = True
                        break
                
                if not disjoint:
                    # Check edges of p2
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p2[j_sat] - p2[i_sat]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True

    def repair_collisions(pose, max_iters=50):
        """
        Iteratively push colliding hexagons apart.
        """
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            verts = get_all_vertices(pose)
            if check_validity(verts):
                return pose, True
            
            # Find first collision and push apart
            for i in range(N):
                for j in range(i + 1, N):
                    p1 = verts[i]
                    p2 = verts[j]
                    
                    # Check if colliding (simplified SAT check for speed)
                    colliding = True
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p1[j_sat] - p1[i_sat]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10: continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                            colliding = False
                            break
                    if colliding:
                        for i_sat in range(6):
                            j_sat = (i_sat + 1) % 6
                            edge = p2[j_sat] - p2[i_sat]
                            normal = np.array([-edge[1], edge[0]])
                            norm = np.linalg.norm(normal)
                            if norm < 1e-10: continue
                            normal /= norm
                            proj1 = p1.dot(normal)
                            proj2 = p2.dot(normal)
                            if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                                colliding = False
                                break
                    
                    if colliding:
                        # Push apart along center-to-center vector
                        c1 = np.mean(p1, axis=0)
                        c2 = np.mean(p2, axis=0)
                        diff = c2 - c1
                        dist = np.linalg.norm(diff)
                        if dist < 1e-6:
                            diff = np.array([1.0, 0.0])
                            dist = 1.0
                        push_dir = diff / dist
                        # Push each by 0.1 units
                        pose[i, :2] -= push_dir * 0.1
                        pose[j, :2] += push_dir * 0.1
                        break
                if colliding:
                    break
            else:
                # No collision found in this pass?
                break
        
        verts = get_all_vertices(pose)
        return pose, check_validity(verts)

    def evaluate(pose):
        verts = get_all_vertices(pose)
        if not check_validity(verts):
            return float('inf')
        return required_outer_side(verts)

    # Initial Populations
    def gen_hex_lattice():
        # Hexagonal packing of 12 hexagons
        # Center + 6 around + 5 more around
        coords = []
        # Center
        coords.append((0, 0))
        # Ring 1
        for i in range(6):
            angle = np.pi/3 * i
            coords.append((np.sqrt(3)*np.cos(angle), 1.5*np.sin(angle)))
        # Ring 2 (partial)
        # We need 5 more. Let's just pick 5 from the next ring
        for i in range(6):
            angle = np.pi/3 * i + np.pi/6
            # Distance for ring 2 is roughly 2 * distance of ring 1?
            # Actually, let's just use a simple grid and take 12
            pass
        
        # Simpler: Generate a grid of hex centers and take the 12 closest to origin
        centers = []
        for q in range(-5, 6):
            for r in range(-5, 6):
                x = np.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                centers.append((x, y))
        centers.sort(key=lambda c: c[0]**2 + c[1]**2)
        selected = centers[:12]
        
        pose = np.zeros((12, 3))
        for i, (x, y) in enumerate(selected):
            pose[i, 0] = x
            pose[i, 1] = y
            pose[i, 2] = 0.0
        return pose

    def gen_circular():
        pose = np.zeros((12, 3))
        for i in range(12):
            angle = 2 * np.pi * i / 12
            r = 3.0
            pose[i, 0] = r * np.cos(angle)
            pose[i, 1] = r * np.sin(angle)
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

    # Genetic Algorithm
    def genetic_algorithm(pop_size=40, generations=200):
        # Initialize population
        pop = []
        # 2 structured
        pop.append(gen_hex_lattice())
        pop.append(gen_circular())
        # Rest random
        for _ in range(pop_size - 2):
            pop.append(gen_random())
        
        best_pose = None
        best_side = float('inf')
        
        # Repair and evaluate initial pop
        for i in range(pop_size):
            if time.time() > deadline:
                break
            pose, valid = repair_collisions(pop[i])
            if valid:
                side = evaluate(pose)
                if side < best_side:
                    best_side = side
                    best_pose = pose.copy()
            pop[i] = pose
            
        # If no valid found yet, try harder repair or just return best
        if best_pose is None:
             # Fallback: linear arrangement
             pose = np.zeros((12, 3))
             for i in range(12):
                 pose[i, 0] = i * 2.5
                 pose[i, 1] = 0
                 pose[i, 2] = 0
             best_pose = pose
             best_side = evaluate(pose)
             if best_side == float('inf'):
                 best_side = 20.0 # Arbitrary large number
        
        for gen in range(generations):
            if time.time() > deadline:
                break
            
            new_pop = []
            # Elitism: keep best
            new_pop.append(best_pose.copy())
            
            # Selection and Crossover
            # Simple tournament selection
            for _ in range(pop_size - 1):
                if time.time() > deadline:
                    break
                # Select parents
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                p1 = pop[p1_idx]
                p2 = pop[p2_idx]
                
                # Crossover: per-gene
                child = np.zeros((N, 3))
                for i in range(N):
                    if np.random.random() < 0.5:
                        child[i] = p1[i]
                    else:
                        child[i] = p2[i]
                
                # Mutation
                for i in range(N):
                    if np.random.random() < 0.3:
                        child[i, 0] += np.random.normal(0, 0.5)
                        child[i, 1] += np.random.normal(0, 0.5)
                        if np.random.random() < 0.1:
                            child[i, 2] += np.random.uniform(-15, 15)
                            # Keep angle in range
                            child[i, 2] = child[i, 2] % 60
                
                # Repair
                child, valid = repair_collisions(child)
                new_pop.append(child)
                
                if valid:
                    side = evaluate(child)
                    if side < best_side:
                        best_side = side
                        best_pose = child.copy()
            
            pop = new_pop
            
            # Occasionally inject random individuals to maintain diversity
            if gen % 10 == 0 and gen > 0:
                pop[-1] = gen_random()
                pose, valid = repair_collisions(pop[-1])
                pop[-1] = pose
                if valid:
                    side = evaluate(pose)
                    if side < best_side:
                        best_side = side
                        best_pose = pose.copy()

        return best_pose, best_side

    # Run GA
    best_positions, best_side = genetic_algorithm()
    
    # Final check
    if best_positions is None:
        best_positions = gen_hex_lattice()
        best_positions, _ = repair_collisions(best_positions)
        best_side = evaluate(best_positions)
        if best_side == float('inf'):
            best_positions = np.array([[i * 3.0, 0.0, 0.0] for i in range(N)])
            best_side = 20.0

    # Construct output
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_positions[i, 2])])
        
    # The outer hexagon is centered at (0,0) with rotation 0
    # The side length is best_side
    # Note: The problem asks for "outer side length". 
    # My `required_outer_side` calculates the minimal side length of a hexagon 
    # centered at origin, rotated 0, containing the points.
    
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
