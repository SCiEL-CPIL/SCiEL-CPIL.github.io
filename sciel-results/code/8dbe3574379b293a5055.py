# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    """
    Finds a path for moving a sofa through an L-shaped corridor.
    Returns a float array of shape (n, 3) where each row is (x, y, angle_degrees).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Keep margin from 630s

    # Grid points for sofa area calculation
    u = np.arange(-3.0, 1.5, 0.01)
    v = np.arange(-1.5, 1.5, 0.01)
    gu, gv = np.meshgrid(u, v, indexing='ij')
    gx, gy = gu.ravel(), gv.ravel()
    N = len(gx)

    # Precompute base corridor boundaries
    # Base: union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    # We can check if a point (ul, vl) is in the base.
    # in_r1 = (ul >= 0) & (ul <= 1) & (vl >= -3) & (vl <= 1)
    # in_r2 = (ul >= -3) & (ul <= 1) & (vl >= 0) & (vl <= 1)

    def get_dense_path(waypoints):
        """
        Prepend (0,0,0), append (-2,0,90), then insert linear interpolations.
        waypoints: (k, 3) array.
        """
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([-2.0, 0.0, 90.0])
        full = np.vstack(([start], waypoints, [end]))
        
        dense_points = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            if m > 1:
                ts = np.linspace(1.0 / m, 1.0 - 1.0 / m, m - 1)
                new_pts = p1 + ts[:, np.newaxis] * (p2 - p1)
                dense_points.extend(new_pts)
            
            dense_points.append(p2)
        
        return np.array(dense_points)

    def compute_score(waypoints):
        """
        Compute the raw score (area).
        """
        dense = get_dense_path(waypoints)
        n_dense = len(dense)
        
        a_rad = np.deg2rad(dense[:, 2])
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        inside_all = np.ones(N, dtype=bool)
        
        # Batch processing to manage memory
        # N = 450 * 300 = 135,000
        # dense can be up to ~100 points.
        # 100 * 135000 = 13.5M floats ~ 100MB. This is fine.
        # Let's process all dense points at once if memory allows, or in large batches.
        # Let's try 20 at a time to be safe.
        
        batch_size = 20
        for batch_start in range(0, n_dense, batch_size):
            if time.time() > deadline:
                return 0.0 # Timeout guard
                
            batch_end = min(batch_start + batch_size, n_dense)
            b_cos = cos_a[batch_start:batch_end]
            b_sin = sin_a[batch_start:batch_end]
            b_x = dense[batch_start:batch_end, 0]
            b_y = dense[batch_start:batch_end, 1]
            
            # Vectorized rotation and check
            # wx, wy shape: (batch, N)
            wx = gx[None, :] - b_x[:, None]
            wy = gy[None, :] - b_y[:, None]
            
            ul = b_cos[:, None] * wx + b_sin[:, None] * wy
            vl = -b_sin[:, None] * wx + b_cos[:, None] * wy
            
            in_r1 = (ul >= 0.0) & (ul <= 1.0) & (vl >= -3.0) & (vl <= 1.0)
            in_r2 = (ul >= -3.0) & (ul <= 1.0) & (vl >= 0.0) & (vl <= 1.0)
            in_base = in_r1 | in_r2
            
            inside_all &= in_base.all(axis=0)
            
            if not np.any(inside_all):
                break
        
        return np.sum(inside_all) * 0.0001

    # Strategy:
    # The sofa moves from (0,0,0) to (-2,0,90).
    # The "inner corner" of the corridor is at (0,0) for a=0?
    # Base corridor:
    # R1: 0<=u<=1, -3<=v<=1. (Horizontal strip extending left from y=0 to y=1, x=0 to x=1? No.)
    # Wait, the problem says: "Row (x, y, a) places the width-1 L-shaped corridor with its inner corner at (x, y)".
    # Base (a=0) is union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}.
    # Let's visualize base at (0,0).
    # R1: x in [0,1], y in [-3,1]. This is a vertical strip to the RIGHT of the y-axis, from y=-3 to y=1.
    # R2: x in [-3,1], y in [0,1]. This is a horizontal strip ABOVE the x-axis, from x=-3 to x=1.
    # The "inner corner" is typically the corner where the two legs meet.
    # The intersection is [0,1] x [0,1].
    # The "outer" parts are:
    # R1 part not in R2: [0,1] x [-3,0).
    # R2 part not in R1: [-3,0) x [0,1].
    # So the L-shape occupies the 4th and 2nd quadrants relative to the corner (0,0)?
    # Actually, it's an L-shape with the corner at (0,0) pointing towards (-1, -1)?
    # Let's check the start position (0,0,0).
    # The sofa is the set of grid points... inside EVERY placement.
    # The sofa is fixed in world space? "The sofa is the set of grid points ... that are inside every placement".
    # Yes, the sofa is a fixed set of points in the world. We are moving the CORRIDOR?
    # "Row (x, y, a) places the ... corridor".
    # So we are moving the corridor over a fixed sofa?
    # "Moving sofa from a corridor path". Usually means moving the sofa through the corridor.
    # But the definition says: "A point w is inside placement (x, y, a) if R(-a)(w - (x, y)) lies in the base".
    # This means the corridor is the moving object, and the sofa points are fixed.
    # We want to maximize the number of sofa points that are inside the corridor for ALL steps in the path.
    # This is equivalent to finding the intersection of all corridor placements.
    # We want this intersection to contain as many sofa grid points as possible.
    # Since the sofa is fixed, we are effectively moving the corridor to "sweep" over the sofa.
    # Wait, if the corridor moves, the points inside it change.
    # We want points that are inside the corridor at step 1, AND step 2, AND ... AND step n.
    # This is the intersection of the corridor regions at all steps.
    # We want to maximize the area of this intersection (sampled by grid points).
    
    # So we are looking for a path of corridor positions such that the intersection of all corridor positions is as large as possible.
    # The corridor is an L-shape.
    # Start: (0,0,0). End: (-2,0,90).
    # At (0,0,0), the corridor covers:
    # R1: [0,1] x [-3,1]
    # R2: [-3,1] x [0,1]
    # Union: L-shape in the "top-right" relative to the corner? No.
    # x in [0,1], y in [-3,1] -> Right of y-axis.
    # x in [-3,1], y in [0,1] -> Above x-axis.
    # So it covers the 1st quadrant [0,1]x[0,1], part of 2nd [-3,0]x[0,1], part of 4th [0,1]x[-3,0].
    # It does NOT cover the 3rd quadrant.
    
    # At (-2,0,90):
    # Rotation 90 deg. R(-90) is -90 deg rotation.
    # Point w is inside if R(-90)(w - (-2,0)) is in base.
    # Let w' = w - (-2,0) = (w_x+2, w_y).
    # R(-90) is matrix [[0, 1], [-1, 0]].
    # u = 0*w'_x + 1*w'_y = w_y
    # v = -1*w'_x + 0*w'_y = -w'_x = -(w_x+2)
    # Base:
    # R1: 0<=u<=1, -3<=v<=1 -> 0<=w_y<=1, -3<=-(w_x+2)<=1 -> -1<=w_x+2<=3 -> -3<=w_x<=1.
    # So R1' is w_x in [-3,1], w_y in [0,1]. (Horizontal strip).
    # R2: -3<=u<=1, 0<=v<=1 -> -3<=w_y<=1, 0<=-(w_x+2)<=1 -> -1<=w_x+2<=0 -> -3<=w_x<=-2.
    # So R2' is w_x in [-3,-2], w_y in [-3,1]. (Vertical strip to the left).
    # Union: L-shape.
    # The intersection of Start and End corridors will be small.
    # Start covers: [0,1]x[-3,1] U [-3,1]x[0,1].
    # End covers: [-3,1]x[0,1] U [-3,-2]x[-3,1].
    # Intersection:
    # 1. [0,1]x[-3,1] INTER [-3,1]x[0,1] = [0,1]x[0,1].
    # 2. [0,1]x[-3,1] INTER [-3,-2]x[-3,1] = Empty (x ranges [0,1] and [-3,-2] don't overlap).
    # 3. [-3,1]x[0,1] INTER [-3,1]x[0,1] = [-3,1]x[0,1].
    # 4. [-3,1]x[0,1] INTER [-3,-2]x[-3,1] = [-3,-2]x[0,1].
    # Total Intersection: [0,1]x[0,1] U [-3,1]x[0,1] U [-3,-2]x[0,1] = [-3,1]x[0,1].
    # Wait, [0,1]x[0,1] is inside [-3,1]x[0,1].
    # So the intersection of just Start and End is the horizontal strip [-3,1]x[0,1].
    # Area = 4 * 1 = 4.
    # Grid points in this strip:
    # x in [-3, 1], step 0.01. 400 points.
    # y in [0, 1], step 0.01. 100 points.
    # Total 40,000 points.
    # Score = 40,000 * 0.0001 = 4.0.
    
    # Can we do better?
    # If we move the corridor such that the intersection is larger?
    # The intersection is bounded by the Start and End positions.
    # The intersection of ANY set of sets is a subset of the intersection of the first and last.
    # So the maximum possible score is the score of the intersection of Start and End.
    # Which is 4.0.
    
    # Wait, is the "sofa" fixed?
    # "The sofa is the set of grid points ... that are inside every placement".
    # Yes.
    # So the score is bounded by the intersection of the first and last placement.
    # The first is (0,0,0). The last is (-2,0,90).
    # The intersection is [-3,1]x[0,1].
    # Is there any path that allows a LARGER intersection?
    # No, because the intersection of {S1, ..., Sn} is a subset of S1 ∩ Sn.
    # So the maximum score is fixed by the start and end positions?
    # If so, the task is trivial: just return any valid path.
    # But the score is 1.07 for the current program.
    # 1.07 corresponds to 10,700 points.
    # 4.0 corresponds to 40,000 points.
    # Why is the current score so low?
    # Maybe my intersection calculation is wrong.
    
    # Let's re-evaluate the intersection of Start and End.
    # Start S:
    # R1: 0<=x<=1, -3<=y<=1
    # R2: -3<=x<=1, 0<=y<=1
    # S = R1 ∪ R2
    
    # End E:
    # R1': -3<=x<=1, 0<=y<=1  (Horizontal strip)
    # R2': -3<=x<=-2, -3<=y<=1 (Vertical strip on left)
    # E = R1' ∪ R2'
    
    # S ∩ E = (R1 ∪ R2) ∩ (R1' ∪ R2')
    # = (R1 ∩ R1') ∪ (R1 ∩ R2') ∪ (R2 ∩ R1') ∪ (R2 ∩ R2')
    
    # 1. R1 ∩ R1':
    # R1: [0,1]x[-3,1]
    # R1': [-3,1]x[0,1]
    # Inter: [0,1]x[0,1]
    
    # 2. R1 ∩ R2':
    # R1: [0,1]x[-3,1]
    # R2': [-3,-2]x[-3,1]
    # Inter: Empty (x: [0,1] vs [-3,-2])
    
    # 3. R2 ∩ R1':
    # R2: [-3,1]x[0,1]
    # R1': [-3,1]x[0,1]
    # Inter: [-3,1]x[0,1]
    
    # 4. R2 ∩ R2':
    # R2: [-3,1]x[0,1]
    # R2': [-3,-2]x[-3,1]
    # Inter: [-3,-2]x[0,1]
    
    # Union:
    # [0,1]x[0,1] ∪ ∅ ∪ [-3,1]x[0,1] ∪ [-3,-2]x[0,1]
    # [-3,1]x[0,1] contains [0,1]x[0,1] and [-3,-2]x[0,1].
    # So S ∩ E = [-3,1]x[0,1].
    
    # This is a 4x1 rectangle.
    # Points: x in [-3, 1] (400 pts), y in [0, 1] (100 pts).
    # Total 40,000.
    # Score 4.0.
    
    # Why does the current program get 1.07?
    # Maybe the intermediate steps cut away parts of this intersection?
    # The score is the intersection of ALL steps.
    # So if the path goes "out of the way", it might intersect with the sofa in a smaller region.
    # But the intersection is a subset of S ∩ E.
    # So the maximum is 4.0.
    # If the current program gets 1.07, it means the intersection of all its steps is much smaller than S ∩ E.
    # This implies that the path is "wasting" the potential intersection by moving the corridor in a way that excludes points that were in S ∩ E.
    # To maximize the score, we need to keep the corridor as "aligned" with the S ∩ E region as possible.
    # Ideally, the corridor should stay within the S ∩ E region as much as possible?
    # No, the corridor is moving. The intersection is the set of points that are covered by the corridor at EVERY step.
    # To maximize this, we want the corridor to cover the S ∩ E region at every step.
    # But the corridor shape changes (rotates).
    # At t=0, it covers S.
    # At t=1, it covers E.
    # In between, it covers some intermediate shape.
    # We want the intersection of all these shapes to be as large as possible.
    # Since S ∩ E is the upper bound, we want the intermediate shapes to contain S ∩ E.
    # Can we find a path such that S ∩ E ⊆ C(t) for all t?
    # If yes, the score is 4.0.
    
    # Let's check if S ∩ E = [-3,1]x[0,1] is contained in the corridor for some intermediate angles.
    # The corridor is an L-shape.
    # Can an L-shape rotated by 45 degrees contain the rectangle [-3,1]x[0,1]?
    # The L-shape has width 1. The rectangle has height 1 and width 4.
    # The L-shape is composed of two strips of width 1.
    # It's unlikely to contain a 4x1 rectangle unless the strips are aligned with it.
    # At t=0, the horizontal strip R2 is [-3,1]x[0,1]. It contains the rectangle.
    # At t=1, the horizontal strip R1' is [-3,1]x[0,1]. It contains the rectangle.
    # In between, the strips rotate.
    # If we rotate slowly, the strips might sweep out and lose coverage of the corners of the rectangle.
    # Specifically, the corners (-3,0), (1,0), (-3,1), (1,1).
    # We need the corridor to cover these points at all times.
    
    # This is a geometric problem.
    # We want to optimize the path (x(t), y(t), a(t)) such that the intersection of the corridor regions is maximized.
    # Since the upper bound is 4.0, we aim for 4.0.
    
    # The current program uses Nelder-Mead to optimize waypoints.
    # It seems to be stuck at 1.07.
    # Let's try a more direct approach:
    # We know the start and end.
    # We can try to construct a path that keeps the corridor "centered" on the intersection region.
    # Or, we can use a grid search / genetic algorithm on the waypoints.
    
    # Let's try a simple path first:
    # Linear interpolation of x, y, a.
    # x: 0 -> -2
    # y: 0 -> 0
    # a: 0 -> 90
    # Let's compute the score of this path.
    
    # Waypoints:
    # (0,0,0)
    # (-1,0,45)
    # (-2,0,90)
    
    # Let's add more waypoints to smooth the rotation.
    # (0,0,0), (-0.5, 0, 22.5), (-1, 0, 45), (-1.5, 0, 67.5), (-2, 0, 90)
    
    # Let's test this specific path.
    
    def test_path(waypoints):
        score = compute_score(waypoints)
        print(f"Score: {score}")
        return score

    # Waypoints for linear path
    k = 4
    angles = np.linspace(22.5, 67.5, k)
    x_coords = np.linspace(-0.5, -1.5, k)
    y_coords = np.zeros(k)
    waypoints = np.column_stack((x_coords, y_coords, angles))
    
    # print("Testing linear path...")
    # test_path(waypoints)
    
    # Let's try to optimize this path using a genetic algorithm or random search.
    # Since the search space is small (8 params), random search might work well.
    
    best_score = 0.0
    best_waypoints = None
    
    # Initial best is the linear path
    linear_score = compute_score(waypoints)
    if linear_score > best_score:
        best_score = linear_score
        best_waypoints = waypoints.copy()
        
    # Random search
    rng = np.random.default_rng(42)
    n_iterations = 5000
    
    # Bounds for x: [-3, 1], y: [-2, 2], a: [0, 90]
    # But we want to keep the path reasonable.
    # x should go from 0 to -2.
    # y should stay near 0.
    # a should go from 0 to 90.
    
    for i in range(n_iterations):
        if time.time() > deadline:
            break
            
        # Perturb the best waypoints
        # x: add noise
        # y: add noise
        # a: keep fixed? Or add noise?
        # If we change angles, we change the dense path significantly.
        # Let's keep angles fixed for now and only optimize x, y.
        
        new_x = best_waypoints[:, 0] + rng.normal(0, 0.1, k)
        new_y = best_waypoints[:, 1] + rng.normal(0, 0.1, k)
        
        # Clip to reasonable bounds
        new_x = np.clip(new_x, -3.0, 1.0)
        new_y = np.clip(new_y, -2.0, 2.0)
        
        new_waypoints = np.column_stack((new_x, new_y, best_waypoints[:, 2]))
        
        score = compute_score(new_waypoints)
        
        if score > best_score:
            best_score = score
            best_waypoints = new_waypoints.copy()
            
    # Now, let's try to optimize angles too.
    # We can do a second phase of random search on angles.
    for i in range(2000):
        if time.time() > deadline:
            break
            
        new_a = best_waypoints[:, 2] + rng.normal(0, 5.0, k)
        # Ensure angles are increasing and within [0, 90]
        # Simple constraint: just clip and sort?
        # Or reject if not increasing.
        
        # Let's just clip
        new_a = np.clip(new_a, 0.0, 90.0)
        
        # Ensure strictly increasing?
        # The dense path generation handles non-increasing angles?
        # m = ceil(|da|). If da is negative, m is still positive.
        # But the path should probably be monotonic in angle.
        # Let's enforce monotonicity.
        if not np.all(np.diff(new_a) > 0):
            continue
            
        new_waypoints = np.column_stack((best_waypoints[:, 0], best_waypoints[:, 1], new_a))
        
        score = compute_score(new_waypoints)
        
        if score > best_score:
            best_score = score
            best_waypoints = new_waypoints.copy()

    return best_waypoints

# EVOLVE-BLOCK-END
