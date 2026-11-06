import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from scipy.sparse import LinearOperator

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    # 120s CPU budget, but wall clock limit is 230s. 
    # We target finishing by 100s to be safe against GC or scheduler delays.
    deadline = start_time + 100.0 

    # 1. Setup Fast Band Solver
    # 289 plane waves (17x17 grid of G vectors)
    half_pw = 8
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Precompute k-points. 
    # The evaluator uses 30 points. 
    # For the search, we use a dense grid in the irreducible BZ (triangle (0,0)-(pi,0)-(pi,pi))
    # to ensure we capture the min/max bands. 
    # We will use a slightly sparser set for the inner optimization loop to save time,
    # and a denser set for the final candidate evaluation.
    
    # Search k-points: 6x6 triangle ~ 21 points
    k_points_search = []
    n_k_search = 6
    for i in range(n_k_search + 1):
        for j in range(n_k_search + 1):
            if i + j <= n_k_search:
                s = i / n_k_search
                t = j / n_k_search
                k_points_search.append((s * np.pi, t * np.pi))
    k_points_search = np.array(k_points_search)
    
    # Final k-points: 10x10 triangle ~ 55 points (more accurate)
    k_points_final = []
    n_k_final = 10
    for i in range(n_k_final + 1):
        for j in range(n_k_final + 1):
            if i + j <= n_k_final:
                s = i / n_k_final
                t = j / n_k_final
                k_points_final.append((s * np.pi, t * np.pi))
    k_points_final = np.array(k_points_final)

    def compute_score(cell, k_points):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        Returns the gap-midgap ratio.
        """
        # Convert to epsilon grid
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon for the convolution matrix
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            # Define the matrix-vector product for A
            def matvec(v):
                w = kG2 * v
                u = np.fft.fft2(w.reshape(17, 17))
                z = inv_eps_F * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            # Solve for lowest 6 eigenvalues
            try:
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-5)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                omega4 = np.sqrt(max(0, vals[3]))
                omega5 = np.sqrt(max(0, vals[4]))
                
                if omega5 < min_omega5:
                    min_omega5 = omega5
                if omega4 > max_omega4:
                    max_omega4 = omega4
                    
            except Exception:
                return -1.0
        
        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    def generate_cell_from_params(params):
        """
        Generate a 32x32 symmetric cell from parametric definition.
        params: [angle, length, thickness, exponent, center_offset]
        We model the dielectric as a superellipse (pill) centered at the cell center,
        but rotated. However, D4 symmetry requires the shape to be invariant under 90 deg rotation.
        A single pill breaks D4 unless it's a square or circle.
        
        To maintain D4 symmetry, we must use a shape that is symmetric under 90 deg rotation.
        Options:
        1. Superellipse (squircle) centered at (16,16).
        2. Four pills arranged in a cross or square pattern.
        
        Let's use a superellipse: |x-cx|^n + |y-cy|^n <= R^n
        This is D4 symmetric.
        Parameters: 
        - R: radius (scale)
        - n: exponent (2=circle, 4=squircle, high=box)
        - We can also add a secondary smaller superellipse or modify the shape?
        
        Actually, the "4-rod" design is 4 separate circles. That is also D4 symmetric.
        Let's parametrize the 4-rod design:
        - radius: r
        - center_distance: d (distance from center to rod center)
        Rods at (d, d), (-d, d), etc? No, (16+d, 16+d) etc.
        
        Let's try a hybrid: A central superellipse AND 4 corner superellipses?
        Or just a single central superellipse? 
        Usually, for TM gaps, a single large dielectric disk/squircle in the center works well, 
        or a checkerboard-like pattern.
        
        Let's stick to a parametric family that includes both:
        1. Central superellipse (R_c, n_c)
        2. 4 corner superellipses (R_k, n_k, dist_k)
        
        To keep it simple and fast, let's optimize a "superellipse ring" or just a "superellipse".
        Actually, the best known designs for large TM gaps often look like a "cross" or "square hole".
        
        Let's define the cell as:
        1 = Dielectric, 0 = Air.
        
        Model: A superellipse defined by |x-15.5|^n + |y-15.5|^n <= R^n is D4 symmetric.
        Let's optimize R and n.
        
        But wait, the "4-rod" design (4 separate circles) is also D4 symmetric.
        Let's allow a "mixture" or just pick the 4-rod model as it's a strong baseline.
        
        Params: [radius, center_dist, shape_exp]
        If center_dist > 0, we place 4 rods at (16-cd, 16-cd), (16+cd, 16-cd), etc.
        If center_dist == 0, it's a single central shape.
        
        Let's use a more flexible "superellipse" that can morph from a square to a circle.
        Equation: |x-15.5|^n + |y-15.5|^n <= R^n
        This is always D4 symmetric.
        
        Let's try optimizing:
        p[0]: R (radius, 0 to 15)
        p[1]: n (exponent, 1 to 10)
        
        Is this enough? Maybe not. What if the optimal shape is a "plus" sign?
        A plus sign can be approximated by a superellipse with low n? No, low n is diamond.
        High n is square.
        
        Let's add a second parameter for a "hole" or "inner shape"?
        Or let's use the 4-rod model which is known to be good.
        Params for 4-rod:
        p[0]: radius
        p[1]: offset (distance from center)
        
        Let's combine:
        If p[1] > 0.5, use 4 rods.
        Else, use central superellipse.
        
        Actually, let's just use a generic "Superellipse" centered at (15.5, 15.5).
        It's the most robust D4 symmetric shape.
        
        Wait, the "4-rod" design is effectively 4 separate superellipses.
        Let's parametrize 4 superellipses at (16-d, 16-d) etc.
        Params: [radius, d, exponent]
        
        Let's implement the 4-rod/superellipse generator.
        """
        cell = np.zeros((size, size), dtype=int)
        
        # Unpack params
        # We will search over:
        # 0: radius (0.5 to 10.0)
        # 1: offset (0.0 to 10.0) - distance from center (15.5, 15.5) to shape center
        # 2: exponent (1.0 to 8.0)
        
        r = params[0]
        d = params[1]
        n_exp = params[2]
        
        # Centers for the 4 shapes (if d>0) or 1 shape (if d==0)
        # To maintain D4, we place shapes at (15.5+d, 15.5+d), (15.5-d, 15.5-d), etc?
        # No, standard 4-rod is at (15.5+d, 15.5+d) is corner.
        # (15.5+d, 15.5-d) is edge.
        
        # Let's use the "corner" placement (closer to the 4-rod seed).
        # Centers:
        # (15.5 + d, 15.5 + d)
        # (15.5 - d, 15.5 + d)
        # (15.5 + d, 15.5 - d)
        # (15.5 - d, 15.5 - d)
        
        # If d is small, they merge into a central blob.
        
        centers = [
            (15.5 + d, 15.5 + d),
            (15.5 - d, 15.5 + d),
            (15.5 + d, 15.5 - d),
            (15.5 - d, 15.5 - d)
        ]
        
        # Vectorized generation
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        
        for cx, cy in centers:
            # Superellipse equation: |X-cx|^n + |Y-cy|^n <= r^n
            # Use absolute difference
            dx = np.abs(X - cx)
            dy = np.abs(Y - cy)
            
            # Handle n=1 (diamond) and n=inf (box) carefully?
            # For n in [1, 8], standard power is fine.
            # If n is very large, dx^n might overflow or be zero.
            # We can use log-sum-exp or just clip.
            # Let's clip n to [1.0, 8.0] in the optimizer.
            
            dist_metric = dx**n_exp + dy**n_exp
            mask = dist_metric <= r**n_exp
            
            cell[mask] = 1
            
        return cell

    # Objective function for minimizer (we want to maximize score, so minimize -score)
    def objective(params):
        if time.time() > deadline:
            return 1e6
            
        cell = generate_cell_from_params(params)
        score = compute_score(cell, k_points_search)
        
        # If score is invalid (overlap), return a large penalty
        if score < 0:
            return 1.0 - score # Higher penalty for worse overlap?
            # Actually, just return a large number.
            return 10.0
        
        return -score

    # Initial guesses
    # Guess 1: 4-rod (small d, medium r)
    init1 = np.array([3.0, 6.0, 2.0]) # r=3, d=6, n=2 (circles)
    # Guess 2: Central Squircle
    init2 = np.array([8.0, 0.0, 4.0]) # r=8, d=0, n=4
    # Guess 3: Large Square-ish
    init3 = np.array([10.0, 0.0, 8.0])
    
    best_score = -10.0
    best_params = init1
    best_cell = generate_cell_from_params(init1)
    
    # Evaluate initial guesses
    for init in [init1, init2, init3]:
        cell = generate_cell_from_params(init)
        s = compute_score(cell, k_points_search)
        if s > best_score:
            best_score = s
            best_params = init.copy()
            best_cell = cell.copy()
            
    # Run Nelder-Mead optimization from the best initial guess
    # Also run from the other guesses in parallel? No, sequential is fine if fast.
    # Let's run from the best one first.
    
    bounds = [(0.5, 12.0), (0.0, 12.0), (1.0, 8.0)]
    
    # We will run multiple optimizations to avoid local minima
    starts = [best_params, init1, init2, init3]
    
    for start_p in starts:
        if time.time() > deadline - 10:
            break
            
        try:
            # Nelder-Mead is robust for this small dimension
            res = minimize(
                objective, 
                start_p, 
                method='Nelder-Mead', 
                bounds=bounds,
                options={'xatol': 0.1, 'fatol': 0.01, 'maxiter': 100, 'maxfev': 200}
            )
            
            if res.fun < -best_score: # i.e. -score > best_score
                new_score = -res.fun
                if new_score > best_score:
                    best_score = new_score
                    best_params = res.x
                    best_cell = generate_cell_from_params(best_params)
                    
        except Exception as e:
            pass

    # Final Polish: 
    # The parametric search might not hit the exact pixel boundaries optimally.
    # We will do a quick local pixel-flip search around the best parametric cell.
    # This combines the global search of params with the local precision of pixels.
    
    current_cell = best_cell.copy()
    current_score = compute_score(current_cell, k_points_final) # Use final k-points for accuracy
    
    # If the final score is worse than search score, it's due to k-point sampling.
    # We should compare using the same k-points or trust the final one.
    # Let's use the final k-points for the final answer.
    
    # Quick local search:
    # We will try flipping symmetric groups of pixels.
    # To save time, we limit this to 10-20 iterations.
    
    pixel_indices = [(i, j) for i in range(size) for j in range(size)]
    random.shuffle(pixel_indices)
    
    def get_symmetric_pixels(i, j, size):
        pixels = set()
        pixels.add((i, j))
        pixels.add((i, size-1-j))
        pixels.add((size-1-i, j))
        pixels.add((size-1-i, size-1-j))
        pixels.add((j, i))
        pixels.add((size-1-j, i))
        pixels.add((j, size-1-i))
        pixels.add((size-1-j, size-1-i))
        return list(pixels)

    # Local hill climbing
    no_improve = 0
    max_no_improve = 5
    
    while time.time() < deadline - 5 and no_improve < 20:
        # Pick a random pixel
        i, j = random.choice(pixel_indices)
        group = get_symmetric_pixels(i, j, size)
        
        # Flip
        for pi, pj in group:
            current_cell[pi, pj] = 1 - current_cell[pi, pj]
            
        new_score = compute_score(current_cell, k_points_final)
        
        if new_score > current_score:
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        else:
            # Revert
            for pi, pj in group:
                current_cell[pi, pj] = 1 - current_cell[pi, pj]
            no_improve += 1

    # Ensure symmetry
    def is_symmetric(cell):
        if not np.array_equal(cell, cell[::-1, :]): return False
        if not np.array_equal(cell, cell[:, ::-1]): return False
        if not np.array_equal(cell, cell.T): return False
        return True

    if not is_symmetric(best_cell):
        # Fallback to the parametric cell which is guaranteed symmetric
        best_cell = generate_cell_from_params(best_params)
        
    return best_cell.tolist()
