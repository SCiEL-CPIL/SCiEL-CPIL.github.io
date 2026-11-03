# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_radii_lp(centers):
        """
        Solve the LP for maximum radii given centers.
        Returns: (sum_radii, radii, duals)
        duals shape: (N, N) for pairwise, (N,) for walls (approximated in return for gradient)
        Actually, let's return the duals in a structured way.
        """
        n = len(centers)
        # Objective: maximize sum(r) => minimize -sum(r)
        c = -np.ones(n)
        
        # Constraints:
        # 1. Wall constraints: r_i <= d_wall_i
        # 2. Pairwise: r_i + r_j <= d_ij
        
        # A_ub * x <= b_ub
        
        # Wall constraints: 1 * r_i + 0 * r_j ... <= d_wall_i
        # We can combine all into one A_ub matrix.
        
        # Number of constraints: n (walls) + n*(n-1)/2 (pairs)
        n_pairs = n * (n - 1) // 2
        A_ub = np.zeros((n + n_pairs, n))
        b_ub = np.zeros(n + n_pairs)
        
        # Fill wall constraints
        d_walls = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                             np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
        
        for i in range(n):
            A_ub[i, i] = 1.0
            b_ub[i] = d_walls[i]
        
        # Fill pairwise constraints
        # r_i + r_j <= d_ij
        # Note: linprog uses <= constraints.
        # We need to map pair (i, j) to row index n + k
        pair_idx = 0
        D = cdist(centers, centers)
        np.fill_diagonal(D, np.inf)
        
        for i in range(n):
            for j in range(i + 1, n):
                A_ub[n + pair_idx, i] = 1.0
                A_ub[n + pair_idx, j] = 1.0
                b_ub[n + pair_idx] = D[i, j]
                pair_idx += 1
        
        # Bounds: r_i >= 0
        bounds = [(0, None)] * n
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback to simple relaxation if LP fails (should be rare)
            radii = d_walls.copy()
            for _ in range(10):
                new_radii = radii.copy()
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        new_radii[i] = min(new_radii[i], D[i, j] - radii[j])
                if np.allclose(new_radii, radii, atol=1e-12):
                    break
                radii = new_radii
            return np.sum(radii), radii, None
            
        radii = res.x
        sum_radii = np.sum(radii)
        
        # Extract duals
        # res.ineqlin.marginals contains duals for A_ub constraints
        duals_wall = res.ineqlin.marginals[:n]
        duals_pair = res.ineqlin.marginals[n:]
        
        # Map duals back to pair structure
        duals_pair_matrix = np.zeros((n, n))
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                val = duals_pair[pair_idx]
                duals_pair_matrix[i, j] = val
                duals_pair_matrix[j, i] = val
                pair_idx += 1
                
        return sum_radii, radii, (duals_wall, duals_pair_matrix, D)

    def compute_gradient(centers, duals_data):
        """
        Compute gradient of sum_radii w.r.t centers.
        duals_data: (duals_wall, duals_pair_matrix, D)
        """
        if duals_data is None:
            return np.zeros_like(centers)
            
        duals_wall, duals_pair, D = duals_data
        n = len(centers)
        grad = np.zeros_like(centers)
        
        # Wall gradient contribution:
        # d(r_i)/dx_i for wall constraint active?
        # The dual variable y_i for r_i <= x_i implies dr_i/dx_i = y_i if active.
        # But the wall distance d_wall_i is min(x_i, 1-x_i, y_i, 1-y_i).
        # The gradient of d_wall_i w.r.t x_i is 1 if x_i < 0.5, -1 if x_i > 0.5.
        # Similarly for y.
        
        # Actually, it's easier to compute the gradient of the objective Z directly from the duals of the LP.
        # Z = sum r_i.
        # The value function V(d) is concave in d.
        # dZ/dx_k = sum_j (dZ/dd_kj) * (dd_kj/dx_k) + wall terms.
        
        # From LP duality, dZ/dd_ij = y_ij (dual for r_i + r_j <= d_ij).
        # dZ/dd_wall_i = y_wall_i (dual for r_i <= d_wall_i).
        
        # dd_ij/dx_i = (x_i - x_j) / d_ij
        # dd_wall_i/dx_i = 1 if x_i is the limiting wall (left), -1 if right.
        
        for i in range(n):
            # Pairwise contributions
            for j in range(n):
                if i == j: continue
                d_ij = D[i, j]
                if d_ij < 1e-9: continue
                # Dual y_ij is the marginal value of increasing d_ij
                y_ij = duals_pair[i, j]
                # Gradient of d_ij w.r.t x_i
                grad_x = (centers[i, 0] - centers[j, 0]) / d_ij
                grad_y = (centers[i, 1] - centers[j, 1]) / d_ij
                grad[i, 0] += y_ij * grad_x
                grad[i, 1] += y_ij * grad_y
                
            # Wall contributions
            # Determine which wall is limiting for i
            # d_wall_i = min(x_i, 1-x_i, y_i, 1-y_i)
            # We need the gradient of the min function.
            # If x_i < 1-x_i and x_i < y_i and x_i < 1-y_i, then d_wall_i = x_i, grad_x = 1.
            # This is non-differentiable at ties. We can use subgradient (average of active gradients).
            
            x_i, y_i = centers[i, 0], centers[i, 1]
            d_left = x_i
            d_right = 1.0 - x_i
            d_bottom = y_i
            d_top = 1.0 - y_i
            
            d_wall = min(d_left, d_right, d_bottom, d_top)
            
            # Find active walls
            active_x = []
            active_y = []
            if abs(d_left - d_wall) < 1e-9: active_x.append(1.0)
            if abs(d_right - d_wall) < 1e-9: active_x.append(-1.0)
            if abs(d_bottom - d_wall) < 1e-9: active_y.append(1.0)
            if abs(d_top - d_wall) < 1e-9: active_y.append(-1.0)
            
            # Average subgradient
            avg_gx = np.mean(active_x) if active_x else 0.0
            avg_gy = np.mean(active_y) if active_y else 0.0
            
            # The dual y_wall_i applies to the constraint r_i <= d_wall_i
            # So the contribution is y_wall_i * grad(d_wall_i)
            grad[i, 0] += duals_wall[i] * avg_gx
            grad[i, 1] += duals_wall[i] * avg_gy
            
        return grad

    def gradient_ascent(centers, max_iters=500, step_size=0.01, tol=1e-8):
        """
        Perform gradient ascent on the sum of radii.
        """
        current_centers = centers.copy()
        current_score, _, duals_data = compute_radii_lp(current_centers)
        
        best_centers = current_centers.copy()
        best_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            if duals_data is None:
                # If LP failed, fall back to random step
                dx = np.random.normal(0, 0.01, size=current_centers.shape)
                new_centers = np.clip(current_centers + dx, 0.01, 0.99)
            else:
                grad = compute_gradient(current_centers, duals_data)
                # Normalize gradient to control step size
                norm = np.linalg.norm(grad)
                if norm > 1e-10:
                    grad = grad / norm
                # Step
                new_centers = current_centers + step_size * grad
                new_centers = np.clip(new_centers, 0.01, 0.99)
            
            new_score, _, new_duals = compute_radii_lp(new_centers)
            
            if new_score > current_score + 1e-10:
                current_centers = new_centers
                current_score = new_score
                duals_data = new_duals
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
            else:
                # Line search or reduce step
                step_size *= 0.8
                if step_size < 1e-5:
                    break
            
        return best_centers, best_score

    def simulated_annealing(centers, max_iters=50000, T_start=0.05, T_end=0.001, seed=0):
        """
        Simulated annealing to optimize circle center positions.
        Uses exact LP evaluation.
        """
        rng = np.random.RandomState(seed)
        current = centers.copy()
        current_score, _, _ = compute_radii_lp(current)

        best = current.copy()
        best_score = current_score

        n_iters = max_iters
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break

            # Cooling schedule: geometric
            T = T_start * (T_end / T_start) ** (it / max(1, n_iters - 1))

            # Pick a random circle to move
            i = rng.randint(N)
            # Adaptive step size
            step = 0.02 * (T / T_start) ** 0.5 + 0.002
            dx = rng.normal(0, step)
            dy = rng.normal(0, step)

            new_centers = current.copy()
            new_centers[i, 0] = np.clip(current[i, 0] + dx, 0.01, 0.99)
            new_centers[i, 1] = np.clip(current[i, 1] + dy, 0.01, 0.99)

            new_score, _, _ = compute_radii_lp(new_centers)
            delta = new_score - current_score

            # Acceptance criterion
            if delta > 0 or rng.uniform() < np.exp(delta / max(T, 1e-10)):
                current = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()

        return best, best_score

    def local_search(centers, max_iters=3000, step_size=0.01, seed=0):
        """
        Best-fit local search: only accept improvements.
        """
        rng = np.random.RandomState(seed)
        current = centers.copy()
        current_score, _, _ = compute_radii_lp(current)
        best = current.copy()
        best_score = current_score

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Occasionally move 2 circles simultaneously
            if rng.random() < 0.3:
                idx = rng.choice(N, size=2, replace=False)
                for i in idx:
                    dx = rng.uniform(-step_size, step_size)
                    dy = rng.uniform(-step_size, step_size)
                    current[i, 0] = np.clip(current[i, 0] + dx, 0.01, 0.99)
                    current[i, 1] = np.clip(current[i, 1] + dy, 0.01, 0.99)
            else:
                i = rng.randint(N)
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                current[i, 0] = np.clip(current[i, 0] + dx, 0.01, 0.99)
                current[i, 1] = np.clip(current[i, 1] + dy, 0.01, 0.99)

            new_score, _, _ = compute_radii_lp(current)

            if new_score > current_score + 1e-12:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()

        return best, best_score

    def grid_init():
        """Structured grid initialization with jitter for better packing."""
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        rng = np.random.RandomState(42)
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    x = (i + 0.5) / side + rng.uniform(-0.02, 0.02)
                    y = (j + 0.5) / side + rng.uniform(-0.02, 0.02)
                    centers.append([np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)])
        return np.array(centers[:N])

    def random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))

    # Main optimization loop
    best_result = None
    best_score = -np.inf

    # Multiple initialization strategies
    inits = []
    inits.append(('grid', grid_init()))
    for s in [1, 7, 13, 23, 37, 53, 71, 89, 101, 131, 151, 173]:
        inits.append((f'rand_{s}', random_init(s)))

    # Run Gradient Ascent + SA + Local Search from each init
    for init_name, init_centers in inits:
        if time.time() > deadline - 15:
            break

        # Phase 1: Gradient Ascent (Fast convergence to local optimum)
        ga_iters = 200 if time.time() < deadline - 40 else 100
        ga_centers, ga_score = gradient_ascent(
            init_centers,
            max_iters=ga_iters,
            step_size=0.05
        )

        # Phase 2: Simulated Annealing (Escape local optima)
        sa_iters = 8000 if time.time() < deadline - 30 else 4000
        sa_centers, sa_score = simulated_annealing(
            ga_centers,
            max_iters=sa_iters,
            T_start=0.03,
            T_end=0.001,
            seed=hash(init_name) % 10000
        )

        # Phase 3: Gradient Ascent again (Polish)
        ga_centers2, ga_score2 = gradient_ascent(
            sa_centers,
            max_iters=100,
            step_size=0.02
        )

        # Phase 4: Local Search (Fine tune)
        ls_iters = 500 if time.time() < deadline - 10 else 200
        ls_centers, ls_score = local_search(
            ga_centers2,
            max_iters=ls_iters,
            step_size=0.005,
            seed=hash(init_name) % 10000 + 1
        )

        # Determine best of the phases
        candidates = [
            (ga_score, ga_centers),
            (sa_score, sa_centers),
            (ga_score2, ga_centers2),
            (ls_score, ls_centers)
        ]
        
        best_local_score, best_local_centers = max(candidates, key=lambda x: x[0])
        
        if best_local_score > best_score:
            best_score = best_local_score
            radii = compute_radii_lp(best_local_centers)[1]
            best_result = np.column_stack([best_local_centers, radii])

    # If no valid result, fall back to grid
    if best_result is None:
        centers = grid_init()
        radii = compute_radii_lp(centers)[1]
        best_result = np.column_stack([centers, radii])

    # Final refinement: Gradient Ascent on best result
    if best_result is not None and time.time() < deadline - 5:
        final_centers, final_score = gradient_ascent(
            best_result[:, :2],
            max_iters=200,
            step_size=0.01
        )
        if final_score > best_score:
            best_score = final_score
            radii = compute_radii_lp(final_centers)[1]
            best_result = np.column_stack([final_centers, radii])

    # Ensure final validity
    if best_result is not None:
        # Recompute radii with tiny margin
        radii = compute_radii_lp(best_result[:, :2])[1]
        radii = np.maximum(0, radii - 1e-9)
        best_result = np.column_stack([best_result[:, :2], radii])

        if not is_valid(best_result, tol=1e-6):
            # Fallback: conservative grid
            centers = grid_init()
            radii = compute_radii_lp(centers)[1]
            radii = np.maximum(0, radii - 0.001)
            best_result = np.column_stack([centers, radii])

    return best_result

# EVOLVE-BLOCK-END
