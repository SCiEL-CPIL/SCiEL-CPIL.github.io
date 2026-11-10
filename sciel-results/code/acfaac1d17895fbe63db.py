import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Helper to compute the exact radius
    def compute_radius(c):
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall clearance
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def smooth_min_and_grad(c, alpha=50.0):
        """
        Computes a smooth approximation of the minimum radius and its gradient.
        r_smooth = -1/alpha * log( sum(exp(-alpha * d_i)) )
        where d_i are the constraints (pair distances/2 and wall distances).
        
        Returns: (r_smooth, grad_r) where grad_r is shape (N, 3)
        """
        # 1. Pairwise constraints
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # We want to maximize min(dist/2). 
        # Let u_ij = dist_ij / 2.
        # Smooth min of u_ij.
        
        # To avoid underflow/overflow in exp, we use the log-sum-exp trick.
        # log(sum(exp(-alpha * u))) = -alpha * min(u) + log(sum(exp(-alpha * (u - min(u)))))
        # But we need the gradient of -1/alpha * log(sum(exp(-alpha * u)))
        
        # Let's compute for pairs first.
        # u_matrix shape (N, N). We only need i < j.
        # For efficiency, let's work with the full matrix but mask diagonal.
        
        # Actually, computing exp(-alpha * dists) for 100x100 is cheap.
        # But alpha * dists can be large.
        
        # Let's collect all constraints into a single vector V.
        # V contains:
        # 1. All pair distances / 2 (i < j)
        # 2. All wall distances (6 per point)
        
        # This vector has N*(N-1)/2 + 6*N elements.
        # For N=100, that's 4950 + 600 = 5550 elements.
        
        # Instead of building a huge vector, let's compute the smooth min parts separately 
        # and combine them? No, the smooth min of a union of sets is not the min of smooth mins.
        # We must treat them as one set.
        
        # Let's build the vector V and the Jacobian structure implicitly.
        
        # To save memory and time, let's compute the log-sum-exp for pairs and walls separately 
        # only if we can combine them.
        # log(sum_pairs + sum_walls) = log(exp(log_sum_pairs) + exp(log_sum_walls))
        
        # Let L_p = log(sum_{i<j} exp(-alpha * d_ij/2))
        # Let L_w = log(sum_{i,k} exp(-alpha * w_ik))
        
        # Total L = log(exp(L_p) + exp(L_w))
        # r_smooth = -1/alpha * L
        
        # Gradient of r_smooth wrt c:
        # dr/dc = -1/alpha * (1/L_sum) * d(L_sum)/dc
        # where L_sum = sum_all exp(-alpha * v_k)
        
        # d(L_p)/dc_m = sum_{i<j, m in {i,j}} exp(-alpha * d_ij/2) * (-alpha) * d(d_ij/2)/dc_m
        # d(d_ij/2)/dc_i = (c_i - c_j) / (2 * |c_i - c_j|) * 2 = (c_i - c_j) / |c_i - c_j|  -- Wait.
        # d_ij = |c_i - c_j|. d(d_ij)/dc_i = (c_i - c_j)/|c_i - c_j|.
        # So d(d_ij/2)/dc_i = 0.5 * (c_i - c_j)/|c_i - c_j|.
        
        # Let's compute the weights w_k = exp(-alpha * v_k) for all constraints.
        
        # 1. Pairs
        # dists is (N, N).
        # We need exp(-alpha * dists/2).
        # To avoid overflow, find min_d = min(dists[i<j]).
        # max_exp_arg = -alpha * min_d / 2.
        # exp_vals = exp(-alpha * (dists/2 - min_d/2)).
        
        min_pair = np.min(dists)
        # We need to be careful with diagonal (inf).
        # Set diagonal to 0 for calculation? No, they are inf so exp(-inf)=0.
        # But log-sum-exp needs care.
        
        # Let's compute L_p and its gradient contribution.
        # Shifted distances: d_shift = dists - min_pair
        # d_shift has 0 on the minimum pair(s), inf on diagonal.
        
        # exp_vals_pairs = exp(-alpha * d_shift / 2)
        # On diagonal, exp(-inf) = 0. Good.
        
        exp_pairs = np.exp(-alpha * (dists - min_pair) / 2.0)
        # exp_pairs[i,i] is exp(-inf) = 0.
        
        sum_exp_pairs = np.sum(exp_pairs)
        
        # Gradient of L_p wrt c:
        # d L_p / d c_m = sum_{i,j} exp_pairs[i,j] * (-alpha/2) * d(dists[i,j])/dc_m
        # d(dists[i,j])/dc_m is non-zero only if m==i or m==j.
        # If m==i: (c_i - c_j) / dists[i,j]
        # If m==j: (c_j - c_i) / dists[i,j]
        
        # Let's compute the vector field F_pairs[m] = sum_{j != m} exp_pairs[m,j] * (-alpha/2) * (c_m - c_j) / dists[m,j]
        # Note: dists[m,j] is inf if m==j, but exp is 0.
        # We need to handle division by zero for diagonal.
        
        # dists_inv = 1/dists. Set diagonal to 0.
        dists_inv = np.zeros_like(dists)
        mask = dists > 1e-9
        dists_inv[mask] = 1.0 / dists[mask]
        
        # term[i,j] = exp_pairs[i,j] * (-alpha/2) * (c_i - c_j) * dists_inv[i,j]
        # This is a tensor. We sum over j for each i.
        
        # Let's do it efficiently.
        # c_diff = c[:, None, :] - c[None, :, :]  (N, N, 3)
        # We want sum_j term[i,j]
        
        # term_weights[i,j] = exp_pairs[i,j] * (-alpha/2) * dists_inv[i,j]
        # grad_pairs = sum_j term_weights[i,j] * (c_i - c_j)
        #             = sum_j term_weights[i,j] * c_i - sum_j term_weights[i,j] * c_j
        #             = c_i * (sum_j term_weights[i,j]) - (term_weights @ c)_i
        
        tw = exp_pairs * (-alpha / 2.0) * dists_inv
        # tw is (N, N)
        
        row_sums = np.sum(tw, axis=1) # (N,)
        # (tw @ c) is (N, 3)
        tw_c = tw @ c
        
        grad_L_p = row_sums[:, np.newaxis] * c - tw_c
        
        # 2. Walls
        # Constraints: 0.5 - |c_ik| for k=0,1,2.
        # Let w_ik = 0.5 - |c_ik|.
        # min_wall = min(w_ik).
        # exp_walls = exp(-alpha * (w_ik - min_wall)).
        
        abs_c = np.abs(c) # (N, 3)
        wall_dists = 0.5 - abs_c # (N, 3)
        min_wall = np.min(wall_dists)
        
        exp_walls = np.exp(-alpha * (wall_dists - min_wall))
        
        sum_exp_walls = np.sum(exp_walls)
        
        # Gradient of L_w wrt c:
        # d L_w / d c_ik = exp_walls[i,k] * (-alpha) * d(0.5 - |c_ik|)/dc_ik
        # d|c_ik|/dc_ik = sign(c_ik)
        # So d w_ik / d c_ik = -sign(c_ik)
        # d L_w / d c_ik = exp_walls[i,k] * (-alpha) * (-sign(c_ik)) = alpha * sign(c_ik) * exp_walls[i,k]
        
        # grad_L_w[i,k] = alpha * sign(c_ik) * exp_walls[i,k]
        
        # Handle c_ik = 0. sign(0) is 0, but derivative of |x| at 0 is undefined. 
        # In practice, subgradient. We can use 0 or random. 0 is stable.
        signs = np.sign(c)
        # If c is 0, sign is 0.
        
        grad_L_w = alpha * signs * exp_walls
        
        # Combine
        # L_total_log = log(exp(L_p_shifted) + exp(L_w_shifted)) + alpha * (min_pair/2 + min_wall) ? 
        # Wait, the shift was different for pairs and walls.
        # L_p = log(sum exp(-alpha * (d_ij/2 - min_pair/2))) = log(sum exp(-alpha * d_ij/2)) + alpha * min_pair/2
        # L_w = log(sum exp(-alpha * (w_ik - min_wall))) = log(sum exp(-alpha * w_ik)) + alpha * min_wall
        
        # Let S_p = sum exp(-alpha * d_ij/2)
        # Let S_w = sum exp(-alpha * w_ik)
        
        # We computed exp_pairs relative to min_pair.
        # sum_exp_pairs = sum exp(-alpha * (d_ij/2 - min_pair/2)) = exp(alpha * min_pair/2) * S_p
        # So S_p = sum_exp_pairs * exp(-alpha * min_pair/2)
        
        # Similarly S_w = sum_exp_walls * exp(-alpha * min_wall)
        
        # We want log(S_p + S_w).
        # Let A = log(S_p), B = log(S_w).
        # A = log(sum_exp_pairs) - alpha * min_pair/2
        # B = log(sum_exp_walls) - alpha * min_wall
        
        log_S_p = np.log(sum_exp_pairs) - alpha * min_pair / 2.0
        log_S_w = np.log(sum_exp_walls) - alpha * min_wall
        
        # log(S_p + S_w) = max(A, B) + log(exp(A - max) + exp(B - max))
        M = np.maximum(log_S_p, log_S_w)
        log_S_total = M + np.log(np.exp(log_S_p - M) + np.exp(log_S_w - M))
        
        r_smooth = -1.0 / alpha * log_S_total
        
        # Gradient
        # d r_smooth / dc = -1/alpha * 1/(S_p + S_w) * (d S_p/dc + d S_w/dc)
        # d S_p/dc = S_p * d(log S_p)/dc = S_p * (d L_p/dc)  -- Wait.
        # S_p = sum exp(...). d S_p/dc = sum exp(...) * d(-alpha v)/dc = -alpha * sum exp(...) * dv/dc.
        # We calculated grad_L_p = d(log(sum exp(shifted)))/dc.
        # log(sum exp(shifted)) = log(S_p) + const.
        # So d(log S_p)/dc = grad_L_p.
        # Thus d S_p/dc = S_p * grad_L_p.
        
        # Similarly d S_w/dc = S_w * grad_L_w.
        
        # grad_r = -1/alpha * 1/S_total * (S_p * grad_L_p + S_w * grad_L_w)
        
        S_p = sum_exp_pairs * np.exp(-alpha * min_pair / 2.0)
        S_w = sum_exp_walls * np.exp(-alpha * min_wall)
        S_total = S_p + S_w
        
        # To avoid underflow in S_p and S_w if they are very small, 
        # we can use the probabilities P_p = S_p/S_total and P_w = S_w/S_total.
        # grad_r = -1/alpha * (P_p * grad_L_p + P_w * grad_L_w)
        
        P_p = S_p / S_total
        P_w = S_w / S_total
        
        grad_r = -1.0 / alpha * (P_p * grad_L_p + P_w * grad_L_w)
        
        return r_smooth, grad_r

    # Initial Configuration: Simple Lattice
    # We start with a grid that is slightly compressed to ensure overlap, 
    # then the optimizer will push them apart.
    # Or start with a valid packing and refine.
    # Let's try a few different initializations.
    
    best_centers = None
    best_radius = 0.0
    
    # Initialization 1: Cubic Lattice
    # N=100. 5x4x5 = 100.
    # Spacing s. 5*s <= 1 => s <= 0.2.
    # If s=0.2, points are at -0.4, -0.2, 0, 0.2, 0.4.
    # Walls at 0.5. Distance 0.1.
    # Pair dist 0.2. Radius 0.1.
    
    def make_lattice(nx, ny, nz):
        xs = np.linspace(-0.5 + 1.0/(2*nx), 0.5 - 1.0/(2*nx), nx) if nx > 1 else np.array([0.0])
        ys = np.linspace(-0.5 + 1.0/(2*ny), 0.5 - 1.0/(2*ny), ny) if ny > 1 else np.array([0.0])
        zs = np.linspace(-0.5 + 1.0/(2*nz), 0.5 - 1.0/(2*nz), nz) if nz > 1 else np.array([0.0])
        
        # Generate grid
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)
    
    # Try 5x4x5
    c0 = make_lattice(5, 4, 5)
    if len(c0) == N:
        # Shuffle to break symmetry
        np.random.shuffle(c0)
        
        # Optimize
        c_curr = c0.copy()
        
        # Adaptive Alpha: Start high (sharp min), decrease? 
        # Or start low (smooth) and increase?
        # Starting low allows global movement. Increasing alpha sharpens the minimum.
        
        alpha = 10.0
        lr = 0.01
        momentum = 0.9
        v = np.zeros_like(c_curr)
        
        steps = 0
        max_steps = 2000
        
        while steps < max_steps and time.time() < deadline - 5:
            r_smooth, grad = smooth_min_and_grad(c_curr, alpha)
            
            # Gradient Ascent
            v = momentum * v + lr * grad
            c_next = c_curr + v
            
            # Project to bounds [-0.5, 0.5]
            # If we clip, we lose gradient info at boundaries, but it's necessary.
            # Actually, for wall constraints, the gradient should push away from walls.
            # If a point is at 0.5, grad should be negative (push left).
            # If we clip, it stays at 0.5.
            # Better to use a penalty or just clip and accept reduced gradient.
            
            c_next = np.clip(c_next, -0.5, 0.5)
            
            # Check improvement in exact radius
            r_exact = compute_radius(c_next)
            
            if r_exact > best_radius:
                best_radius = r_exact
                best_centers = c_next.copy()
                
            # Heuristic: If no improvement for a while, reduce alpha (smooth more) to escape?
            # Or increase alpha to focus on the minimum?
            # Let's keep alpha constant or slowly increase it to sharpen.
            # Actually, increasing alpha makes the landscape sharper, easier to get stuck.
            # Decreasing alpha makes it flatter, easier to move.
            # Let's try decreasing alpha over time.
            alpha = max(5.0, alpha * 0.99)
            
            # Learning rate decay
            lr = max(0.001, lr * 0.999)
            
            c_curr = c_next
            steps += 1

    # Initialization 2: Random
    rng = np.random.default_rng(42)
    c0 = rng.uniform(-0.4, 0.4, size=(N, 3))
    
    c_curr = c0.copy()
    alpha = 10.0
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(c_curr)
    
    steps = 0
    max_steps = 2000
    
    while steps < max_steps and time.time() < deadline - 5:
        r_smooth, grad = smooth_min_and_grad(c_curr, alpha)
        
        v = momentum * v + lr * grad
        c_next = c_curr + v
        c_next = np.clip(c_next, -0.5, 0.5)
        
        r_exact = compute_radius(c_next)
        
        if r_exact > best_radius:
            best_radius = r_exact
            best_centers = c_next.copy()
            
        alpha = max(5.0, alpha * 0.99)
        lr = max(0.001, lr * 0.999)
        
        c_curr = c_next
        steps += 1

    # Final Local Refinement:
    # Use the best_centers found by gradient ascent.
    # Do a few steps of single-particle Nelder-Mead or similar to polish?
    # The gradient method might stop at a point where grad is small but not optimal.
    # Let's do a quick local search on the best candidate.
    
    if best_centers is not None:
        c_curr = best_centers.copy()
        
        # Identify bottlenecks
        r_curr = compute_radius(c_curr)
        
        # Simple hill climbing: move each particle slightly to improve global min.
        # This is expensive O(N^3) if done naively.
        # Let's just rely on the gradient result.
        
        # One more polish with smaller alpha (sharper)
        alpha = 100.0
        lr = 0.001
        v = np.zeros_like(c_curr)
        
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            r_smooth, grad = smooth_min_and_grad(c_curr, alpha)
            v = 0.9 * v + lr * grad
            c_next = c_curr + v
            c_next = np.clip(c_next, -0.5, 0.5)
            
            r_exact = compute_radius(c_next)
            if r_exact > best_radius:
                best_radius = r_exact
                best_centers = c_next.copy()
            c_curr = c_next

    if best_centers is None:
        # Fallback
        best_centers = np.zeros((N, 3))
        
    return best_centers

if __name__ == "__main__":
    c = construct()
    print("Radius:", compute_radius(c))
