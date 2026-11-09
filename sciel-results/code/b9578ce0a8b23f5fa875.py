import numpy as np
import time

def construct():
    N = 200
    start_time = time.time()
    max_time = 115.0 # 5s safety margin for 120s limit
    
    # Helper to compute exact score
    def compute_score(centers):
        if not np.all(np.isfinite(centers)):
            return -np.inf
        if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
            return -np.inf
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Wall distances
        # Distance to boundaries: 0.5 - |x|, 0.5 - |y|
        wall_dists = np.minimum(
            0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])
        )
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall)

    def soft_min_score(centers, k=1000.0):
        """
        Compute a smooth approximation of the score using log-sum-exp for min.
        score = min( min_pair_dist, min_wall_dist )
        soft_min(x, y) = -1/k * log(exp(-k*x) + exp(-k*y))
        """
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = np.minimum(
            0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])
        )
        
        # Soft min of all pairwise distances (excluding inf)
        # We need to be careful with inf in log-sum-exp.
        # Replace inf with a large number? Or just filter?
        # N=200, 19900 pairs. Filtering is fast enough.
        valid_dists = dists[~np.isinf(dists)]
        if len(valid_dists) == 0:
            min_pair_soft = 0.0
        else:
            # log-sum-exp trick for numerical stability
            # min(x) = -1/k log(sum(exp(-k*x)))
            # log(sum(exp(-k*x))) = max(-k*x) + log(sum(exp(-k*x - max(-k*x))))
            # Let m = max(-k*x) = -k * min(x)
            # This is getting complicated. Let's just use the standard formula with clipping.
            # Since distances are positive and bounded, exp(-k*d) might underflow if k is large and d is large.
            # But we only care about the minimum, so large d contributes 0.
            # Let's use a simpler soft min: 
            # s = -1/k * log(sum(exp(-k*d)))
            # To avoid underflow/overflow, subtract max(-k*d) which is -k*min(d).
            # This is effectively just min(d) when k is large.
            
            # Let's just compute the gradient of the soft min directly?
            # Gradient of -1/k log(sum(exp(-k*d_ij))) wrt c_i
            # = 1/k * (sum_j exp(-k*d_ij) * (-k * grad d_ij)) / sum_j exp(-k*d_ij)
            # = - sum_j w_ij * grad d_ij, where w_ij = exp(-k*d_ij) / sum exp(-k*d)
            
            # For simplicity and speed, let's use a moderate k and standard log-sum-exp.
            # Actually, for N=200, we can just compute the gradient of the hard min by finding the min pair?
            # No, that's what the previous code did and it failed.
            # Let's implement the gradient of the soft min.
            
            # We will return the score and the gradient in a separate function for optimization?
            # Or just use finite differences? No, too slow.
            
            # Let's return the soft min value.
            # We need to handle the wall distances similarly.
            
            pass # This function is just a placeholder for the value. We need gradients for ascent.
            
        return 0.0 # Placeholder

    def gradient_ascent(centers, max_iters=2000, k=100.0, lr=0.001):
        """
        Perform gradient ascent on the soft-min score.
        """
        centers = centers.copy()
        best_score = compute_score(centers)
        best_centers = centers.copy()
        
        for i in range(max_iters):
            if time.time() - start_time > max_time:
                break
                
            # Compute pairwise distances and gradients
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Wall distances
            # w_i = min(0.5 - |x_i|, 0.5 - |y_i|)
            # We want to maximize min(dists, wall_dists).
            # Let's use a soft min of the vector of all constraints:
            # Constraints: d_ij for i<j, and w_i for all i.
            # Soft min of vector V: -1/k log(sum(exp(-k*V)))
            # Gradient wrt c_p:
            # sum_{constraints c involving p} w_c * grad_c
            # where w_c = exp(-k*V_c) / sum exp(-k*V)
            
            # 1. Pairwise terms
            # V_pair = dists (excluding diagonal)
            # We need sum over j != i of w_ij * (c_i - c_j)/d_ij
            # w_ij = exp(-k * d_ij) / Z
            # Z = sum_{i<j} exp(-k*d_ij) + sum_i exp(-k*w_i)
            
            # Compute exp(-k*d) for all pairs
            # To avoid underflow, we can scale. But k=100, d~0.03, k*d ~ 3. exp(-3) is fine.
            # Max d is ~1.4. exp(-140) is ~0. So only close pairs matter.
            
            pair_weights = np.exp(-k * dists)
            np.fill_diagonal(pair_weights, 0)
            
            # Wall weights
            wall_dists = np.minimum(
                0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])
            )
            wall_weights = np.exp(-k * wall_dists)
            
            Z = np.sum(pair_weights) + np.sum(wall_weights)
            
            if Z < 1e-100:
                break
                
            # Normalize weights
            pair_weights_norm = pair_weights / Z
            wall_weights_norm = wall_weights / Z
            
            # Gradient accumulation
            grads = np.zeros_like(centers)
            
            # Pairwise gradient contribution
            # grad_i += sum_j w_ij * (c_i - c_j) / d_ij
            # (c_i - c_j) / d_ij is the unit vector from j to i.
            # So we are moving i away from j.
            
            # Vectorized computation:
            # diff is (N, N, 2)
            # unit_vecs = diff / dists[:, :, np.newaxis]
            # handle dists=0 (should not happen with init)
            with np.errstate(divide='ignore', invalid='ignore'):
                unit_vecs = diff / dists[:, :, np.newaxis]
                # Set diagonal to 0
                np.fill_diagonal(unit_vecs[:, :, 0], 0)
                np.fill_diagonal(unit_vecs[:, :, 1], 0)
                # Handle any nan from 0/0
                unit_vecs = np.nan_to_num(unit_vecs)
            
            # weighted_sum = sum_j w_ij * unit_vec_ij
            # pair_weights_norm is (N, N)
            # We want to multiply each vector by its corresponding weight.
            # weighted_vecs = pair_weights_norm[:, :, np.newaxis] * unit_vecs
            # grads += sum_j weighted_vecs[i, j, :]
            
            weighted_vecs = pair_weights_norm[:, :, np.newaxis] * unit_vecs
            grads += np.sum(weighted_vecs, axis=1)
            
            # Wall gradient contribution
            # For each i, we have weight wall_weights_norm[i].
            # We need the gradient of the wall distance w_i wrt c_i.
            # w_i = min(0.5 - |x_i|, 0.5 - |y_i|)
            # This is not differentiable at the minimum switch.
            # However, the soft min handles the "min" part.
            # The term in the sum is exp(-k * w_i).
            # grad_i (wall part) = wall_weights_norm[i] * grad w_i
            # If w_i = 0.5 - |x_i|, grad = ( -sign(x_i), 0 )
            # If w_i = 0.5 - |y_i|, grad = ( 0, -sign(y_i) )
            # If equal, it's undefined, but the soft min blends them?
            # Actually, the soft min is over the *values*.
            # The gradient of the soft min value wrt c_i includes the gradient of the wall distance *value*.
            # Since we are using a soft min over the set of all distances, the weight wall_weights_norm[i] applies to the gradient of that specific distance value.
            # But w_i is the min of two values.
            # Let's approximate the gradient of w_i.
            # If 0.5 - |x| < 0.5 - |y|, grad is (-sign(x), 0).
            # If 0.5 - |x| > 0.5 - |y|, grad is (0, -sign(y)).
            # If equal, we can use the average or just pick one.
            
            for idx in range(N):
                x, y = centers[idx]
                dx = 0.5 - abs(x)
                dy = 0.5 - abs(y)
                
                if dx < dy:
                    # Dominated by x wall
                    gx = -np.sign(x) if x != 0 else 0
                    gy = 0
                elif dy < dx:
                    # Dominated by y wall
                    gx = 0
                    gy = -np.sign(y) if y != 0 else 0
                else:
                    # Corner or tie
                    gx = -np.sign(x) if x != 0 else 0
                    gy = -np.sign(y) if y != 0 else 0
                    # Normalize? The gradient of min is not unique.
                    # Let's just use the direction that increases both? No, min increases if both increase.
                    # Actually, if we are at a corner, moving diagonally out increases both.
                    # But we are constrained to stay in.
                    # The gradient of the distance to the boundary in the direction that increases the distance.
                    # If x>0, y>0, distance to right is 0.5-x, to top is 0.5-y.
                    # If they are equal, moving (-1, -1) normalized increases both.
                    # Let's use the sum of the individual gradients.
                    gx += -np.sign(x) if x != 0 else 0
                    gy += -np.sign(y) if y != 0 else 0
                    # Normalize the vector to have unit length?
                    # The gradient of the distance function should have unit length.
                    norm = np.sqrt(gx*gx + gy*gy)
                    if norm > 0:
                        gx /= norm
                        gy /= norm
                        # Wait, the gradient of min is not necessarily unit length in this sum.
                        # But for the soft min, the weights are normalized.
                        # Let's just use the unit vector pointing away from the closest wall(s).
                
                grads[idx, 0] += wall_weights_norm[idx] * gx
                grads[idx, 1] += wall_weights_norm[idx] * gy
            
            # Normalize gradients?
            # The step size lr controls the magnitude.
            
            # Update
            centers += lr * grads
            
            # Project back to square?
            # The gradient should keep them inside if we start inside and step is small.
            # But let's clip to be safe.
            centers = np.clip(centers, -0.5, 0.5)
            
            # Check score
            curr_score = compute_score(centers)
            if curr_score > best_score:
                best_score = curr_score
                best_centers = centers.copy()
                
            # Adaptive learning rate
            # If no improvement for a while, reduce lr
            # Simple: decay lr
            lr *= 0.995
            
            if lr < 1e-8:
                break
                
        return best_centers, best_score

    def polish(centers, max_iters=1000):
        """
        Resolve overlaps by moving points.
        """
        centers = centers.copy()
        for i in range(max_iters):
            if time.time() - start_time > max_time:
                break
            
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i_idx, j_idx = min_idx
            d_min = dists[i_idx, j_idx]
            
            # Wall check
            wall_dists = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
            min_wall_idx = np.argmin(wall_dists)
            d_wall = wall_dists[min_wall_idx]
            
            # If min pair is closer than current radius (which is min_pair/2), we have overlap?
            # No, radius is min_pair/2. So min_pair is the distance.
            # We want to maximize min_pair.
            # If d_min < d_wall, push i and j apart.
            # If d_wall < d_min, push wall_idx away from wall.
            
            step = 1e-4 # Small step
            
            if d_min < d_wall:
                vec = centers[i_idx] - centers[j_idx]
                norm = d_min
                if norm > 1e-10:
                    grad_i = vec / norm
                    grad_j = -vec / norm
                    centers[i_idx] += step * grad_i
                    centers[j_idx] += step * grad_j
            else:
                idx = min_wall_idx
                x, y = centers[idx]
                dl = x + 0.5
                dr = 0.5 - x
                db = y + 0.5
                dt = 0.5 - y
                
                min_wall_val = min(dl, dr, db, dt)
                
                if dl == min_wall_val:
                    centers[idx, 0] += step
                elif dr == min_wall_val:
                    centers[idx, 0] -= step
                elif db == min_wall_val:
                    centers[idx, 1] += step
                else:
                    centers[idx, 1] -= step
            
            centers = np.clip(centers, -0.5, 0.5)
            
        return centers

    # Hexagonal lattice initialization
    def hex_init(N):
        # Target radius ~ 0.0366
        # Spacing ~ 0.073
        # Rows needed ~ 1 / 0.073 ~ 13
        # Cols needed ~ 1 / (0.073 * sqrt(3)/2) ~ 15
        # Total ~ 195. Close to 200.
        
        # Generate a large hex grid and pick N points
        # Or generate exactly N points.
        
        # Let's use a grid with hexagonal offset.
        # r = 0.0366
        # dx = 2*r = 0.0732
        # dy = sqrt(3)*r = 0.0633
        
        # Number of rows
        # Area = 1.0. Area per circle = pi*r^2 ~ 0.0042
        # N=200. Total area covered = 0.84. Packing density ~ 0.84?
        # Hex packing density is pi/sqrt(12) ~ 0.9069.
        # So 0.84 is reasonable for finite square.
        
        # Let's just generate points on a hex grid covering [-0.5, 0.5]
        # and select the first N.
        
        r_target = 0.036
        dx = 2 * r_target
        dy = np.sqrt(3) * r_target
        
        points = []
        # Start from center? Or corner?
        # Let's start from -0.5 + r_target
        
        y = -0.5 + r_target
        row = 0
        while y <= 0.5 - r_target + 1e-6:
            x_offset = 0 if row % 2 == 0 else dx / 2
            x = -0.5 + r_target + x_offset
            while x <= 0.5 - r_target + 1e-6:
                points.append((x, y))
                x += dx
            y += dy
            row += 1
            
        points = np.array(points)
        
        # If we have more than N, we need to select N.
        # If we have less, we need to add more?
        # Let's just use the first N if available.
        # If not, pad with random?
        # Better: generate exactly N by adjusting grid size.
        
        # Let's try to fit N points.
        # Approximate number of rows m, cols k.
        # m * k ~ N.
        # Let's try m=14, k=15 -> 210 points.
        # We can remove some from the edges.
        
        # Let's just use the generated points and if len > N, take random N?
        # Or take the first N in a spiral order from center?
        
        # For simplicity, if len(points) >= N, take random N.
        # If len(points) < N, add random points?
        
        if len(points) >= N:
            # Select N points randomly to break symmetry?
            # Or take a contiguous block?
            # Random selection might lead to bad clustering.
            # Let's take the points closest to the center of the square.
            dists_to_center = np.linalg.norm(points, axis=1)
            sorted_indices = np.argsort(dists_to_center)
            return points[sorted_indices[:N]]
        else:
            # Pad with random
            remaining = N - len(points)
            random_points = np.random.uniform(-0.5 + r_target, 0.5 - r_target, (remaining, 2))
            return np.vstack([points, random_points])

    inits = [
        hex_init(N),
        np.random.uniform(-0.4, 0.4, (N, 2)),
        np.random.uniform(-0.4, 0.4, (N, 2))
    ]
    
    best_centers = None
    best_score = -np.inf
    
    for init in inits:
        if time.time() - start_time > max_time:
            break
            
        # Phase 1: Gradient Ascent
        c1, s1 = gradient_ascent(init, max_iters=1500, k=100.0, lr=0.002)
        
        # Phase 2: Polish
        c2 = polish(c1, max_iters=500)
        s2 = compute_score(c2)
        
        if s2 > best_score:
            best_score = s2
            best_centers = c2.copy()
            
    # Final check
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_score = compute_score(best_centers)
        
    return best_centers
