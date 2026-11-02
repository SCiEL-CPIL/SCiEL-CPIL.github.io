import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    max_time = 110.0  # Safety margin
    
    # Initial grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    def get_score(c):
        # Min pairwise distance
        # N=50, 1225 pairs. Efficient enough.
        d = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Min boundary distance
        # Distance to closest face
        # Faces are at +-0.5
        # dist to face x=0.5 is 0.5 - c_x
        # dist to face x=-0.5 is 0.5 + c_x
        # min dist for point i is min(0.5 - |c_ix|, 0.5 - |c_iy|, 0.5 - |c_iz|)
        # Actually, min_k (0.5 - |c_ik|)
        abs_c = np.abs(c)
        dists = 0.5 - abs_c
        min_bound_dist = np.min(dists)
        
        return min(min_pair_dist / 2.0, min_bound_dist)
    
    best_centers = centers.copy()
    best_score = get_score(centers)
    
    # Surrogate objective: Sum of inverse distances to power P
    P = 4
    
    def get_gradient(c):
        grad = np.zeros_like(c)
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                diff = c[i] - c[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    continue
                dist = np.sqrt(dist_sq)
                # d/dci (1/dist^P) = -P * dist^(-P-2) * diff
                # We want to maximize, so move in direction of -grad = P * diff / dist^(P+2)
                factor = P * (dist ** (-(P+2)))
                grad[i] += factor * diff
                grad[j] -= factor * diff
        
        # Boundary repulsion
        for i in range(N):
            for k in range(3):
                val = c[i, k]
                dist_to_face = 0.5 - abs(val)
                if dist_to_face < 1e-9:
                    # Push strongly
                    grad[i, k] += 1000.0 * np.sign(val)
                else:
                    # d/dval (1/dist^P) = -P * dist^(-P-1) * d(dist)/dval
                    # d(dist)/dval = -sign(val)
                    # So d/dval = P * sign(val) * dist^(-P-1)
                    factor = P * (dist_to_face ** (-(P+1)))
                    grad[i, k] += factor * np.sign(val)
                    
        return grad
    
    # Gradient Ascent
    step_size = 0.01
    decay = 0.999
    min_step = 1e-6
    
    while time.time() - start_time < max_time:
        grad = get_gradient(centers)
        centers += step_size * grad
        # Clip to box
        centers = np.clip(centers, -0.5, 0.5)
        
        score = get_score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
        step_size *= decay
        if step_size < min_step:
            break
            
    return best_centers
