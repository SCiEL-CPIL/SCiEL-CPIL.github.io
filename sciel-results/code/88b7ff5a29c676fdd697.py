import numpy as np
from scipy.optimize import minimize
import time

N = 300

# Pre-compute indices for pairwise distances
# We need i < j
i_idx, j_idx = np.triu_indices(N, k=1)
# There are N*(N-1)/2 pairs
num_pairs = len(i_idx)

def _compute_values(x):
    """
    Compute the vector of constraint values v.
    v contains:
    - d_ij / 2 for all i < j
    - 1 - ||c_i|| for all i
    """
    # Reshape x to (N, 2)
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    # Vectorized calculation of distances between all pairs in i_idx, j_idx
    # pts[i_idx] is shape (num_pairs, 2)
    # pts[j_idx] is shape (num_pairs, 2)
    diff = pts[i_idx] - pts[j_idx]
    dists = np.linalg.norm(diff, axis=1)
    v_pairs = dists / 2.0
    
    # Container distances
    norms = np.linalg.norm(pts, axis=1)
    v_cont = 1.0 - norms
    
    # Concatenate
    v = np.concatenate([v_pairs, v_cont])
    return v, pts

def _objective_and_grad(x, beta):
    """
    Compute the soft-min objective and its gradient.
    Objective: h(x) = min(v) + (1/beta) * log(sum(exp(-beta * (v_k - min(v)))))
    This is a smooth approximation of min(v). We minimize this to maximize min(v).
    """
    v, pts = _compute_values(x)
    
    # Numerical stability for log-sum-exp
    # We want log(sum(exp(-beta * (v - min_v))))
    min_v = np.min(v)
    # shifted_v = v - min_v
    # exponents = -beta * shifted_v
    # Note: if beta is large, exponents can be very negative, exp() -> 0.
    # The term with min_v will have exponent 0, exp(0)=1.
    # So sum >= 1. No underflow issues for the dominant term.
    # Overflow? exponents are <= 0. So exp() <= 1. No overflow.
    
    shifted_v = v - min_v
    exponents = -beta * shifted_v
    # Clip exponents to avoid underflow to 0 if beta is very large and v-min_v is large
    # Actually, if v-min_v is large, exp is 0, which is fine.
    # But for gradient, we need probabilities p_k = exp(exponents) / sum(exp(exponents))
    
    exp_terms = np.exp(exponents)
    sum_exp = np.sum(exp_terms)
    
    # Objective value
    obj = min_v + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    # d obj / dx = d min_v / dx + (1/beta) * d log(sum_exp) / dx
    # d min_v / dx is 0 almost everywhere (subgradient), but in soft-min we treat it via the weights.
    # Actually, the derivative of the soft-min approximation:
    # h(x) = (1/beta) log sum exp(-beta v_k)
    # dh/dx = (1/beta) * (1/sum) * sum (-beta exp(-beta v_k)) dv_k/dx
    #        = - sum p_k dv_k/dx
    # where p_k = exp(-beta v_k) / sum exp(-beta v_k)
    # Wait, my formula above included min_v for stability.
    # Let's stick to the stable form:
    # h(x) = min_v + (1/beta) log sum exp(-beta (v_k - min_v))
    # The gradient of min_v is technically the subgradient of the min, but in the context of the soft-min approximation, the "weight" is distributed.
    # The standard gradient of the log-sum-exp soft-min is:
    # grad = - sum_k p_k * grad(v_k)
    # where p_k = exp(-beta v_k) / sum_j exp(-beta v_j)
    # Let's verify with the stable form.
    # Let S = sum exp(-beta(v_k - min_v)).
    # h = min_v + (1/beta) log S.
    # dh/dx = grad(min_v) + (1/beta) (1/S) sum (-beta exp(-beta(v_k-min_v))) (grad v_k - grad min_v)
    # This is getting complicated because min_v depends on x.
    # It is safer to use the direct gradient of the unstable form if we handle stability carefully, 
    # or just compute the weights p_k based on the stable exponents.
    
    # Let's use the weights from the stable calculation.
    # p_k = exp(-beta(v_k - min_v)) / sum_j exp(-beta(v_j - min_v))
    # This is equivalent to p_k = exp(-beta v_k) / sum_j exp(-beta v_j) because the min_v factor cancels out.
    
    weights = exp_terms / sum_exp
    
    # Now compute grad(v_k) for each k
    # v_k for pairs: d_ij / 2
    # d_ij = ||c_i - c_j||
    # grad_{c_i} d_ij = (c_i - c_j) / d_ij
    # grad_{c_j} d_ij = (c_j - c_i) / d_ij
    # So grad_{c_i} v_k = 0.5 * (c_i - c_j) / d_ij
    
    grad_x = np.zeros_like(x)
    
    # Handle pairs
    # weights_pairs = weights[:num_pairs]
    # We need to accumulate gradients.
    # For each pair (i, j), contribution to grad[c_i] is 0.5 * w * (c_i - c_j) / d_ij
    # Contribution to grad[c_j] is 0.5 * w * (c_j - c_i) / d_ij
    
    # To avoid division by zero if points coincide (shouldn't happen with good init, but just in case)
    dists_safe = np.where(dists < 1e-12, 1e-12, dists)
    
    # Vectorized gradient accumulation for pairs
    # diff is (num_pairs, 2)
    # unit_vecs = diff / dists_safe[:, np.newaxis]  # (num_pairs, 2)
    # contrib = 0.5 * weights[:num_pairs][:, np.newaxis] * unit_vecs
    # 
    # We need to add contrib to grad_x[i_idx] and subtract from grad_x[j_idx]
    # np.add.at is slow. 
    # Alternative: use np.bincount or scatter.
    # Since N=300 is small, we can loop over pairs if needed, but vectorized is better.
    # Let's try a vectorized scatter using np.add.at for correctness, checking speed.
    # If too slow, we might need to optimize this part.
    
    # Actually, for N=300, num_pairs ~ 45k. 
    # np.add.at on an array of size 600 with 45k updates is okay-ish but might be slow in Python.
    # Let's try to do it without np.add.at if possible.
    # Or just use a loop if the number of iterations is small.
    # L-BFGS-B calls objective/gradient many times.
    
    # Let's use a simple loop for the gradient accumulation of pairs to be safe on speed?
    # No, 45k iterations in Python is ~0.01s. If we do 1000 iters, that's 10s. Might be tight.
    # Let's stick to vectorized operations.
    
    # We can construct the gradient by summing vectors.
    # grad_i = sum_{j!=i} 0.5 * w_{ij} * (c_i - c_j)/d_{ij}
    
    # Let's compute the unit vectors and weighted diffs.
    unit_vecs = diff / dists_safe[:, np.newaxis]
    weighted_unit_vecs = 0.5 * weights[:num_pairs][:, np.newaxis] * unit_vecs
    
    # Accumulate for i_idx
    # grad_x[i_idx] += weighted_unit_vecs
    # grad_x[j_idx] -= weighted_unit_vecs
    
    # Using np.add.at
    np.add.at(grad_x, i_idx * 2, weighted_unit_vecs[:, 0])
    np.add.at(grad_x, i_idx * 2 + 1, weighted_unit_vecs[:, 1])
    np.add.at(grad_x, j_idx * 2, -weighted_unit_vecs[:, 0])
    np.add.at(grad_x, j_idx * 2 + 1, -weighted_unit_vecs[:, 1])
    
    # Handle container constraints
    # v_cont = 1 - ||c_i||
    # grad_{c_i} v_cont = - c_i / ||c_i||
    # weights_cont = weights[num_pairs:]
    
    norms_safe = np.where(norms < 1e-12, 1e-12, norms)
    # grad_cont = - weights_cont[:, np.newaxis] * pts / norms_safe[:, np.newaxis]
    # grad_x += grad_cont
    
    # Accumulate container gradients
    grad_cont = - (weights[num_pairs:] * (pts / norms_safe[:, np.newaxis]))
    grad_x += grad_cont
    
    # The total gradient of h(x) is - sum p_k grad v_k
    # Wait, the objective is to MINIMIZE h(x).
    # h(x) approximates min(v). We want to MAXIMIZE min(v).
    # So we MINIMIZE -h(x)?
    # No, the soft-min h(x) is a lower bound on min(v) and approaches it.
    # Maximizing min(v) is equivalent to minimizing -min(v).
    # So we should minimize -h(x).
    # Let's check the sign.
    # If we increase x to increase v, h(x) increases.
    # We want to increase h(x).
    # So we minimize -h(x).
    # Gradient of -h(x) is - grad(h(x)).
    
    return -obj, -grad_x

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: Hexagonal lattice
    # Estimate spacing
    # Area of unit circle = pi. Area of hex cell ~ sqrt(3)/2 * s^2.
    # N * Area_cell ~ pi => s ~ sqrt(2*pi / (N*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    # Generate hex lattice points
    points = []
    # Radius of lattice needed to cover unit circle
    # Max dist from origin is 1. 
    # Number of rings ~ 1/s
    max_rings = int(1.0 / s) + 2
    for q in range(-max_rings, max_rings + 1):
        for r in range(-max_rings, max_rings + 1):
            # Axial to cartesian
            x = s * (q + 0.5 * r)
            y = s * (np.sqrt(3.0) / 2.0) * r
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N points, select N closest to center?
    # Or just take the first N?
    # Better to select the N points that are most "interior" to avoid boundary issues initially?
    # Or just random subset?
    # The hex lattice is uniform. Let's take the N points with smallest radius to start central.
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        # Add random points
        num_missing = N - len(points)
        random_pts = np.random.uniform(-1, 1, (num_missing, 2))
        # Ensure inside circle
        while True:
            norms = np.linalg.norm(random_pts, axis=1)
            if np.all(norms <= 1.0):
                break
            random_pts[norms > 1.0] = np.random.uniform(-1, 1, (np.sum(norms > 1.0), 2))
        points = np.vstack([points, random_pts])
        
    x0 = points.flatten()
    
    # Optimization loop with annealing beta
    betas = [10.0, 50.0, 100.0, 200.0, 400.0]
    
    best_x = x0
    best_score = -np.inf
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Run L-BFGS-B
        try:
            res = minimize(
                _objective_and_grad, 
                best_x, 
                args=(beta,), 
                method='L-BFGS-B', 
                jac=True, # We provide gradient
                options={'maxiter': 200, 'gtol': 1e-6, 'disp': False}
            )
            if res.success:
                # Evaluate actual score
                v, _ = _compute_values(res.x)
                actual_min = np.min(v)
                if actual_min > best_score:
                    best_score = actual_min
                    best_x = res.x
        except Exception as e:
            pass
            
        # Check time
        if time.time() > deadline:
            break
            
    # Final polish: Nelder-Mead on the non-smooth objective?
    # Or just return the best x found.
    
    # Verify validity
    pts = best_x.reshape(N, 2)
    # Check if any point is outside
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms > 1.0):
        # Clip or re-initialize? 
        # If invalid, return a valid fallback
        # Fallback: random points
        pts = np.random.uniform(-0.9, 0.9, (N, 2))
        # Ensure inside
        for i in range(N):
            while np.linalg.norm(pts[i]) > 1.0:
                pts[i] = np.random.uniform(-0.9, 0.9, 2)
        return pts
        
    return pts

# The function signature must be construct()
# The code above defines construct() correctly.
