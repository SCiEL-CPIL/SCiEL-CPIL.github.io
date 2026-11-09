"""
Strategy: Dense Multi-Scale Fixed RBF Basis with Orthogonal Initialization.
Instead of a global polynomial basis, this strategy uses a dense grid of Radial Basis Functions (Gaussians) with multiple scales.
The model is linear in the 10 parameters, which BFGS can fit.
Crucially, we pre-compute the basis matrix for the training data inside the first call to `equation` (or once globally if possible, but here we assume x,y change per call? No, the contract says "equation(x, y, params)". Usually x,y are fixed for the fit phase. However, the evaluator calls equation repeatedly with the same x,y during the BFGS fit, and then again with test x,y.
We can cache the basis matrix for the training points (identified by the first call or a specific hash) to speed up the fit.
Actually, the evaluator fits params on training data. The function `equation` is called with the training x,y during the fit.
To make it fast and robust, we will:
1. Use a fixed set of 10 RBF centers and scales.
2. Since the model is linear, BFGS will converge quickly.
3. To improve the chance of a good fit, we can use a "warm start" or better initialization? No, we control the function, not the optimizer call.
4. However, we can structure the basis to be well-conditioned.
5. A key insight from inspirations: Dense Multi-Scale Fixed RBF Basis.
Let's define 10 RBFs with centers spread across [-1, 1]^2 and varying bandwidths to capture both low and high frequency features.

Centers:
We need 10 centers.
Grid: 3x3 = 9 centers + 1 center at (0,0) with a large scale? Or just 10 distinct points.
Let's use a grid of centers:
x_centers = [-1, 0, 1]
y_centers = [-1, 0, 1]
This gives 9 centers. Add one more at (0.5, 0.5) or similar?
Or use 5 centers with 2 scales each?
Let's try 10 centers:
c1: (-1, -1), c2: (1, -1), c3: (-1, 1), c4: (1, 1), c5: (0, 0), c6: (0.5, 0.5), c7: (-0.5, -0.5), c8: (0.5, -0.5), c9: (-0.5, 0.5), c10: (0, 0) with different scale?
Actually, mixing scales is better.
Let's define 10 basis functions:
5 Gaussian bumps with narrow width (sigma=0.2) at corners and center.
5 Gaussian bumps with wide width (sigma=0.8) at midpoints?
Or simply 10 Gaussians with sigma=0.3 at a 3x3 grid + 1 center?

Let's stick to a robust set:
Centers:
(-1, -1), (1, -1), (-1, 1), (1, 1)  (Corners)
(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5) (Inner grid)
(0, 0) (Center)
(0, 0) (Center, different scale)

Scales (sigma):
Corners: 0.5
Inner: 0.3
Center1: 0.8
Center2: 0.2

This provides multi-scale coverage.

Since the problem is linear in parameters, the optimization is convex. BFGS should find the global minimum for this basis. The key is that this basis is more expressive than a degree-3 polynomial for non-polynomial smooth functions, and potentially better conditioned if the centers are well-chosen.

To handle the "fitting" phase efficiently, we don't need to do anything special because the basis is fixed. The cost is just evaluating 10 exponentials per point. 200 points * 10 exps is very fast.

One potential issue: BFGS starting from all ones might be far from the optimum if the optimal weights are large or small. However, for RBFs, the weights are usually moderate.

Let's implement this.
"""
import numpy as np
import time

# Pre-define centers and scales
# 10 RBFs
# 4 corners, 4 inner, 2 center (different scales)
centers = np.array([
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0,  1.0],
    [-0.5, -0.5],
    [ 0.5, -0.5],
    [-0.5,  0.5],
    [ 0.5,  0.5],
    [ 0.0,  0.0],
    [ 0.0,  0.0]
])

scales = np.array([
    0.5,  # Corner
    0.5,  # Corner
    0.5,  # Corner
    0.5,  # Corner
    0.3,  # Inner
    0.3,  # Inner
    0.3,  # Inner
    0.3,  # Inner
    0.8,  # Center wide
    0.2   # Center narrow
])

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Ensure params length
    if len(params) < 10:
        p = np.zeros(10)
        p[:len(params)] = params
    else:
        p = params[:10]
    
    # Compute RBF basis
    # x and y are 1D arrays of length N
    # centers is (10, 2)
    # We need dist^2 for each point to each center
    
    # Vectorized computation
    # x: (N,), y: (N,)
    # centers[:, 0]: (10,), centers[:, 1]: (10,)
    
    # dx = x[:, None] - centers[None, :, 0]  # (N, 10)
    # dy = y[:, None] - centers[None, :, 1]  # (N, 10)
    # dist2 = dx**2 + dy**2
    # basis = exp(-dist2 / (2 * scales**2))  # (N, 10)
    # pred = basis @ p
    
    # To save memory/time, we can loop over centers or use broadcasting carefully.
    # N=200, K=10. Broadcasting is fine.
    
    dx = x[:, np.newaxis] - centers[np.newaxis, :, 0]
    dy = y[:, np.newaxis] - centers[np.newaxis, :, 1]
    dist2 = dx * dx + dy * dy
    
    # scales shape (10,) -> (1, 10) for broadcasting
    s2 = 2.0 * (scales[np.newaxis, :] ** 2)
    
    basis = np.exp(-dist2 / s2)
    
    pred = basis @ p
    
    # Safety check
    if time.time() - start_time > 10:
         # Fallback: simple average
         return np.mean(pred)
         
    return pred
