"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev tensor-product (degree 3) + 9 local Gaussian RBFs.
    10 params: 10 weights for the combined 19-dimensional feature vector.
    Wait, we only have 10 params. 
    Let's use 10 Chebyshev terms (degree 3 tensor product has (3+1)*(3+1)=16 terms, too many).
    
    Let's rethink. We have 10 parameters.
    Strategy: Use a mixture of global smooth basis and local RBFs.
    But with only 10 params, we can't have both a full cubic (10 terms) and RBFs.
    
    Alternative: Use a sparse set of high-order monomials? No, condition number issues.
    
    Let's look at the inspiration: "Replacing sparse polynomial+corner Gaussians with a dense 3x3 grid of local Gaussians... Measured: 2.056 -> 45.92."
    Wait, the score improved from 2.056 to 45.92? No, lower is better.
    "Measured: 2.056 -> 45.92" usually means the score went from 2.056 to 45.92 which is worse.
    OR it means the MSE went from 2.056 to 45.92?
    The text says "allows better approximation... while maintaining numerical stability".
    If the score (normalized MSE) went from 2.056 to 45.92, that's a huge regression.
    However, the child p43 has score 205.4, parent p37 has 354096566.
    So p43 is much better than p37.
    
    The current program (p37) is a full cubic polynomial. Score ~3.5e8.
    The child (p43) is a 3x3 Gaussian RBF. Score ~205.
    
    So RBFs are much better. Why?
    Maybe the function is not smooth or has sharp features that polynomials struggle with?
    Or maybe the polynomial fit is ill-conditioned and BFGS fails to converge to a good minimum?
    
    Let's try a better RBF setup.
    10 params.
    Option A: 9 RBF centers on a 3x3 grid + 1 bias.
    Option B: 10 RBF centers (maybe irregular or denser).
    Option C: Mixture of a few global terms and a few local RBFs.
    
    Let's stick to the 3x3 grid + bias as it worked well (score 205 vs 3.5e8).
    We can try to improve the RBF width or center placement.
    Fixed sigma is risky. Can we learn sigma?
    If we learn sigma, we need to allocate params to it.
    With 10 params, if we use 9 for weights, 1 for bias, we have no param for sigma.
    We can fix sigma.
    
    What if we use a different kernel? Inverse Multiquadric?
    Or maybe the centers should be slightly inside the domain to avoid boundary effects?
    Centers at [-0.5, 0, 0.5] instead of [-1, 0, 1]?
    
    Let's try centers at [-0.6, 0, 0.6] and a tuned sigma.
    Also, adding a linear term might help capture global trends better than just local bumps.
    
    Let's try:
    1 global linear term (x + y + 1) -> 3 params?
    7 local RBFs?
    
    Actually, the 3x3 grid + bias (10 params) is a very natural decomposition.
    Let's try to optimize the centers and width.
    Since we can't optimize centers/width with the 10 params (they are fixed in the basis), we must choose them wisely.
    
    Let's try a denser grid? 4x4 = 16 centers. Too many.
    How about 3x3 grid but with a smaller sigma to allow more localized fitting?
    Or larger sigma for smoother?
    
    Let's try:
    Centers: [-0.8, 0, 0.8]
    Sigma: 0.5
    
    Also, we should ensure numerical stability.
    exp(-dist^2 / (2*sigma^2))
    
    Let's also add a simple linear term?
    If we use 3 params for linear (1, x, y) and 7 for RBFs?
    7 RBFs is awkward.
    
    Let's stick to 9 RBFs + 1 bias.
    But maybe the bias is not the best use of the 10th param.
    What if we use 10 RBFs?
    Centers: 3x3 grid + center (0,0) duplicated? No.
    
    Let's try a different basis:
    Radial Basis Functions with a fixed width, but centers optimized?
    No, centers must be fixed in `equation` because `params` are the only variables optimized by BFGS.
    
    Let's try to improve the RBF formulation.
    Instead of Gaussian, use a compactly supported kernel?
    Or just tune the Gaussian.
    
    Current best known: 3x3 grid at [-1,0,1], score 205.
    Let's try shifting centers to [-0.5, 0, 0.5] to cover the bulk of the data better (corners are sparse in uniform dist).
    And sigma = 0.6.
    
    Also, let's add a safety check for time.
    
    Another idea: Use a polynomial basis but with orthogonal polynomials (Legendre) to improve conditioning?
    Legendre polynomials on [-1,1].
    P0=1, P1=x, P2=(3x^2-1)/2, etc.
    Tensor product: P_i(x) * P_j(y).
    Degree 2: 3x3 = 9 terms + 1 bias? No, P0*P0 is bias.
    So 9 terms for degree 2 tensor product.
    Plus 1 more term for degree 3?
    
    Let's compare RBF vs Orthogonal Polynomial.
    RBF score 205.
    Polynomial score 3.5e8.
    RBF is clearly better. The function likely has non-polynomial features or the polynomial fit is unstable.
    
    Let's refine the RBF.
    Centers: cx = np.array([-0.7, 0.0, 0.7]), cy = np.array([-0.7, 0.0, 0.7])
    Sigma: 0.5
    
    Let's try this.
    """
    p = params
    
    # 3x3 grid of centers, shifted slightly inward
    cx = np.array([-0.7, 0.0, 0.7])
    cy = np.array([-0.7, 0.0, 0.7])
    
    sigma = 0.5
    
    # Compute RBF values
    # x, y are 1D arrays
    # We need to compute exp(-((x-cxi)^2 + (y-cyi)^2) / (2*sigma^2)) for each center (cxi, cyi)
    
    # Efficient computation:
    # x is (N,), cx is (3,)
    # diff_x = x[:, None] - cx[None, :]  -> (N, 3)
    # diff_y = y[:, None] - cy[None, :]  -> (N, 3)
    
    dx = x[:, None] - cx[None, :]
    dy = y[:, None] - cy[None, :]
    
    # dist_sq = dx**2 + dy**2  -> This is not correct. dx is (N,3), dy is (N,3).
    # We need distance to each center (cx[i], cy[i]).
    # So for center i, dist^2 = (x - cx[i])^2 + (y - cy[i])^2
    
    # Let's construct a matrix of distances to each of the 9 centers.
    # Centers are combinations of cx and cy.
    # Cx_grid = cx[:, None] * np.ones((1,3)) -> (3,3)
    # Cy_grid = cy[None, :] * np.ones((3,1)) -> (3,3)
    # Flatten to (9,)
    
    cx_grid = np.repeat(cx, 3)
    cy_grid = np.tile(cy, 3)
    
    # Now cx_grid is [cx0, cx0, cx0, cx1, cx1, cx1, ...]
    # cy_grid is [cy0, cy1, cy2, cy0, cy1, cy2, ...]
    # This matches the 3x3 grid.
    
    dx = x[:, None] - cx_grid[None, :]  # (N, 9)
    dy = y[:, None] - cy_grid[None, :]  # (N, 9)
    
    dist_sq = dx**2 + dy**2  # (N, 9)
    
    # Gaussian kernel
    rbf = np.exp(-dist_sq / (2.0 * sigma**2))  # (N, 9)
    
    # Add bias column
    # rbf is (N, 9). We need to multiply by p[0:9] and add p[9].
    # Or add a column of 1s.
    
    # pred = rbf @ p[0:9] + p[9]
    pred = rbf @ p[:9] + p[9]
    
    return pred

# EVOLVE-BLOCK-END
