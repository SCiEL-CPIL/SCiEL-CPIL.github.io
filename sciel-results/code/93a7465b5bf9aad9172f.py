import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    
    # Check initial point
    f0 = f(x0)
    if abs(f0) <= 1e-8:
        return x0

    # Strategy: Iterative Grid Search with Expanding Range
    # We evaluate f at a set of points, check for sign changes, and refine.
    
    # Define ranges and number of points for each stage
    # Stage 1: Range 20, 50 points (step 0.4)
    # Stage 2: Range 200, 100 points (step 2.0)
    # Stage 3: Range 2000, 200 points (step 20.0)
    # Stage 4: Range 2000, 150 points (step ~26.6) - Refine? No, let's just do more points in the middle.
    
    # To be safe and distinct, let's do:
    # 1. Coarse scan over [-10, 10] with 20 points.
    # 2. If no sign change, coarse scan over [-100, 100] with 40 points.
    # 3. If no sign change, coarse scan over [-1000, 1000] with 80 points.
    # 4. If no sign change, fine scan over [-10, 10] with 100 points.
    
    # We need to be careful with call count.
    # Max 500 calls.
    
    def check_and_solve(points):
        """
        points: list of (x, f_x)
        Returns root if found, else None.
        """
        for i in range(len(points) - 1):
            x1, f1 = points[i]
            x2, f2 = points[i+1]
            if f1 * f2 < 0:
                try:
                    # Ensure bracket is valid for brentq
                    # brentq requires f(a) and f(b) to have different signs
                    root = brentq(f, x1, x2, xtol=1e-12, rtol=1e-12, maxiter=100)
                    fr = f(root)
                    if abs(fr) <= 1e-8:
                        return root
                except Exception:
                    pass
        return None

    # Stage 1: Local coarse
    range1 = 10.0
    n1 = 20
    xs1 = np.linspace(x0 - range1, x0 + range1, n1)
    points1 = []
    for x in xs1:
        fx = f(x)
        if abs(fx) <= 1e-8:
            return x
        points1.append((x, fx))
        
    root = check_and_solve(points1)
    if root is not None:
        return root

    # Stage 2: Wider coarse
    range2 = 100.0
    n2 = 40
    xs2 = np.linspace(x0 - range2, x0 + range2, n2)
    points2 = []
    for x in xs2:
        fx = f(x)
        if abs(fx) <= 1e-8:
            return x
        points2.append((x, fx))
        
    root = check_and_solve(points2)
    if root is not None:
        return root

    # Stage 3: Even wider coarse
    range3 = 1000.0
    n3 = 80
    xs3 = np.linspace(x0 - range3, x0 + range3, n3)
    points3 = []
    for x in xs3:
        fx = f(x)
        if abs(fx) <= 1e-8:
            return x
        points3.append((x, fx))
        
    root = check_and_solve(points3)
    if root is not None:
        return root

    # Stage 4: Fine local scan (in case we missed a high freq root in Stage 1)
    range4 = 10.0
    n4 = 100
    xs4 = np.linspace(x0 - range4, x0 + range4, n4)
    points4 = []
    for x in xs4:
        fx = f(x)
        if abs(fx) <= 1e-8:
            return x
        points4.append((x, fx))
        
    root = check_and_solve(points4)
    if root is not None:
        return root

    # Stage 5: Randomized search in local area to catch any missed sign changes
    # We have used approx 20 + 40 + 80 + 100 = 240 calls.
    # We have ~250 calls left.
    # Let's do 100 random points in [x0-10, x0+10] and check for sign changes with neighbors?
    # Or just check if any are close to 0.
    
    import random
    random.seed(42) # Fixed seed for reproducibility? The task doesn't forbid it.
    # Actually, let's just do a denser grid in the most promising area.
    # Or, let's try to find a local minimum of |f(x)| using a few random starts and gradient descent?
    # Too complex.
    
    # Let's just do another pass with a different offset to avoid aliasing.
    range5 = 50.0
    n5 = 100
    xs5 = np.linspace(x0 - range5, x0 + range5, n5)
    # Shift the grid slightly to avoid missing roots exactly on grid points from before?
    # No, linspace includes endpoints.
    points5 = []
    for x in xs5:
        fx = f(x)
        if abs(fx) <= 1e-8:
            return x
        points5.append((x, fx))
        
    root = check_and_solve(points5)
    if root is not None:
        return root

    return x0
