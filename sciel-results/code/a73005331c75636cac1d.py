# EVOLVE-BLOCK-START
import numpy as np


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path for the moving sofa problem through an L-shaped corridor.
    
    The path consists of intermediate placements (x, y, angle) between the start
    (0, 0, 0) and end (-2, 0, 90). The sofa is the intersection of all corridor
    placements along the path (including endpoints and interpolations).
    
    We use a simplified optimization approach: search over a parametric family
    of paths that approximate the known optimal "aspirin" shape motion.
    
    Returns:
        path: np.ndarray of shape (n, 3) with n >= 1, all finite values.
    """
    # The classic moving sofa solution uses a path where the sofa rotates
    # through the corner. The optimal area is approximately 2.2195.
    
    # We'll construct a path based on the known geometric solution.
    # The key insight: the sofa must slide along both corridors while rotating.
    
    # A good parametric path: the inner corner traces a curve while the angle
    # goes from 0 to 90 degrees. The known solution has the corner moving
    # along a specific curve.
    
    # Let's use a path with several intermediate points that approximate
    # the optimal motion. Based on literature, the optimal path has the
    # inner corner following a curve related to the sofa shape.
    
    # Simple approach: use a smooth arc for the corner position while
    # rotating the angle. The corner should move from near (0,0) to near
    # (-2, 0) as angle goes from 0 to 90.
    
    # Let's try a path where:
    # - Angle goes from ~5 to ~85 degrees (endpoints added by evaluator)
    # - Corner position follows a curve
    
    # Based on the known solution structure, let's use:
    # x(t) = -2 * t * (1 - t) * k1 - 2*t  ... let me think more carefully
    
    # Actually, let's use a simpler parameterization. The corner of the L
    # moves along a path. For the standard moving sofa:
    # At angle 0: corner at (0,0), corridor extends right and up
    # At angle 90: corner at (-2,0), corridor extends left and up
    
    # The intermediate positions should keep the sofa (intersection) as large
    # as possible.
    
    # Let me use a well-known approximate path from literature:
    # The inner corner follows approximately:
    # x = -t, y = 0.5*sin(pi*t) for t in [0,1] mapped to angle 0->90
    
    # But let's be more careful. Let's use a path with points that are
    # known to work well.
    
    # Actually, the simplest good path is one where the corner moves
    # smoothly and the rotation is gradual. Let's use:
    
    n_points = 20
    t = np.linspace(0.05, 0.95, n_points)
    angles = 90.0 * t  # from 4.5 to 85.5 degrees
    
    # Corner position: move from near origin to near (-2, 0)
    # Use a curve that goes "around" the corner
    x = -2.0 * t
    y = 0.3 * np.sin(np.pi * t)
    
    path = np.column_stack([x, y, angles])
    
    return path


# EVOLVE-BLOCK-END
