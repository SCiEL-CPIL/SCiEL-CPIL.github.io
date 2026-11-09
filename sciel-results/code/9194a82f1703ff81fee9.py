# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by excluding the line y = 0 (49 points).
    
    Field: F_49 = F_7[alpha] with alpha^2 = 3.
    A point is [[x_a, x_b], [y_a, y_b]] representing (x_a + x_b*alpha, y_a + y_b*alpha).
    
    We exclude all points where y = 0, i.e., y_a = 0 and y_b = 0.
    This gives |S| = 49, so |N| = 2401 - 49 = 2352.
    
    Validity: For any point x:
    - If x is not on the excluded line, use the vertical direction v=(0,1).
      The line through x in direction (0,1) is {x + t*(0,1) : t in F_49}.
      Since x has y != 0, none of the points on this line have y = 0
      (because adding t*(0,1) only changes the y-coordinate by t, and
      x_y + t = 0 has at most one solution t = -x_y, but wait, that point
      WOULD have y=0. Let me reconsider.
      
      Actually, the line {x + t*(0,1)} contains the point (x_x, x_y + t).
      When t = -x_y, we get (x_x, 0), which IS in the excluded set.
      So the vertical line through x intersects the excluded line at (x_x, 0).
      That means one point on the line (other than x) is in S, not in N.
      So vertical direction doesn't work either!
      
    Let me reconsider the validity condition.
    
    Actually, re-reading: "every point x of F_49^2 (in N or not) must lie on a line 
    {x + t*v : t in F_49}, with v = (1, m) for some m in F_49 or v = (0, 1), 
    whose points other than x all belong to N."
    
    So for x in N, we need a line through x such that all OTHER points on that line are in N.
    For x not in N (i.e., x in S), same condition.
    
    Let me verify: S = {(xa, xb, 0, 0) : xa, xb in F_7}.
    
    Take x = (u, v) with v != 0 (so x in N).
    Consider direction v_dir = (1, m) for some m.
    The line is {(u + t, v + t*m) : t in F_49}.
    A point (u+t, v+t*m) is in S iff v + t*m = 0, i.e., t = -v/m if m != 0, 
    or never if m = 0 and v != 0.
    
    If m = 0: the line is {(u+t, v) : t in F_49}. All points have y = v != 0, 
    so none are in S. Thus all points other than x are in N. VALID.
    
    So for x in N with y != 0, use direction (1, 0). The horizontal line through x 
    stays at height y, so no point on it is in S. All other points are in N.
    
    Now take x = (u, 0) in S (y = 0).
    Use direction (1, 0): line is {(u+t, 0) : t in F_49}. All these points have y=0, 
    so ALL are in S. Points other than x are in S, not in N. INVALID.
    
    Use direction (0, 1): line is {(u, t) : t in F_49}. The point (u, 0) is x.
    Other points are (u, t) for t != 0, which have y = t != 0, so they are in N. VALID!
    
    So for x in S (y=0), use direction (0,1). All other points on the vertical line 
    through x have y != 0, hence are in N.
    
    Therefore the construction is valid.
    """
    p = 7
    k = 3  # alpha^2 = 3
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Exclude points where y = 0 (ya=0, yb=0)
                    if ya == 0 and yb == 0:
                        continue
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
