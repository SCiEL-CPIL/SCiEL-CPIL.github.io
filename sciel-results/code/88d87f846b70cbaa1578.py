"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed nodes for the spline. 
# We use 10 nodes spanning the strain range [-0.002, 0.964].
# The first node is at the lower bound, the last at the upper bound.
# This ensures the spline is well-defined and bounded.
_NODES = np.linspace(-0.002, 0.964, 10)

def _cubic_interp(x, nodes, ys):
    """
    Perform cubic interpolation.
    x: array of query points
    nodes: array of node x-coordinates (sorted)
    ys: array of node y-values (length 10)
    Returns: interpolated values
    """
    # Clip x to node range to avoid extrapolation issues, though inputs are in range
    x_clipped = np.clip(x, nodes[0], nodes[-1])
    
    # Find indices for the interval
    # searchsorted returns index i such that nodes[i-1] <= x < nodes[i]
    # We want the interval [i, i+1] such that nodes[i] <= x <= nodes[i+1]
    # Actually, let's use a simpler approach:
    # For each x, find the interval.
    
    # Vectorized approach:
    # Determine which segment each x falls into.
    # segments are between nodes[0]-nodes[1], ..., nodes[8]-nodes[9]
    # We can use np.searchsorted with 'right' or 'left' to find the index.
    # If x == nodes[i], it's on the node.
    
    # Let's compute the index of the left node for each x.
    # np.searchsorted(nodes, x, side='right') - 1 gives the index of the node <= x.
    # If x is exactly a node, side='right' gives i+1, so -1 gives i.
    # If x is between nodes[i] and nodes[i+1], side='right' gives i+1, so -1 gives i.
    # So idx = np.searchsorted(nodes, x_clipped, side='right') - 1
    # We need to ensure idx is in [0, 8] (since we have 10 nodes, 9 intervals).
    # If x is nodes[-1], searchsorted returns 10, idx=9. But nodes[9] is the last.
    # We should clamp idx to max 8 to use the last interval? Or handle the last node separately.
    # Actually, if x == nodes[-1], the interpolation should return ys[-1].
    # The interval formula works for x in [nodes[i], nodes[i+1]].
    # If idx=9, nodes[9] and nodes[10] (out of bounds).
    # So we clamp idx to 0..8.
    
    idx = np.searchsorted(nodes, x_clipped, side='right') - 1
    idx = np.clip(idx, 0, len(nodes) - 2)
    
    # Get the two nodes bounding each x
    x0 = nodes[idx]
    x1 = nodes[idx + 1]
    y0 = ys[idx]
    y1 = ys[idx + 1]
    
    # Also get the next/prev nodes for slope calculation (for natural or not-a-knot boundary conditions)
    # To keep it simple and robust, we can use a monotonic cubic interpolation (PCHIP) or simple cubic Hermite.
    # However, implementing PCHIP vectorized is complex.
    # Let's use a simple cubic spline approximation or just quadratic/cubic Hermite with estimated slopes.
    
    # Slope estimation:
    # m_i = (y_{i+1} - y_{i-1}) / (x_{i+1} - x_{i-1}) for interior points.
    # For boundaries, m_0 = (y_1 - y_0) / (x_1 - x_0) and m_{n-1} = (y_{n-1} - y_{n-2}) / (x_{n-1} - x_{n-2}).
    
    n = len(nodes)
    slopes = np.zeros(n)
    # Interior slopes
    slopes[1:-1] = (ys[2:] - ys[:-2]) / (nodes[2:] - nodes[:-2])
    # Boundary slopes
    slopes[0] = (ys[1] - ys[0]) / (nodes[1] - nodes[0])
    slopes[-1] = (ys[-1] - ys[-2]) / (nodes[-1] - nodes[-2])
    
    # Get slopes for the current interval
    m0 = slopes[idx]
    m1 = slopes[idx + 1]
    
    # Hermite basis functions
    t = (x_clipped - x0) / (x1 - x0)
    t2 = t * t
    t3 = t2 * t
    
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    
    # Hermite interpolation formula
    # y = h00*y0 + h10*(x1-x0)*m0 + h01*y1 + h11*(x1-x0)*m1
    h = (x1 - x0)
    y_interp = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
    
    # Handle exact node matches (though interpolation should be close enough)
    # If x is very close to a node, the formula should yield the node value.
    
    return y_interp

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using cubic spline interpolation for strain dependence
    and linear temperature correction.
    
    params:
    p[0:10] : y-values at the 10 strain nodes (for T=1.0 reference)
    """
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Extract spline nodes
    ys = params[0:10]
    
    # Base stress from spline
    stress_base = _cubic_interp(strain, _NODES, ys)
    
    # Temperature correction:
    # Assume stress scales with temperature.
    # Simple linear model: stress = stress_base * (1 + p10*(temp - 1.0)) ? 
    # But we only have 10 params. 
    # We must fit the temperature effect within the 10 params or assume it's minor.
    # Given the prompt "up to 10 fitted constants", and we used 10 for the spline.
    # Can we use the temperature to modulate the spline?
    # Perhaps: stress = stress_base * temp_factor?
    # If we don't have extra params, we might assume the temperature effect is captured by the spline's flexibility or is weak.
    # However, the seed program had a temp term.
    # Let's try to incorporate temperature by shifting the strain or scaling the stress.
    # Since we are out of params, we might have to sacrifice some spline resolution or use a different structure.
    
    # Alternative: Use 9 params for spline (8 intervals) and 1 for temp?
    # Or 10 params for spline and assume temp is negligible?
    # Let's check the score of the seed (1.0) which was linear in both.
    # A spline in strain alone might be better than linear in strain+temp if temp is weak.
    # But if temp is strong, we will lose.
    
    # Let's try to use a hybrid:
    # 7 params for spline (6 intervals, quadratic maybe?) and 3 for temp?
    # No, 10 params total.
    
    # Let's stick to 10 params for the spline and assume the temperature dependence is weak or can be partially captured by the spline if the data is structured such that temp correlates with strain? Unlikely.
    
    # Better idea: Use 5 params for strain spline and 5 for temp interaction?
    # Or, use the params to define the spline at T=1, and add a simple temp term using derivatives?
    # No, we can't compute derivatives of params.
    
    # Let's try: 
    # stress = spline(strain) + (temp - 1.0) * spline2(strain)
    # This requires 20 params. We have 10.
    
    # What if we use 5 nodes (5 params) for the base curve and 5 params for the temp-modulated curve?
    # stress = spline_5(strain) + (temp - 1.0) * spline_5(strain)
    # This uses 10 params total.
    # This allows the shape to change with temperature.
    
    # Let's implement this.
    # Nodes for 5-point spline.
    nodes_5 = np.linspace(-0.002, 0.964, 5)
    
    # Split params
    ys_base = params[0:5]
    ys_temp = params[5:10]
    
    # Interpolate base
    stress_base = _cubic_interp(strain, nodes_5, ys_base)
    # Interpolate temp component
    stress_temp_comp = _cubic_interp(strain, nodes_5, ys_temp)
    
    # Combine
    # Use (temp - 1.0) to center the temperature effect
    stress = stress_base + (temp - 1.0) * stress_temp_comp
    
    return stress
# EVOLVE-BLOCK-END
