import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based nonlinear oscillator model.
    Uses 5 Gaussian RBFs centered at learned locations in (t,x,v) space.
    params: [c0_t, c0_x, c0_v, w0, a0,
             c1_t, c1_x, c1_v, w1, a1]
    where ci are center coords, wi are widths, ai are amplitudes.
    Actually, with 10 params we can do:
    5 RBFs × 2 params each (width + amplitude) with fixed centers,
    or a different split. Let me use:
    4 RBFs with 2 params each (width, amplitude) = 8 params
    + 2 global params (bias, linear x coeff)
    
    Better: use 5 RBFs with fixed centers spread over the input range,
    and learn 5 widths + 5 amplitudes = 10 params.
    """
    # Fixed centers spread over the input ranges
    # t in [30, 50], x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    # Use 5 centers
    centers = np.array([
        [35.0, -0.15, -0.25],
        [40.0, 0.0, 0.0],
        [45.0, 0.15, 0.25],
        [37.0, 0.1, -0.1],
        [43.0, -0.1, 0.15]
    ])
    
    # params: [w0, a0, w1, a1, w2, a2, w3, a3, w4, a4]
    # wi = width of RBF i, ai = amplitude of RBF i
    
    # Compute squared distances from each input point to each center
    # t, x, v are 1D arrays of same length
    # Centers shape: (5, 3)
    
    # Normalize inputs to [0,1] range for better RBF behavior
    t_norm = (t - 30.0) / 20.0   # [0, 1]
    x_norm = (x - (-0.2056)) / (0.1749 - (-0.2056))  # [0, 1]
    v_norm = (v - (-0.3391)) / (0.3624 - (-0.3391))   # [0, 1]
    
    # Normalize centers similarly
    centers_norm = np.array([
        [(35.0 - 30.0) / 20.0, (-0.15 - (-0.2056)) / (0.1749 - (-0.2056)), (-0.25 - (-0.3391)) / (0.3624 - (-0.3391))],
        [(40.0 - 30.0) / 20.0, (0.0 - (-0.2056)) / (0.1749 - (-0.2056)), (0.0 - (-0.3391)) / (0.3624 - (-0.3391))],
        [(45.0 - 30.0) / 20.0, (0.15 - (-0.2056)) / (0.1749 - (-0.2056)), (0.25 - (-0.3391)) / (0.3624 - (-0.3391))],
        [(37.0 - 30.0) / 20.0, (0.1 - (-0.2056)) / (0.1749 - (-0.2056)), (-0.1 - (-0.3391)) / (0.3624 - (-0.3391))],
        [(43.0 - 30.0) / 20.0, (-0.1 - (-0.2056)) / (0.1749 - (-0.2056)), (0.15 - (-0.3391)) / (0.3624 - (-0.3391))]
    ])
    
    # Compute RBF values
    # For each RBF i: exp(-w_i * ||p - c_i||^2)
    # p = [t_norm, x_norm, v_norm] for each row
    
    # Stack inputs
    pts = np.column_stack([t_norm, x_norm, v_norm])  # shape (N, 3)
    
    # Compute distances to each center
    # diff shape: (N, 5, 3)
    diff = pts[:, np.newaxis, :] - centers_norm[np.newaxis, :, :]  # (N, 5, 3)
    sq_dist = np.sum(diff ** 2, axis=2)  # (N, 5)
    
    # Widths and amplitudes from params
    widths = params[0:5]  # w0..w4
    amps = params[5:10]   # a0..a4
    
    # Ensure positive widths for numerical stability
    widths = np.maximum(widths, 1e-6)
    
    # RBF values: exp(-width_i * sq_dist[:, i])
    rbf_vals = np.exp(-widths[np.newaxis, :] * sq_dist)  # (N, 5)
    
    # Output = sum of amp_i * rbf_i
    a = np.sum(amps[np.newaxis, :] * rbf_vals, axis=1)
    
    return a
