import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a 1-hidden-layer neural network (MLP) with Tanh activation.
    
    This model uses a single hidden neuron with a linear combination of inputs passed through a tanh activation,
    combined with a direct linear path from inputs to output.
    
    Parameters:
    p[0:5] -> Weights and bias for the hidden layer: [w_b, w_s, w_temp, w_ph, bias]
    p[5]   -> Weight for the hidden neuron output
    p[6:10]-> Weights for the direct linear path: [w_b_lin, w_s_lin, w_temp_lin, w_ph_lin]
    
    Note: A global bias is omitted to fit within 10 parameters, assuming the linear terms can absorb constant offsets 
    if the inputs are centered, or we can shift the inputs. However, standard practice is to include bias.
    Let's re-allocate to ensure a bias is present if possible, or assume the data allows it.
    Actually, we can use the bias in the hidden layer to shift the activation, and the linear terms to set the baseline.
    """
    # Unpack parameters
    w_b_hid, w_s_hid, w_t_hid, w_p_hid, bias_hid = params[0], params[1], params[2], params[3], params[4]
    w_hid_out = params[5]
    w_b_lin, w_s_lin, w_t_lin, w_p_lin = params[6], params[7], params[8], params[9]
    
    # Normalize inputs to [0, 1] for better scaling of weights
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Clip to [0, 1]
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    ph_norm = np.clip(ph_norm, 0.0, 1.0)
    
    # Compute hidden layer activation
    z = w_b_hid * b_norm + w_s_hid * s_norm + w_t_hid * temp_norm + w_p_hid * ph_norm + bias_hid
    h = np.tanh(z)
    
    # Compute output: combination of hidden unit and linear terms
    y = w_hid_out * h + w_b_lin * b_norm + w_s_lin * s_norm + w_t_lin * temp_norm + w_p_lin * ph_norm
    
    return y
