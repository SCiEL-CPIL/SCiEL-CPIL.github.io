"""
Strategy: Small Multilayer Perceptron (MLP) Neural Network.

Instead of using a fixed basis (like polynomials or Gaussian RBFs) with fixed
centers or degrees, we treat the problem as learning a function from data using
a small neural network. The 10 parameters are the weights and biases of a
single hidden layer network.

Architecture:
Input: [strain, temp] (2 features)
Hidden Layer: 4 neurons, ReLU activation
Output: 1 neuron (stress)

Parameter Count:
- Weights Input->Hidden: 2 inputs * 4 neurons = 8 weights
- Biases Hidden: 4 biases
- Weights Hidden->Output: 4 neurons * 1 output = 4 weights
- Bias Output: 1 bias
Total = 8 + 4 + 4 + 1 = 17 parameters.

Wait, the constraint is EXACTLY 10 parameters.
Let's resize the network to fit exactly 10 parameters.

Option A:
Input: 2
Hidden: 3 neurons
Output: 1
Weights I->H: 2 * 3 = 6
Biases H: 3
Weights H->O: 3 * 1 = 3
Bias O: 1
Total: 6 + 3 + 3 + 1 = 13. Still too many.

Option B:
Input: 2
Hidden: 2 neurons
Output: 1
Weights I->H: 2 * 2 = 4
Biases H: 2
Weights H->O: 2 * 1 = 2
Bias O: 1
Total: 4 + 2 + 2 + 1 = 9.
We have 1 spare parameter. We can add a direct connection (skip connection) from
Input to Output? Or just use a larger hidden layer with fewer inputs?
No, inputs are fixed.

Let's try:
Input: 2
Hidden: 2 neurons
Output: 1
Weights I->H: 4
Biases H: 2
Weights H->O: 2
Bias O: 1
Total: 9.

Can we add a linear term directly?
The network output is $y = \sum_j w_j \sigma(\sum_i v_{ji} x_i + b_j) + c$.
If we add a direct linear term $k \cdot x_{linear}$, that's 1 more parameter.
Which $x$? Maybe just strain? Or a constant?
Actually, the bias $c$ handles the constant.
The linear terms are handled by the ReLU network if the optimizer pushes values to be linear (i.e., inside the linear regime of ReLU).
But ReLU is not smooth at 0. BFGS might struggle with the kink.
However, the task allows `scipy.optimize.minimize(method="BFGS")`. BFGS assumes smoothness. ReLU has a kink.
To make it smooth, we can use `tanh` or `sigmoid` or `softplus`.
Let's use `tanh` for smoothness, which is better for BFGS.

Let's stick to 10 parameters.
Hidden layer size $H=2$.
Weights $W_1$ (2x2) = 4 params.
Biases $b_1$ (2) = 2 params.
Weights $W_2$ (2x1) = 2 params.
Bias $b_2$ (1) = 1 param.
Total = 9.
We have 1 extra parameter.
Let's add a direct linear connection from strain to output.
$y = \text{net}(\mathbf{x}) + p_9 \cdot \text{strain}$.
This ensures the model can always capture the primary linear elastic behavior easily,
providing a strong inductive bias while the hidden layer captures the non-linear
yield/hardening.

Total 10 params:
p0..p3: W1 (2x2)
p4..p5: b1 (2)
p6..p7: W2 (2x1)
p8: b2 (1)
p9: direct strain coefficient

This is a fundamentally different design family: a parametric neural network
rather than a fixed-basis function approximation.
"""

import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress prediction using a small MLP (2-2-1) with tanh activation
    and a direct linear strain term.
    
    Parameters:
    strain: np.ndarray, shape (N,)
    temp: np.ndarray, shape (N,)
    params: np.ndarray, shape (10,)
    
    Returns:
    stress: np.ndarray, shape (N,)
    """
    # Unpack parameters
    # W1: 2x2 matrix (indices 0-3)
    w1_11, w1_12, w1_21, w1_22 = params[0], params[1], params[2], params[3]
    # b1: 2 vector (indices 4-5)
    b1_1, b1_2 = params[4], params[5]
    # W2: 2x1 vector (indices 6-7)
    w2_1, w2_2 = params[6], params[7]
    # b2: scalar (index 8)
    b2 = params[8]
    # Direct strain coefficient (index 9)
    p_direct = params[9]
    
    s = strain
    t = temp
    
    # Input vector: [s, t]
    # Hidden layer calculations
    # z1 = w1_11*s + w1_12*t + b1_1
    # z2 = w1_21*s + w1_22*t + b1_2
    
    z1 = w1_11 * s + w1_12 * t + b1_1
    z2 = w1_21 * s + w1_22 * t + b1_2
    
    # Activation: tanh (smooth, bounded, good for BFGS)
    a1 = np.tanh(z1)
    a2 = np.tanh(z2)
    
    # Output layer
    # y_net = w2_1*a1 + w2_2*a2 + b2
    y_net = w2_1 * a1 + w2_2 * a2 + b2
    
    # Add direct linear term
    stress = y_net + p_direct * s
    
    return stress
