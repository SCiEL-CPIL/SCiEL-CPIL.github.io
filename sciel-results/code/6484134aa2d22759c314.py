import numpy as np

def construct():
    # Gauss-Lobatto quadrature of order 8 on [0, 1]
    # Nodes are -1, 1 and roots of P_6'(x) on [-1, 1]
    # P_6(x) = (1/16)(231x^6 - 315x^4 + 105x^2 - 5)
    # P_6'(x) = (1/16)(1386x^5 - 1260x^3 + 210x)
    # Roots of 1386x^5 - 1260x^3 + 210x = 0
    # x(1386x^4 - 1260x^2 + 210) = 0
    # x(693x^4 - 630x^2 + 105) = 0
    # x(231x^4 - 210x^2 + 35) = 0
    
    # Roots: x=0, and 231y^2 - 210y + 35 = 0 where y=x^2
    # y = (210 +/- sqrt(210^2 - 4*231*35)) / (2*231)
    # 210^2 = 44100
    # 4*231*35 = 4*8085 = 32340
    # sqrt(44100 - 32340) = sqrt(11760)
    
    disc = np.sqrt(11760.0)
    y1 = (210.0 - disc) / (2.0 * 231.0)
    y2 = (210.0 + disc) / (2.0 * 231.0)
    
    x_inner = np.array([0.0, -np.sqrt(y2), -np.sqrt(y1), np.sqrt(y1), np.sqrt(y2)])
    # Add endpoints -1 and 1
    x_nodes = np.array([-1.0] + list(x_inner) + [1.0])
    # Sort
    x_nodes = np.sort(x_nodes)
    
    # Weights for Gauss-Lobatto on [-1, 1]
    # w_k = 2 / (n(n-1) [P_{n-1}(x_k)]^2)
    # n=8, n-1=7
    # Need P_7(x_k)
    # P_7(x) = (1/32)(429x^7 - 693x^5 + 315x^3 - 35x)
    
    def P7(x):
        return (429*x**7 - 693*x**5 + 315*x**3 - 35*x) / 32.0
        
    n = 8
    P_vals = P7(x_nodes)
    weights_x = 2.0 / (n * (n - 1) * P_vals**2)
    
    # Transform to [0, 1]
    # t = (x + 1) / 2
    # dt = dx / 2
    # w_t = w_x / 2
    nodes = (x_nodes + 1.0) / 2.0
    weights = weights_x / 2.0
    
    # Ensure sum is 1
    # Sum of weights should be 1
    # Normalize just in case of floating point errors
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
