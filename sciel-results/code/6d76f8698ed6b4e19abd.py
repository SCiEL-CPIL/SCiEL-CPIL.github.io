import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # RBF basis functions with fixed centers and bandwidths
    # Centers spread across the domain [0.1, 6]
    # We use 9 RBFs and 1 constant term to make 10 parameters
    
    centers = np.array([0.5, 1.2, 2.0, 2.8, 3.6, 4.4, 5.2, 5.8, 6.0])
    bandwidths = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0])
    
    # Compute RBF terms: exp(-((x - center)^2) / (2 * bandwidth^2))
    # This gives local flexibility to capture non-polynomial features
    
    rbf_terms = []
    for i in range(9):
        rbf_terms.append(np.exp(-((x - centers[i])**2) / (2.0 * bandwidths[i]**2)))
    
    # Constant term
    const_term = np.ones_like(x)
    
    # Linear combination
    result = p0 * const_term
    for i in range(9):
        result = result + (p1 + i * 0) * rbf_terms[i]  # This is wrong, need to use p1..p9
    
    # Let me redo this properly
    result = p0 * const_term
    result = result + p1 * rbf_terms[0]
    result = result + p2 * rbf_terms[1]
    result = result + p3 * rbf_terms[2]
    result = result + p4 * rbf_terms[3]
    result = result + p5 * rbf_terms[4]
    result = result + p6 * rbf_terms[5]
    result = result + p7 * rbf_terms[6]
    result = result + p8 * rbf_terms[7]
    result = result + p9 * rbf_terms[8]
    
    return result
