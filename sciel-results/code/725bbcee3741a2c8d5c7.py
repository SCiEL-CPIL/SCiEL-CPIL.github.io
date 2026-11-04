import numpy as np

def get_strassen_coeffs():
    # Indices: 0: a, 1: b, 2: c, 3: d for X
    # Indices: 0: e, 1: f, 2: g, 3: h for Y
    # Z indices: 0: z11, 1: z12, 2: z21, 3: z22
    
    # M1 = (a+d)(e+h)
    alpha1 = np.array([[1, 0], [0, 1]]) # a, d
    beta1 = np.array([[1, 0], [0, 1]]) # e, h
    
    # M2 = (c+d)e
    alpha2 = np.array([[0, 0], [1, 1]]) # c, d
    beta2 = np.array([[1, 0], [0, 0]]) # e
    
    # M3 = a(g-h)
    alpha3 = np.array([[1, 0], [0, 0]]) # a
    beta3 = np.array([[0, 0], [1, -1]]) # g, h
    
    # M4 = d(f-e)
    alpha4 = np.array([[0, 0], [0, 1]]) # d
    beta4 = np.array([[-1, 1], [0, 0]]) # e, f
    
    # M5 = (a+c)h
    alpha5 = np.array([[1, 0], [1, 0]]) # a, c
    beta5 = np.array([[0, 0], [0, 1]]) # h
    
    # M6 = (c-a)(e+g)
    alpha6 = np.array([[-1, 0], [1, 0]]) # a, c
    beta6 = np.array([[1, 0], [1, 0]]) # e, g
    
    # M7 = (d-b)(h+f)
    alpha7 = np.array([[0, -1], [0, 1]]) # b, d
    beta7 = np.array([[0, 1], [0, 1]]) # f, h
    
    # Reconstruction
    # z11 = M1 + M4 - M5 + M7
    gamma1 = np.array([[1, 0], [0, 0]])
    gamma4 = np.array([[1, 0], [0, 0]])
    gamma5 = np.array([[-1, 0], [0, 0]])
    gamma7 = np.array([[1, 0], [0, 0]])
    
    # z12 = M3 + M5
    gamma3_12 = np.array([[0, 1], [0, 0]])
    gamma5_12 = np.array([[0, 1], [0, 0]])
    
    # z21 = M2 + M4
    gamma2_21 = np.array([[0, 0], [1, 0]])
    gamma4_21 = np.array([[0, 0], [1, 0]])
    
    # z22 = M1 - M2 + M3 + M6
    gamma1_22 = np.array([[0, 0], [0, 1]])
    gamma2_22 = np.array([[0, 0], [0, -1]])
    gamma3_22 = np.array([[0, 0], [0, 1]])
    gamma6_22 = np.array([[0, 0], [0, 1]])
    
    # Combine gammas for each M
    gammas = []
    gammas.append(gamma1 + gamma1_22) # M1
    gammas.append(gamma2_21 + gamma2_22) # M2
    gammas.append(gamma3_12 + gamma3_22) # M3
    gammas.append(gamma4 + gamma4_21) # M4
    gammas.append(gamma5 + gamma5_12) # M5
    gammas.append(gamma6_22) # M6
    gammas.append(gamma7) # M7
    
    alphas = [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7]
    betas = [beta1, beta2, beta3, beta4, beta5, beta6, beta7]
    
    return alphas, betas, gammas

# Verify
alphas, betas, gammas = get_strassen_coeffs()
X = np.random.rand(2, 2)
Y = np.random.rand(2, 2)
Z_true = X @ Y
Z_calc = np.zeros((2, 2))
for s in range(7):
    M_s = np.sum(alphas[s] * X) * np.sum(betas[s] * Y)
    Z_calc += gammas[s] * M_s
print(np.allclose(Z_true, Z_calc))
