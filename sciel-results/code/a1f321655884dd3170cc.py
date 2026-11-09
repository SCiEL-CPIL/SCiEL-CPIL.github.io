import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to [0, 1] based on known ranges
    # b: [1, 5]
    xb = (b - 1.0) / 4.0
    # s: [1, 5] 
    xs = (s - 1.0) / 4.0
    # T: [1, 40]
    xT = (temp - 1.0) / 39.0
    # pH: [2, 12]
    xPH = (pH - 2.0) / 10.0
    
    # RBF centers for each variable (3 centers per variable)
    # b centers: 0.2, 0.5, 0.8
    rb_b = np.exp(-((xb - 0.2)**2) / 0.1) + np.exp(-((xb - 0.5)**2) / 0.1) + np.exp(-((xb - 0.8)**2) / 0.1)
    
    # s centers: 0.2, 0.5, 0.8
    rb_s = np.exp(-((xs - 0.2)**2) / 0.1) + np.exp(-((xs - 0.5)**2) / 0.1) + np.exp(-((xs - 0.8)**2) / 0.1)
    
    # T centers: 0.2, 0.5, 0.8
    rb_T = np.exp(-((xT - 0.2)**2) / 0.1) + np.exp(-((xT - 0.5)**2) / 0.1) + np.exp(-((xT - 0.8)**2) / 0.1)
    
    # pH centers: 0.2, 0.5, 0.8
    rb_pH = np.exp(-((xPH - 0.2)**2) / 0.1) + np.exp(-((xPH - 0.5)**2) / 0.1) + np.exp(-((xPH - 0.8)**2) / 0.1)
    
    # Linear terms
    lin_b = xb
    lin_s = xs
    lin_T = xT
    lin_pH = xPH
    
    # Build feature matrix: [1, lin_b, rb_b, lin_s, rb_s, lin_T, rb_T, lin_pH, rb_pH]
    # 10 features: intercept + 4*(linear + rbf)
    # But we only have 10 params, so use: intercept, lin_b, lin_s, lin_T, lin_pH, rb_b, rb_s, rb_T, rb_pH
    # That's 9, add one more: maybe a quadratic or another RBF width
    
    # Actually, let's use: intercept + 4 linear + 4 RBF + 1 quadratic = 10
    # Or: intercept + 4 linear + 5 RBF terms (using different widths)
    
    # Let's use: intercept, 4 linear terms, and 5 RBF terms with different centers/widths
    # RBF1: centered at 0.3, width 0.15
    # RBF2: centered at 0.5, width 0.15  
    # RBF3: centered at 0.7, width 0.15
    # RBF4: centered at 0.4, width 0.2
    # RBF5: centered at 0.6, width 0.2
    
    # Combine all inputs into a single feature vector
    # Use weighted combination of all variables through RBFs
    
    # Feature 0: intercept
    f0 = np.ones_like(xb)
    
    # Features 1-4: linear terms
    f1 = xb
    f2 = xs
    f3 = xT
    f4 = xPH
    
    # Features 5-9: RBF terms combining all variables
    # RBF on b
    f5 = np.exp(-((xb - 0.3)**2) / 0.1)
    
    # RBF on s
    f6 = np.exp(-((xs - 0.3)**2) / 0.1)
    
    # RBF on T
    f7 = np.exp(-((xT - 0.5)**2) / 0.1)
    
    # RBF on pH
    f8 = np.exp(-((xPH - 0.5)**2) / 0.1)
    
    # Interaction-like RBF: product of normalized T and pH (smooth)
    f9 = xT * xPH
    
    # Final prediction: linear combination
    return p[0] * f0 + p[1] * f1 + p[2] * f2 + p[3] * f3 + p[4] * f4 + \
           p[5] * f5 + p[6] * f6 + p[7] * f7 + p[8] * f8 + p[9] * f9
