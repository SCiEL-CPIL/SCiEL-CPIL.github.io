import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to [0, 1] range to improve conditioning
    # b: [1, 5]
    xb = (b - 1.0) / 4.0
    # s: [1, 40] (based on training range [1.002, 39.98])
    xs = (s - 1.0) / 39.0
    # temp: [1, 40]
    xT = (temp - 1.0) / 39.0
    # pH: [2, 12]
    xPH = (pH - 2.0) / 10.0
    
    # Quadratic Additive Model + Interactions
    # This approach was successful in the current program (score 1.15).
    # We keep the linear-in-params structure for BFGS stability.
    # We expand the interaction space to include more biologically relevant pairs
    # and potentially higher-order terms if needed, but 10 params limit us.
    
    # Current params usage in the 1.15 program:
    # p[0]: offset
    # p[1]: xb
    # p[2]: xs
    # p[3]: xT
    # p[4]: xPH
    # p[5]: xb^2
    # p[6]: xs^2
    # p[7]: xT^2
    # p[8]: xPH^2
    # p[9]: xb * xs
    
    # Proposed Change:
    # The quadratic additive model is very flexible. 
    # E. coli growth is often limited by temperature and pH in a bell-shaped manner.
    # A pure quadratic in xT and xPH approximates a bell curve.
    # The interaction xb*xs (density * substrate) is less biologically intuitive than
    # interactions between environmental factors (T, pH) and substrate, or between T and pH.
    
    # Let's replace p[9] (xb*xs) with a more complex interaction or a different term.
    # However, the 1.15 score is already quite good. 
    # Inspiration 3 (score 2.5) used a multiplicative base + correction. It scored worse.
    # Inspiration 1 & 2 (score 41) used multiplicative biological models. They scored much worse.
    # This suggests the data might not strictly follow a simple multiplicative biological model,
    # or the BFGS optimizer struggles with the non-linear landscape of the multiplicative models.
    # The linear-in-params quadratic surface is robust.
    
    # To improve, we can try to capture more complex interactions within the 10 param limit.
    # Current: 1 offset + 4 linear + 4 quadratic + 1 interaction = 10 params.
    
    # Alternative allocation:
    # 1 offset
    # 4 linear
    # 3 quadratic (drop one? or keep all 4?)
    # 2 interactions
    
    # Let's try:
    # p[0]: offset
    # p[1]: xb
    # p[2]: xs
    # p[3]: xT
    # p[4]: xPH
    # p[5]: xb^2
    # p[6]: xs^2
    # p[7]: xT^2
    # p[8]: xPH^2
    # p[9]: xT * xPH  (Temperature-pH interaction is often significant in microbial growth)
    
    # Or maybe xb * xT? (Density-temperature)
    # Or xs * xT? (Substrate-temperature)
    
    # Let's stick to the most likely biologically relevant interaction.
    # Temperature and pH often interact (e.g., optimal pH shifts with temperature).
    # Substrate and Temperature also interact (Monod kinetics are T-dependent).
    
    # Let's try replacing xb*xs with xT*xPH.
    
    return (p[0] 
            + p[1] * xb 
            + p[2] * xs 
            + p[3] * xT 
            + p[4] * xPH 
            + p[5] * xb**2 
            + p[6] * xs**2 
            + p[7] * xT**2 
            + p[8] * xPH**2 
            + p[9] * xT * xPH)
