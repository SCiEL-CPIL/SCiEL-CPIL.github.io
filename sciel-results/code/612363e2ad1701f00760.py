import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [-1, 1]
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    p = params
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
        
    # Basis:
    # 0: Intercept
    # 1: Amp_u, 2: Center_u, 3: Width_u
    # 4: Amp_v, 5: Center_v, 6: Width_v
    # 7: Amp_w, 8: Center_w, 9: Width_w
    
    # Centers constrained to [-1, 1] via tanh
    cu = np.tanh(p[2])
    cv = np.tanh(p[5])
    cw = np.tanh(p[8])
    
    # Widths kept positive and reasonable via exp
    # Start p at 1 -> exp(1)=2.7. Range of u is [-1,1].
    # Width 2.7 is quite wide (covers whole range).
    # Width 0.5 is local.
    # If p is negative, width is small.
    # To avoid underflow/overflow in exp for large |p|, we can clip p.
    # But BFGS might go to -100? Unlikely in 120s.
    # Let's use a safer width transformation: 0.1 + 0.9 * sigmoid(p) ?
    # No, sigmoid maps to [0,1]. Width 0.1 to 1.0.
    # This limits width to 1.0 (half range).
    # A width of 1.0 in [-1,1] space is a bump of size ~2.
    # If we need a wider bump, we can't.
    # Let's use: width = 0.1 + np.exp(p)
    # If p=0, width=1.1.
    # If p=-5, width=0.1.
    # If p=5, width=149. (Very wide, approx constant).
    
    wu = 0.1 + np.exp(p[3])
    wv = 0.1 + np.exp(p[6])
    ww = 0.1 + np.exp(p[9])
    
    # Amplitudes are free
    au = p[1]
    av = p[4]
    aw = p[7]
    
    # Gaussian: exp(-((u-c)^2) / (2*w^2))
    # To avoid overflow in exp if w is small and u-c is large?
    # u-c is max 2. w min 0.1.
    # (2)^2 / (2*0.01) = 4/0.02 = 200.
    # exp(-200) is 0. Safe.
    
    g_u = np.exp(-((u - cu)**2) / (2 * wu**2))
    g_v = np.exp(-((v - cv)**2) / (2 * wv**2))
    g_w = np.exp(-((w - cw)**2) / (2 * ww**2))
    
    return p[0] + au * g_u + av * g_v + aw * g_w
