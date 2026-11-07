import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params: [c_v, x_c1, v_c1, sigma1, w1, x_c2, v_c2, sigma2, w2, A, omega]
    # Wait, that's 11. Let me recount.
    # 10 params: c_v, x_c1, v_c1, sigma1, w1, x_c2, v_c2, sigma2, w2, drive_omega
    # Drive amplitude can be absorbed... or use A as part of one weight.
    # Let me use: [c_v, x_c1, v_c1, sigma1, w1, x_c2, v_c2, sigma2, w2, omega]
    # Drive: w2 * sin(omega * t) is too coupled. 
    # Better: [c_v, c_x, x_c1, v_c1, sigma1, w1, x_c2, v_c2, sigma2, omega]
    # Model: c_v*v + c_x*x + w1*gauss1 + w2*gauss2 + drive
    # But that's 10 params and drive needs amplitude+phase or amplitude+freq.
    # Let me do: [c_v, c_x, x_c1, v_c1, sigma1, w1, x_c2, v_c2, sigma2, omega]
    # Drive = w1 * sin(omega * t) -- reuse w1? No, that couples nonlinear force with drive.
    
    # Final design (10 params):
    # [c_v, c_x, x_c, v_c, sigma, w_gauss, x_c2, v_c2, sigma2, omega]
    # Model: c_v*v + c_x*x + w_gauss*exp(-((x-x_c)^2+(v-v_c)^2)/(2*sigma^2))
    #         + w_gauss*exp(-((x-x_c2)^2+(v-v_c2)^2)/(2*sigma2^2))
    #         + w_gauss*sin(omega*t)
    # This reuses w_gauss for both bumps and drive -- not ideal.
    
    # Cleanest 10-param GP-inspired:
    # [c_v, c_x, x_c1, v_c1, sigma, w1, x_c2, v_c2, sigma2, omega]
    # Two RBF bumps with shared sigma? No, separate sigmas.
    # [c_v, c_x, x_c1, v_c1, w1, x_c2, v_c2, w2, omega, sigma]
    # Shared sigma for both bumps. Drive: w2*sin(omega*t)? No, w2 is bump weight.
    
    # Let me just do it properly with 10 distinct params:
    # p0: c_v (linear damping)
    # p1: c_x (linear stiffness)  
    # p2: x_center1
    # p3: v_center1
    # p4: weight1 (also drive amplitude)
    # p5: x_center2
    # p6: v_center2
    # p7: weight2
    # p8: sigma (shared Gaussian width)
    # p9: omega (drive frequency)
    # Model: c_v*v + c_x*x + w1*exp(-((x-xc1)^2+(v-vc1)^2)/(2*sigma^2))
    #         + w2*exp(-((x-xc2)^2+(v-vc2)^2)/(2*sigma^2))
    #         + w1*sin(omega*t)
    # w1 does double duty -- acceptable for this structural change.
    
    c_v = params[0]
    c_x = params[1]
    xc1 = params[2]
    vc1 = params[3]
    w1 = params[4]
    xc2 = params[5]
    vc2 = params[6]
    w2 = params[7]
    sigma = params[8]
    omega = params[9]
    
    # Ensure sigma doesn't cause numerical issues
    sigma_safe = np.where(np.abs(sigma) < 1e-6, 1e-6, sigma)
    
    # Gaussian bump 1
    d1_sq = (x - xc1) ** 2 + (v - vc1) ** 2
    gauss1 = np.exp(-d1_sq / (2.0 * sigma_safe ** 2))
    
    # Gaussian bump 2
    d2_sq = (x - xc2) ** 2 + (v - vc2) ** 2
    gauss2 = np.exp(-d2_sq / (2.0 * sigma_safe ** 2))
    
    # Linear terms
    linear = c_v * v + c_x * x
    
    # Sinusoidal drive
    drive = w1 * np.sin(omega * t)
    
    result = linear + w1 * gauss1 + w2 * gauss2 + drive
    
    return result
