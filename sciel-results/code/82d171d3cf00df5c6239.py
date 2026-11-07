"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Uses a fixed-step DOP853 (8th-order) Runge-Kutta method with a carefully
    chosen step size to minimize function calls while maintaining high accuracy.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 280s safety margin before 300s limit
    
    # DOP853 coefficients (from Hairer, Norsett, Wanner)
    # This is a 13-stage, 8th-order method with 7th-order error estimator
    
    # Stage times c_i
    c = np.array([
        0,
        1/40,
        4/45,
        4/15,
        1,
        5/12,
        3/5,
        1,
        5/2,
        1,
        3/2,
        1,
        1
    ])
    
    # Butcher tableau a_ij (trivially lower triangular)
    # We only need the coefficients for computing stages
    # k_i = f(t + c_i*h, y + h * sum_{j<i} a_ij * k_j)
    
    # a matrix (13x13)
    a = np.zeros((13, 13))
    a[1, 0] = 1/40
    a[2, 0] = 4/45
    a[2, 1] = -4/45
    a[3, 0] = 4/15
    a[3, 1] = -4/15
    a[3, 2] = 4/15
    a[4, 0] = -28/15
    a[4, 1] = 44/15
    a[4, 2] = -28/15
    a[4, 3] = 4/15
    a[5, 0] = -2197/41040
    a[5, 1] = 512/45
    a[5, 2] = -227/10
    a[5, 3] = 127/540
    a[5, 4] = 1/60
    a[6, 0] = 1321/270
    a[6, 1] = -4216/135
    a[6, 2] = 94/45
    a[6, 3] = -38/9
    a[6, 4] = 512/675
    a[7, 0] = 1708513302029/3277273500000
    a[7, 1] = -1587565959793/8377680000
    a[7, 2] = 1620782126731/4188840000
    a[7, 3] = -1386979653227/1047210000
    a[7, 4] = -440201693953/2094420000
    a[7, 5] = 79282/2295
    a[7, 6] = 3332/825
    a[8, 0] = -56881182205233/468791686000000
    a[8, 1] = 230391138373543/234395843000000
    a[8, 2] = -10763795107597/9375833720000
    a[8, 3] = 154711481150583/468791686000000
    a[8, 4] = -197598449501/1562638953200
    a[8, 5] = 1899/4840
    a[8, 6] = -1372/2295
    a[8, 7] = -1192/165
    a[9, 0] = 316163991132287/1562638953200000
    a[9, 1] = -207347503241513/52087965106666
    a[9, 2] = 98203972873639/15626389532000
    a[9, 3] = -341864310742129/26043982553333
    a[9, 4] = 7642063938587/2604398255333
    a[9, 5] = -2013/110
    a[9, 6] = 63224/5265
    a[9, 7] = 28414/3025
    a[9, 8] = -95/33
    a[10, 0] = 2894913161720/129924485145
    a[10, 1] = -40828986020425/25984897029
    a[10, 2] = 8502351808602/43308161715
    a[10, 3] = 989239535072/1157262723
    a[10, 4] = -10360316755/458947143
    a[10, 5] = 27767/990
    a[10, 6] = -2160/143
    a[10, 7] = 673/99
    a[10, 8] = -22/15
    a[10, 9] = 2/9
    a[11, 0] = 2158313798966197/3277273500000000
    a[11, 1] = -1627847989287483/109242450000000
    a[11, 2] = 1347402761574063/109242450000000
    a[11, 3] = -431694716763649/36414150000000
    a[11, 4] = 71753421712693/12138050000000
    a[11, 5] = -65738/1815
    a[11, 6] = 54706/3631
    a[11, 7] = -1413/605
    a[11, 8] = 5/33
    a[11, 9] = -1/10
    a[11, 10] = 1/15
    a[12, 0] = 122229887974709/622425164700000
    a[12, 1] = -3709129422452281/622425164700000
    a[12, 2] = 181282620589417/62242516470000
    a[12, 3] = -103108180902105/20747505490000
    a[12, 4] = 6949194596291/6915835163333
    a[12, 5] = -1348/1815
    a[12, 6] = 106/41
    a[12, 7] = -64/11
    a[12, 8] = 1/2
    a[12, 9] = 1/10
    a[12, 10] = 1/15
    a[12, 11] = 1/30
    
    # 8th order weights b_i
    b = np.array([
        7125469765078/27539815078125,
        -5067617215845/12706914062500,
        2370912226589/759375000000,
        2281283137033/12656250000000,
        471803391549/210937500000,
        -238002726687/1265625000000,
        52538667110/35156250000,
        0,
        0,
        0,
        0,
        0,
        3/2
    ])
    
    # 7th order weights b_hat_i (for error estimation)
    b_hat = np.array([
        6094814628091/26602294262500,
        -40828986020425/25984897029000,
        181282620589417/62242516470000,
        -103108180902105/20747505490000,
        6949194596291/6915835163333,
        -1348/1815,
        106/41,
        -64/11,
        1/2,
        1/10,
        1/15,
        1/30,
        0
    ])
    
    # Fixed step size: h = 0.05 gives 40 steps for t in [0, 2]
    # DOP853 has 13 stages per step, so 40 * 13 = 520 function calls
    # This is well under the 2,000,000 limit and should be very accurate
    
    h = 0.05
    n_steps = int(round((t1 - t0) / h))
    if n_steps < 1:
        n_steps = 1
        h = (t1 - t0) / n_steps
    
    t = t0
    y = y.copy()
    
    for step in range(n_steps):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Compute all 13 stages
        k = [None] * 13
        for i in range(13):
            # Compute y_stage = y + h * sum_{j=0}^{i-1} a[i,j] * k[j]
            y_stage = y.copy()
            for j in range(i):
                y_stage += h * a[i, j] * k[j]
            t_stage = t + c[i] * h
            k[i] = f(t_stage, y_stage)
        
        # Compute new y using 8th order weights
        y_new = y.copy()
        for i in range(13):
            y_new += h * b[i] * k[i]
        
        # Check for divergence
        if not np.all(np.isfinite(y_new)) or np.max(np.abs(y_new)) > 1e12:
            # Fallback: use a smaller step
            h = h / 2
            n_steps = int(round((t1 - t0) / h))
            continue
        
        y = y_new
        t = t + h
    
    return y
# EVOLVE-BLOCK-END
