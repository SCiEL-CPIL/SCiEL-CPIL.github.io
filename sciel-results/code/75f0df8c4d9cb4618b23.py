import numpy as np
import time

def _union_length_1d(f, n_samples=200000, seed=42):
    """
    Compute the total length of the union of 8 intervals that sweep as c goes from 0 to 1.
    f: array of shape (8,) with f[i] being the base x-position for tube i.
    The interval for tube i at height c is [f[i] + c*i/8, f[i] + c*i/8 + 1/8].
    We want the volume of the union in the (x, c) plane, which is the integral over c of the 1D union length.
    We approximate this integral using Monte Carlo or deterministic quadrature.
    Actually, the "volume" V_x is the area of the union of the 8 parallelograms in the (x, c) plane.
    V_x = \int_0^1 L(c) dc, where L(c) is the length of the union of intervals at height c.
    """
    N = 8
    i_vals = np.arange(N, dtype=np.float64)
    
    # We can compute this exactly or via high-accuracy sampling.
    # Since the intervals are linear in c, the union length L(c) is a piecewise linear function of c.
    # The breakpoints occur when an endpoint of one interval meets an endpoint of another.
    # Endpoints are:
    # Left_i(c) = f[i] + c * i / N
    # Right_i(c) = f[i] + c * i / N + 1/N
    #
    # Left_i(c) = Left_j(c) => f[i] + c*i/N = f[j] + c*j/N => c = N*(f[j]-f[i]) / (i-j)
    # Left_i(c) = Right_j(c) => f[i] + c*i/N = f[j] + c*j/N + 1/N => c = N*(f[j] + 1/N - f[i]) / (i-j)
    # Right_i(c) = Left_j(c) => f[i] + c*i/N + 1/N = f[j] + c*j/N => c = N*(f[j] - f[i] - 1/N) / (i-j)
    # Right_i(c) = Right_j(c) => f[i] + c*i/N + 1/N = f[j] + c*j/N + 1/N => c = N*(f[j]-f[i]) / (i-j)
    
    # So we just need to collect all critical c values in [0, 1].
    # There are 8 intervals, so 8*8 = 64 pairs.
    # For each pair (i, j) with i != j, we have up to 4 critical values.
    
    critical_c = [0.0, 1.0]
    
    for i in range(N):
        for j in range(N):
            if i == j: continue
            denom = i - j
            if denom == 0: continue
            
            # L_i = L_j
            c1 = N * (f[j] - f[i]) / denom
            # L_i = R_j
            c2 = N * (f[j] + 1.0/N - f[i]) / denom
            # R_i = L_j
            c3 = N * (f[j] - f[i] - 1.0/N) / denom
            # R_i = R_j
            c4 = N * (f[j] - f[i]) / denom
            
            for c_val in [c1, c2, c3, c4]:
                if 0.0 < c_val < 1.0:
                    critical_c.append(c_val)
    
    # Sort and unique
    critical_c = np.unique(np.array(critical_c))
    
    # If there are no critical points in (0,1), the union structure is constant.
    # We can just evaluate at c=0.5.
    # Otherwise, we evaluate L(c) at midpoints of each interval defined by critical_c.
    
    # To be safe and fast, let's just use a dense grid if the number of critical points is small.
    # Max critical points ~ 8*7*4 = 224.
    # So we have ~225 intervals.
    # We can evaluate L(c) at the midpoint of each interval.
    # L(c) is linear between breakpoints, so midpoint rule is exact for the integral over that sub-interval?
    # No, L(c) is piecewise linear, so the integral over a sub-interval where it's linear is exactly 
    # (width) * (L(mid)).
    
    # So V_x = sum_{k} (c_{k+1} - c_k) * L( (c_k + c_{k+1}) / 2 )
    
    V_x = 0.0
    for k in range(len(critical_c) - 1):
        c_mid = (critical_c[k] + critical_c[k+1]) / 2.0
        # Compute L(c_mid)
        starts = f + c_mid * (i_vals / N)
        ends = starts + 1.0 / N
        
        # Sort by start
        idx = np.argsort(starts)
        s = starts[idx]
        e = ends[idx]
        
        # Sweep line
        current_max = s[0]
        length = 0.0
        for m in range(1, N):
            if s[m] > current_max:
                length += current_max - s[m-1] # Wait, this logic is slightly off for general union
                # Standard union length algorithm:
                # Keep track of current_end
                # If s[m] > current_end:
                #    length += current_end - s[m-1] is wrong.
                #    length += current_end - previous_start? No.
                #    
                # Correct:
                # length = 0
                # current_end = e[0]
                # for m in 1..N-1:
                #    if s[m] > current_end:
                #       length += current_end - s[m-1] ... no, length += current_end - s[m-1] is not right.
                #       length += current_end - s[m-1] is the length of the previous interval?
                #       No, length += current_end - s[m-1] is wrong.
                #       length += current_end - s[m-1] is the gap?
                #       
                # Let's restart the union length calc.
                # s, e are sorted by s.
                # total = 0
                # current_end = e[0]
                # for m in 1..N-1:
                #    if s[m] > current_end:
                #       total += current_end - s[m-1] ... NO.
                #       total += current_end - s[m-1] is the length of the segment from s[m-1] to current_end?
                #       No, we are adding the length of the union up to current_end.
                #       The union so far is [s[0], current_end].
                #       If s[m] > current_end, we add (current_end - s[m-1])? No, we add (current_end - s[m-1]) is wrong.
                #       We add (current_end - s[m-1]) is the length of the last interval?
                #       No, we add (current_end - s[m-1]) is not right.
                #       
                # Correct logic:
                # total = 0
                # current_start = s[0]
                # current_end = e[0]
                # for m in 1..N-1:
                #    if s[m] > current_end:
                #       total += current_end - current_start
                #       current_start = s[m]
                #       current_end = e[m]
                #    else:
                #       current_end = max(current_end, e[m])
                # total += current_end - current_start
                
                pass # I will implement this properly below.
        
        # Proper implementation
        total_len = 0.0
        cur_s = s[0]
        cur_e = e[0]
        for m in range(1, N):
            if s[m] > cur_e:
                total_len += cur_e - cur_s
                cur_s = s[m]
                cur_e = e[m]
            else:
                if e[m] > cur_e:
                    cur_e = e[m]
        total_len += cur_e - cur_s
        
        V_x += (critical_c[k+1] - critical_c[k]) * total_len
        
    return V_x

def _volume_objective(params, n_samples=100000, seed=12345):
    """
    params: (16,) -> f[0:8], g[8:16]
    Returns estimated 3D volume V.
    """
    f = params[:8]
    g = params[8:16]
    
    V_x = _union_length_1d(f)
    V_y = _union_length_1d(g)
    
    # The 3D volume is V_x * V_y ONLY IF the x and y movements are independent and the union
    # in 3D is the product of the unions in x and y.
    # Tube k=(i,j) is [x_k + c*i/8, x_k + c*i/8 + 1/8] x [y_k + c*j/8, y_k + c*j/8 + 1/8] x [c, c+dc]
    # The set is a union of boxes (parallelepipeds).
    # Is the volume of the union equal to V_x * V_y?
    # Let S_x be the union of the x-projections (parallelograms in x-c plane).
    # Let S_y be the union of the y-projections (parallelograms in y-c plane).
    # The 3D set is {(x,y,c) : (x,c) in S_x AND (y,c) in S_y}.
    # Yes! Because for a fixed c, the x-interval is determined by i, and y-interval by j.
    # But wait, the tube (i,j) exists only if there is a base position (x_k, y_k) for that pair.
    # In our construction, x_k = f[i] and y_k = g[j].
    # So for a fixed c, the x-coordinate is in the union of intervals I_i(c) = [f[i] + c*i/8, f[i] + c*i/8 + 1/8].
    # The y-coordinate is in the union of intervals J_j(c) = [g[j] + c*j/8, g[j] + c*j/8 + 1/8].
    # AND we need (x,y) to come from a tube that exists.
    # The tubes are all pairs (i,j). So for any i and j, there is a tube.
    # Thus, at height c, the cross-section is (Union_i I_i(c)) x (Union_j J_j(c)).
    # This is exactly the Cartesian product of the 1D unions at height c.
    # Therefore, the volume is \int_0^1 L_x(c) * L_y(c) dc.
    
    # BUT, V_x = \int L_x(c) dc and V_y = \int L_y(c) dc.
    # Is \int L_x(c) L_y(c) dc equal to (\int L_x(c) dc) * (\int L_y(c) dc)?
    # NO. That would only be true if L_x and L_y were constant or uncorrelated in a specific way.
    # Generally, \int A(c) B(c) dc != (\int A(c) dc) (\int B(c) dc).
    
    # So the factorization V = V_x * V_y is INCORRECT for the volume.
    # The volume is \int_0^1 L_x(c) L_y(c) dc.
    
    # However, we can still optimize f and g separately if we approximate the integral.
    # Or we can compute the integral exactly using the critical points of both L_x and L_y.
    # The critical points for L_x are c_x, for L_y are c_y.
    # The function L_x(c) L_y(c) is piecewise bilinear? No, L_x is piecewise linear, L_y is piecewise linear.
    # Their product is piecewise quadratic.
    # We can integrate exactly over the combined grid of critical points.
    
    # Let's implement the exact integral.
    
    # Get critical points for x
    N = 8
    i_vals = np.arange(N, dtype=np.float64)
    
    def get_criticals(f):
        crit = [0.0, 1.0]
        for i in range(N):
            for j in range(N):
                if i == j: continue
                denom = i - j
                c1 = N * (f[j] - f[i]) / denom
                c2 = N * (f[j] + 1.0/N - f[i]) / denom
                c3 = N * (f[j] - f[i] - 1.0/N) / denom
                c4 = N * (f[j] - f[i]) / denom
                for c_val in [c1, c2, c3, c4]:
                    if 0.0 < c_val < 1.0:
                        crit.append(c_val)
        return np.unique(np.array(crit))
    
    crit_x = get_criticals(f)
    crit_y = get_criticals(g)
    
    # Combined critical points
    crit_all = np.unique(np.concatenate([crit_x, crit_y]))
    
    V = 0.0
    for k in range(len(crit_all) - 1):
        c1 = crit_all[k]
        c2 = crit_all[k+1]
        dc = c2 - c1
        
        # In this interval, L_x(c) is linear: L_x(c) = a_x * c + b_x
        # L_y(c) is linear: L_y(c) = a_y * c + b_y
        # We need \int_{c1}^{c2} (a_x c + b_x)(a_y c + b_y) dc
        
        # Evaluate L_x and L_y at two points to determine the linear coefficients?
        # Or just evaluate at midpoint and use Simpson's? 
        # Since it's quadratic, we can use the 3-point rule or just integrate analytically.
        # Let's evaluate at c1, c2, and midpoint to get the quadratic coefficients?
        # Or simpler: evaluate L_x and L_y at c1 and c2.
        # L_x(c) = L_x(c1) + (L_x(c2)-L_x(c1))/(c2-c1) * (c-c1)
        # Let u = c - c1. L_x = L_x1 + m_x u. L_y = L_y1 + m_y u.
        # Integral = \int_0^{dc} (L_x1 + m_x u)(L_y1 + m_y u) du
        # = \int_0^{dc} (L_x1 L_y1 + (L_x1 m_y + L_y1 m_x) u + m_x m_y u^2) du
        # = L_x1 L_y1 dc + (L_x1 m_y + L_y1 m_x) dc^2 / 2 + m_x m_y dc^3 / 3
        
        # Compute L_x(c1), L_x(c2)
        def L_at(f, c_val):
            starts = f + c_val * (i_vals / N)
            ends = starts + 1.0 / N
            idx = np.argsort(starts)
            s = starts[idx]
            e = ends[idx]
            
            total_len = 0.0
            cur_s = s[0]
            cur_e = e[0]
            for m in range(1, N):
                if s[m] > cur_e:
                    total_len += cur_e - cur_s
                    cur_s = s[m]
                    cur_e = e[m]
                else:
                    if e[m] > cur_e:
                        cur_e = e[m]
            total_len += cur_e - cur_s
            return total_len
        
        Lx1 = L_at(f, c1)
        Lx2 = L_at(f, c2)
        Ly1 = L_at(g, c1)
        Ly2 = L_at(g, c2)
        
        mx = (Lx2 - Lx1) / dc if dc > 1e-12 else 0.0
        my = (Ly2 - Ly1) / dc if dc > 1e-12 else 0.0
        
        V += Lx1 * Ly1 * dc + (Lx1 * my + Ly1 * mx) * dc**2 / 2.0 + mx * my * dc**3 / 3.0
        
    return V

def kakeya_needle_3d8() -> np.ndarray:
    deadline = time.time() + 580
    
    # Since we can compute the volume exactly (deterministically), we can use a gradient-based
    # or local search method more effectively.
    # The objective is non-smooth (due to the union length changing structure), but we can
    # use a stochastic local search or differential evolution.
    # Given the low dimensionality (16), DE is still good.
    # But we can use more iterations since each eval is fast (exact integration).
    
    bounds = [(-1.0, 1.0)] * 16
    
    # Initial guess: spread out to minimize overlap.
    # f[i] = -i/16, g[j] = -j/16
    init_params = np.zeros(16)
    i_vals = np.arange(8)
    init_params[:8] = -i_vals / 16.0
    init_params[8:] = -i_vals / 16.0
    
    # Let's try a simple local search or DE.
    # DE with exact objective.
    
    from scipy.optimize import differential_evolution
    
    def objective(params):
        return _volume_objective(params)
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20, 
            tol=1e-8,
            polish=False,
            init='sobol'
        )
        best_params = result.x
    except Exception:
        best_params = init_params
    
    # Polish with Nelder-Mead
    from scipy.optimize import minimize
    
    if time.time() < deadline - 30:
        try:
            res = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < objective(best_params):
                best_params = res.x
        except Exception:
            pass
    
    f = best_params[:8]
    g = best_params[8:16]
    
    N = 8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    x = f[i_flat]
    y = g[j_flat]
    
    positions = np.stack([x, y], axis=1)
    
    return positions
