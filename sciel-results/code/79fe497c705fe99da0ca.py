import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
        values = [0, 1, 2, 3]
        num_types = 16
        types = [(v1, v2) for v1 in values for v2 in values]
        
        # Parse probs
        probs = np.zeros(num_types)
        # ... parsing logic ...
        
        # LP Setup
        n_vars = 48
        x_off = 0
        u_off = 32
        
        c = np.zeros(n_vars)
        for k in range(num_types):
            v1, v2 = types[k]
            p = probs[k]
            c[x_off + 2*k] = -p * v1
            c[x_off + 2*k + 1] = -p * v2
            c[u_off + k] = p
            
        # Constraints
        # IC: 16*15 = 240 (excluding self) or 256 (including self, which is 0=0). Let's use 256 for simplicity or 240.
        # Using 240 is slightly faster.
        n_ic = 240
        n_ir = 16
        n_pay = 16
        n_cons = n_ic + n_ir + n_pay
        
        A_ub = np.zeros((n_cons, n_vars))
        b_ub = np.zeros(n_cons)
        
        # IC
        idx = 0
        for v_idx in range(num_types):
            v1, v2 = types[v_idx]
            for vp_idx in range(num_types):
                if v_idx == vp_idx: continue
                vp1, vp2 = types[vp_idx]
                A_ub[idx, u_off + v_idx] = -1.0
                A_ub[idx, u_off + vp_idx] = 1.0
                A_ub[idx, x_off + 2*vp_idx] = v1 - vp1
                A_ub[idx, x_off + 2*vp_idx + 1] = v2 - vp2
                b_ub[idx] = 0.0
                idx += 1
                
        # IR: -u_k <= 0
        for k in range(num_types):
            row = n_ic + k
            A_ub[row, u_off + k] = -1.0
            b_ub[row] = 0.0
            
        # Payment Non-Negativity: p_k >= 0 => v_k x_k - u_k >= 0 => -v_k x_k + u_k <= 0
        # Wait. p_k = v x - u. p_k >= 0 => v x - u >= 0 => -v x + u <= 0.
        # Current code:
        # A_ub[row, u_off + k] = 1.0
        # A_ub[row, x_off + 2*k] = -v1
        # A_ub[row, x_off + 2*k + 1] = -v2
        # This is -v1 x1 - v2 x2 + u <= 0.
        # Which is u <= v1 x1 + v2 x2.
        # Which is v x - u >= 0.
        # Correct.
        
        for k in range(num_types):
            row = n_ic + n_ir + k
            v1, v2 = types[k]
            A_ub[row, u_off + k] = 1.0
            A_ub[row, x_off + 2*k] = -v1
            A_ub[row, x_off + 2*k + 1] = -v2
            b_ub[row] = 0.0
            
        bounds = [(0, 1)] * 32 + [(0, None)] * 16
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'disp': False})
        
        if not res.success:
             return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
             
        z = res.x
        allocations = z[x_off:x_off+32].reshape(16, 2)
        utilities = z[u_off:]
        
        payments = np.array([types[k][0]*allocations[k,0] + types[k][1]*allocations[k,1] - utilities[k] for k in range(16)])
        
        # Clean up
        allocations = np.clip(allocations, 0, 1)
        payments = np.clip(payments, 0, None)
        
        return {"allocations": allocations, "payments": payments}
