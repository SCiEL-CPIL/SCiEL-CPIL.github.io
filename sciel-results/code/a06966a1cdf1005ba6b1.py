import sympy as sp
import time

def evaluate_z(z_list):
    """
    Evaluates the score for a given list of z_i.
    Returns (B, is_valid).
    """
    try:
        # Convert z_i to Rational
        z_rats = [sp.Rational(float(z)) for z in z_list]
        
        x = sp.Symbol('x')
        alpha = sp.Rational(-1, 2)
        
        # Basis functions: L_{2k}^{alpha}(x) for k=0..25
        # We need the polynomial expressions
        basis = []
        for k in range(26):
            L_2k = sp.laguerre(2 * k, alpha, x)
            basis.append(L_2k)
        
        # Derivatives
        basis_derivs = [sp.diff(b, x) for b in basis]
        
        # Build the system A c = b
        # 26 equations
        # Eq 0: g(0) = 0
        # Eq 1: g'(0) = 1
        # Eq 2i: g(z_i) = 0
        # Eq 2i+1: g'(z_i) = 0
        
        A = sp.zeros(26, 26)
        b = sp.zeros(26, 1)
        
        # Row 0
        for j in range(26):
            A[0, j] = basis[j].subs(x, 0)
        b[0, 0] = 0
        
        # Row 1
        for j in range(26):
            A[1, j] = basis_derivs[j].subs(x, 0)
        b[1, 0] = 1
        
        # Rows 2..25
        for i in range(12):
            z_val = z_rats[i]
            row_val = 2 + 2 * i
            row_der = row_val + 1
            
            for j in range(26):
                A[row_val, j] = basis[j].subs(x, z_val)
                A[row_der, j] = basis_derivs[j].subs(x, z_val)
            
            b[row_val, 0] = 0
            b[row_der, 0] = 0
        
        # Solve
        try:
            c = sp.LUsolve(A, b)
        except Exception:
            return float('inf'), False
            
        # Construct g(x)
        g = sum(c[j] * basis[j] for j in range(26))
        
        # Construct D(x) = x * prod (x - z_i)^2
        D = x
        for z_val in z_rats:
            D = D * (x - z_val) ** 2
        
        # q(x) = g(x) / D(x)
        # Use polynomial division
        q, rem = sp.div(g, D, x)
        
        if rem != 0:
            # Should be 0 if z_i are distinct and not 0
            # If not, the construction is flawed or numerical error in rational conversion?
            # The problem states "exact polynomial division", implying it should divide evenly.
            # If rem != 0, it's invalid.
            return float('inf'), False
            
        # Find real roots of q
        roots = sp.real_roots(q)
        
        # Filter roots with sign change test
        counted_roots = []
        epsilon = sp.Rational(1, 10) ** 198 # 1e-198
        
        for r in roots:
            # Check sign at r - eps and r + eps
            val_minus = q.subs(x, r - epsilon)
            val_plus = q.subs(x, r + epsilon)
            
            # Check if strictly opposite signs
            # Since q is a polynomial with rational coeffs, and r is a root,
            # q(r)=0. The sign change indicates a root of odd multiplicity.
            # The problem says "counts if q has strictly opposite signs".
            if (val_minus > 0 and val_plus < 0) or (val_minus < 0 and val_plus > 0):
                counted_roots.append(float(r))
                
        if not counted_roots:
            B = 0.0
        else:
            B = max(0.0, max(counted_roots))
            
        return B, True

    except Exception:
        return float('inf'), False

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    x = sp.Symbol('x')
    
    candidates = []
    
    # 1. Roots of L_12^(-1/2)(x)
    try:
        L12_m12 = sp.laguerre(12, sp.Rational(-1, 2), x)
        roots = sp.nroots(L12_m12)
        z1 = [float(r) for r in roots if abs(r.as_real_imag()[1]) < 1e-10 and r.as_real_imag()[0] > 0]
        if len(z1) == 12:
            candidates.append(z1)
    except:
        pass
        
    # 2. Scaled roots of L_12^(-1/2)(x)
    if candidates:
        base_z = candidates[0]
        for scale in [0.1, 0.5, 2.0]:
            candidates.append([z * scale for z in base_z])
            
    # 3. Roots of L_12^(0)(x)
    try:
        L12_0 = sp.laguerre(12, 0, x)
        roots = sp.nroots(L12_0)
        z2 = [float(r) for r in roots if abs(r.as_real_imag()[1]) < 1e-10 and r.as_real_imag()[0] > 0]
        if len(z2) == 12:
            candidates.append(z2)
            candidates.append([z * 0.5 for z in z2])
    except:
        pass
        
    # 4. Uniform grids
    candidates.append([0.01 * (i + 1) for i in range(12)])
    candidates.append([0.1 * (i + 1) for i in range(12)])
    
    best_z = None
    best_B = float('inf')
    
    for z in candidates:
        if time.time() > deadline:
            break
            
        # Ensure distinct and positive
        if len(set(z)) != 12 or any(zi <= 0 for zi in z):
            continue
            
        B, valid = evaluate_z(z)
        
        if valid:
            if B < best_B:
                best_B = B
                best_z = z
                
    if best_z is not None:
        return best_z
    else:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]
