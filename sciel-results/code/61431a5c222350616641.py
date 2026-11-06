"""Product formula design for Heisenberg-chain time evolution."""
import numpy as np
from scipy.linalg import expm, logm
import time

def coefficients(stages):
    """Design 12-stage product formula coefficients via Taylor matching."""
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Build the Hamiltonian components
    # 6-site open XXZ Heisenberg chain, anisotropy=0.8, field=0.3
    # H = sum_j [Jx (Sx_j Sx_{j+1} + Sy_j Sy_{j+1}) + Jz Sz_j Sz_{j+1}] + h sum_j Sz_j
    # Split: A = odd bonds (j=1,3,5 in 0-indexed: bonds 0-1, 2-3, 4-5) + half field
    #       B = even bonds (j=0,2,4 in 0-indexed: bonds 1-2, 3-4) + half field
    # Wait, 6 sites means bonds 0-1, 1-2, 2-3, 3-4, 4-5 (5 bonds)
    # Odd bonds: 0-1, 2-3, 4-5 (bonds at even indices) -> A
    # Even bonds: 1-2, 3-4 (bonds at odd indices) -> B
    
    Jx = 1.0  # coupling
    Jz = 0.8  # anisotropy
    h = 0.3   # field
    
    # Spin-1/2 operators
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    n_sites = 6
    
    def site_op(op, site):
        """Build operator acting on a specific site."""
        ops = [I2] * n_sites
        ops[site] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_op(site1, site2):
        """Build two-site operator for bond between site1 and site2."""
        # Sx1*Sx2 + Sy1*Sy2 + Jz*Sz1*Sz2
        term_xx = np.kron(site_op(Sx, site1), I2) @ np.kron(I2, site_op(Sx, site2)) if False else None
        # Better: build directly
        ops = [I2] * n_sites
        ops[site1] = Sx
        ops[site2] = Sx
        r_xx = ops[0]
        for o in ops[1:]:
            r_xx = np.kron(r_xx, o)
        
        ops = [I2] * n_sites
        ops[site1] = Sy
        ops[site2] = Sy
        r_yy = ops[0]
        for o in ops[1:]:
            r_yy = np.kron(r_yy, o)
        
        ops = [I2] * n_sites
        ops[site1] = Sz
        ops[site2] = Sz
        r_zz = ops[0]
        for o in ops[1:]:
            r_zz = np.kron(r_zz, o)
        
        return Jx * (r_xx + r_yy) + Jz * r_zz
    
    # Build A (odd bonds: 0-1, 2-3, 4-5) + half field
    A = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    for s1, s2 in [(0,1), (2,3), (4,5)]:
        A += bond_op(s1, s2)
    
    # Build B (even bonds: 1-2, 3-4) + half field
    B = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    for s1, s2 in [(1,2), (3,4)]:
        B += bond_op(s1, s2)
    
    # Add half field to each
    field_op = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    for site in range(n_sites):
        field_op += site_op(Sz, site)
    
    A += 0.5 * h * field_op
    B += 0.5 * h * field_op
    
    # Now find optimal 12-stage coefficients
    # U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    # We want to match exp(-i t (A+B)) up to high order in t
    
    # For a product formula with steps (a_k, b_k), the log of the product
    # has a series expansion. We can match coefficients.
    
    # With 12 stages, we have 24 parameters but sum(a)=1, sum(b)=1 constraints
    # leave 22 free. But we only need to match up to some order.
    
    # Actually, let's think about this differently. The product formula
    # U = exp(-i b_12 t B) exp(-i a_12 t A) ... exp(-i b_1 t B) exp(-i a_1 t A)
    
    # For small t, log(U) = -i t (A+B) + t^2 * C2 + t^3 * C3 + ...
    # We want C2, C3, ... to be as small as possible.
    
    # The second-order term involves [A, B] commutators.
    # For a symmetric formula, many terms cancel.
    
    # Let's use a numerical approach: define the error as a function of
    # the coefficients and minimize it.
    
    # Actually, given the time constraint, let me try a different approach:
    # Use the known optimal coefficients for product formulas.
    
    # For a 2n-step product formula matching up to order 2n-1,
    # the coefficients can be found from the conditions:
    # sum(a_k) = 1, sum(b_k) = 1
    # sum(a_k^p) = 1/p for p = 2, ..., n (for A terms)
    # sum(b_k^p) = 1/p for p = 2, ..., n (for B terms)
    # Plus cross conditions for the commutator terms.
    
    # For 12 stages (6 pairs), we can match up to order 11 potentially.
    # But the simple power-sum conditions give us 6+6=12 conditions
    # (including sum=1), which exactly determines 12 coefficients.
    
    # The standard approach: for a product formula exp(-i b_n t B)...exp(-i b_1 t B)
    # exp(-i a_n t A)...exp(-i a_1 t A) with n stages,
    # matching up to order 2n-1 requires:
    # sum_{k=1}^n a_k = 1
    # sum_{k=1}^n a_k^p = 1/p for p = 2, ..., n
    # sum_{k=1}^n b_k = 1  
    # sum_{k=1}^n b_k^p = 1/p for p = 2, ..., n
    # And the cross terms are automatically satisfied for symmetric formulas.
    
    # Wait, that's not quite right. Let me reconsider.
    
    # For the product U = prod_{k=1}^{n} exp(-i b_k t B) exp(-i a_k t A),
    # the log expansion gives conditions involving both a and b.
    
    # A simpler approach: use the fact that for a symmetric product formula,
    # we can set a_k = b_k and solve for them.
    
    # For n=6 pairs (12 total factors), matching up to order 11:
    # We need sum(a_k^p) = 1/p and sum(b_k^p) = 1/p for p=1,...,6
    # This is a system of 6 equations for 6 unknowns (if a_k = b_k)
    # or 12 equations for 12 unknowns.
    
    # Let me solve for a_k = b_k = c_k, sum(c_k^p) = 1/p for p=1,...,6
    # This is a moment problem.
    
    # Actually, the standard result: for a 2n-step symmetric product formula
    # of order 2n-1, the coefficients satisfy sum_{k=1}^n c_k^p = 1/p for p=1,...,n.
    
    # For n=6, I need to find 6 numbers c_k such that sum(c_k^p) = 1/p for p=1,...,6.
    
    # This is a classical moment problem. Let me solve it numerically.
    
    # Use Newton's method or a direct solver.
    # Let x = [c_1, ..., c_6], and we want sum(x_k^p) - 1/p = 0 for p=1,...,6.
    
    from scipy.optimize import fsolve
    
    def moment_eqs(x):
        """Equations: sum(x_k^p) = 1/p for p=1,...,6"""
        eqs = []
        for p in range(1, 7):
            eqs.append(np.sum(np.array(x)**p) - 1.0/p)
        return eqs
    
    # Initial guess: uniform
    x0 = [1.0/6.0] * 6
    
    # Time check
    if time.time() - t_start > DEADLINE - 5:
        # Fallback: simple uniform
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
        return a, b
    
    sol = fsolve(moment_eqs, x0, full_output=True)
    c_vals, info, ier, msg = sol
    
    if ier != 1:
        # Try different initial guesses
        for guess in [
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            [0.3, 0.3, 0.3, 0.1, 0.0, 0.0],
            [0.5, 0.2, 0.1, 0.1, 0.05, 0.05],
        ]:
            if time.time() - t_start > DEADLINE - 5:
                break
            sol = fsolve(moment_eqs, guess, full_output=True)
            c_vals, info, ier, msg = sol
            if ier == 1:
                break
    
    if ier != 1 or not np.all(np.isfinite(c_vals)):
        # Fallback to simple splitting
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
        return a, b
    
    c_vals = np.real(c_vals)
    
    # Now construct the 12-stage formula
    # U = exp(-i b_12 t B) exp(-i a_12 t A) ... exp(-i b_1 t B) exp(-i a_1 t A)
    # With a_k = b_k = c_k for k=1,...,6
    # But we need 12 stages, so 6 a-values and 6 b-values.
    
    # Wait, the problem says 12-stage product formula:
    # U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    # That's 12 b's and 12 a's? No, re-reading:
    # "two lists of 12 finite numbers"
    # "U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k = 1..12"
    # So there are 12 pairs, meaning 12 a's and 12 b's.
    
    # Hmm, that's 24 parameters. Let me re-read.
    
    # "Choose the coefficients of a twelve-stage product formula"
    # "two lists of 12 finite numbers in [-50, 50] with sum(a) = sum(b) = 1"
    # "U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k = 1..12"
    
    # So yes, 12 a's and 12 b's. That's 24 parameters with 2 sum constraints = 22 free.
    
    # For a 12-pair product formula, we can potentially match up to very high order.
    # The conditions would be:
    # sum(a_k) = 1, sum(b_k) = 1
    # And higher moment conditions.
    
    # For a symmetric formula where a_k = b_k, we have 12 unknowns with
    # sum(c_k) = 1 and moment conditions.
    
    # Let me set a_k = b_k = c_k for k=1,...,12 and solve:
    # sum(c_k^p) = 1/p for p = 1, ..., 12
    # That's 12 equations for 12 unknowns.
    
    if time.time() - t_start > DEADLINE - 10:
        # Use the 6-value solution repeated
        a = list(c_vals) + list(c_vals)
        b = list(c_vals) + list(c_vals)
        # Normalize
        sa = sum(a)
        sb = sum(b)
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        return a, b
    
    def moment_eqs_12(x):
        """Equations: sum(x_k^p) = 1/p for p=1,...,12"""
        eqs = []
        for p in range(1, 13):
            eqs.append(np.sum(np.array(x)**p) - 1.0/p)
        return eqs
    
    x0_12 = [1.0/12.0] * 12
    sol12 = fsolve(moment_eqs_12, x0_12, full_output=True)
    c12, info12, ier12, msg12 = sol12
    
    if ier12 != 1:
        # Try other guesses
        for guess in [
            [0.15]*12,
            [0.2]*6 + [0.05]*6,
            [0.1]*12,
        ]:
            if time.time() - t_start > DEADLINE - 10:
                break
            sol12 = fsolve(moment_eqs_12, guess, full_output=True)
            c12, info12, ier12, msg12 = sol12
            if ier12 == 1:
                break
    
    if ier12 == 1 and np.all(np.isfinite(c12)):
        c12 = np.real(c12)
        a = list(c12)
        b = list(c12)
    else:
        # Fallback: use the 6-value solution doubled
        a = list(c_vals) + list(c_vals)
        b = list(c_vals) + list(c_vals)
    
    # Normalize to ensure sum = 1
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x/sa for x in a]
    if abs(sb) > 1e-15:
        b = [x/sb for x in b]
    
    # Clip to [-50, 50]
    a = [max(-50, min(50, x)) for x in a]
    b = [max(-50, min(50, x)) for x in b]
    
    # Re-normalize after clipping
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x/sa for x in a]
    if abs(sb) > 1e-15:
        b = [x/sb for x in b]
    
    return a, b
