import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 135.0

_N = 6
_DELTA = 0.8
_H = 0.3


def _time_left():
    return _DEADLINE - time.time()


def _matrices():
    # Build the two-body Heisenberg Hamiltonian for a 6-site chain
    # Sx*Sx + Sy*Sy + Delta * Sz*Sz for each bond
    # Plus uniform field term 0.5 * H * Sz for each site
    
    # Single spin matrices
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    A = np.zeros((_N, _N), dtype=complex)
    B = np.zeros((_N, _N), dtype=complex)
    
    for i in range(_N - 1):
        # Build S_i op S_{i+1} terms
        # We need to place operators at positions i and i+1 in a chain of N sites
        # Using tensor products
        op = np.array([[1]], dtype=complex)
        for j in range(_N):
            if j == i or j == (i + 1):
                # Here we need to be careful. 
                # For bond i, we have S_i and S_{i+1}.
                # Let's construct the full operator term by term.
                pass
        
        # A simpler way: iterate over all possible operator combinations on the bond
        # Term 1: Sx_i Sx_{i+1}
        # Term 2: Sy_i Sy_{i+1}
        # Term 3: Delta * Sz_i Sz_{i+1}
        
        def bond_op(op1, op2):
            # op1 acts on site i, op2 acts on site i+1
            # Full operator is I...I op1 op2 I...I
            result = np.array([[1]], dtype=complex)
            for j in range(_N):
                if j == i:
                    result = np.kron(result, op1)
                elif j == (i + 1):
                    result = np.kron(result, op2)
                else:
                    result = np.kron(result, np.eye(2, dtype=complex))
            return result
        
        term_xx = bond_op(Sx, Sx)
        term_yy = bond_op(Sy, Sy)
        term_zz = _DELTA * bond_op(Sz, Sz)
        
        bond_ham = term_xx + term_yy + term_zz
        
        if i % 2 == 0:
            A += bond_ham
        else:
            B += bond_ham
    
    # Add field term: H * sum_i Sz_i
    # The task says "each including half of the uniform field"
    # So A gets 0.5 * H * sum Sz_i and B gets 0.5 * H * sum Sz_i
    field_term = np.zeros((_N, _N), dtype=complex)
    for i in range(_N):
        def site_op(op, site):
            result = np.array([[1]], dtype=complex)
            for j in range(_N):
                if j == site:
                    result = np.kron(result, op)
                else:
                    result = np.kron(result, np.eye(2, dtype=complex))
            return result
        field_term += site_op(Sz, i)
    
    # Wait, the field term is 0.3 * sum Sz_i.
    # The task says "including half of the uniform field".
    # So A += 0.5 * 0.3 * sum Sz_i
    # B += 0.5 * 0.3 * sum Sz_i
    
    field_val = 0.5 * _H * field_term
    A += field_val
    B += field_val
    
    return A, B


def _prod_formula(U, a, b, A, B):
    # U = prod_k exp(-i b_k t B) exp(-i a_k t A)
    # Here t is passed as a scalar multiplier inside the exponentials?
    # The formula is U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
    # So we compute exp(-i * b[k] * t * B) etc.
    # But the function _prod_formula doesn't take t.
    # Let's assume t is passed or the coefficients are pre-scaled.
    # Looking at the original code:
    # U = np.dot(np.exp(-1j * b[k] * B), U)
    # It seems the original code assumed t=1?
    # Let's check the original _true_err:
    # e += np.linalg.norm(U - np.exp(-1j * t * (A + B)), 2)
    # And _prod was called without t, implying the coefficients a,b were used directly.
    # But the task says: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
    # So if I want to evaluate at time t, I should use coefficients a*t, b*t?
    # Or does the function coefficients return a,b such that the formula is for general t?
    # "The evaluator forms the product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)"
    # This means the coefficients a,b are fixed, and t is a scalar parameter.
    # So in my objective function, I must multiply a,b by t.
    
    # However, the original code's _prod function did:
    # U = np.dot(np.exp(-1j * b[k] * B), U)
    # This implies the original code was flawed or I misread.
    # Let's look at _true_err in the original:
    # U = _prod(np.eye(_N, dtype=complex), a, b)
    # It did not pass t.
    # But the error was computed as:
    # e += np.linalg.norm(U - np.exp(-1j * t * (A + B)), 2)
    # This suggests the original code was computing the product formula at t=1 and comparing to t=0.25, 0.5, 1.0?
    # That would be wrong.
    # Let's assume the correct interpretation:
    # The coefficients a,b are time-independent.
    # For a given t, U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A).
    # So in the objective function, for each t in [0.25, 0.5, 1.0],
    # I must compute the product formula using a*t and b*t.
    
    # Let's define a helper that takes t.
    pass


def _prod_formula_t(a, b, t, A, B):
    U = np.eye(_N, dtype=complex)
    for k in range(len(a)):
        U = np.dot(np.exp(-1j * b[k] * t * B), U)
        U = np.dot(np.exp(-1j * a[k] * t * A), U)
    return U


def _objective(x, A, B):
    stages = len(x) // 2
    a = x[:stages]
    b = x[stages:]
    
    # Enforce sum constraints?
    # The problem states sum(a)=1 and sum(b)=1.
    # We can use a penalty method or parameterization.
    # Let's use a penalty for simplicity, or normalize.
    # Normalizing inside the objective is risky for gradients.
    # Let's try normalizing.
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = a / sa
    if abs(sb) > 1e-10:
        b = b / sb
    
    err = 0.0
    for t in (0.25, 0.5, 1.0):
        U = _prod_formula_t(a, b, t, A, B)
        U_exact = np.exp(-1j * t * (A + B))
        err += np.linalg.norm(U - U_exact, 2)
    return err / 3.0


def coefficients(stages):
    A, B = _matrices()
    stages = int(stages)
    if stages < 2:
        stages = 2
    
    # Initial guess: uniform distribution
    # a_k = 1/stages, b_k = 1/stages
    x0 = [1.0 / stages] * (2 * stages)
    
    # Use scipy.optimize.minimize
    # Since we have no gradient info easily, use 'Powell' or 'Nelder-Mead'
    # 'Powell' is often robust for such problems.
    
    best_x = x0
    best_val = _objective(x0, A, B)
    
    # We can try a few different initial guesses if time permits
    # 1. Uniform
    # 2. Symmetric (a_k = a_{N+1-k})
    # 3. Random
    
    methods = ['Powell', 'Nelder-Mead']
    
    for method in methods:
        if _time_left() < 10.0:
            break
            
        try:
            result = minimize(
                _objective, 
                x0, 
                args=(A, B),
                method=method,
                options={'maxiter': 200, 'maxfev': 500}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
                
            # Check time
            if _time_left() < 10.0:
                break
                
        except Exception as e:
            print(f"Optimization failed with {method}: {e}")
            continue
    
    a = best_x[:stages]
    b = best_x[stages:]
    
    # Final normalization
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
        
    return a, b


# Precompute matrices at import? 
# The function coefficients calls _matrices, which is fast.
# But to be safe, let's not do it at import to avoid side effects if called multiple times.
# However, the task says "coefficients runs in a sandboxed subprocess".
# So import happens once.

if __name__ == "__main__":
    # Simple test
    a, b = coefficients(12)
    print("a:", a)
    print("b:", b)
    print("sum a:", sum(a))
    print("sum b:", sum(b))
