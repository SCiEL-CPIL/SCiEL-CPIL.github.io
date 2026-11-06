"""Quantum circuit synthesis with exact algebraic decompositions and optimized generic synthesis."""
import math
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 250.0  # Safety margin for 270s limit


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build the unitary matrix from a gate list using in-place operations."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Apply single-qubit gate using in-place row operations
            # We need to be careful with the order of operations if we do in-place.
            # It is safer to create a new matrix or use a temporary buffer for the affected rows.
            # For n=4 or 8, copying is cheap.
            U_new = U.copy()
            bit_pos = qubits - 1 - q
            for i in range(n):
                if (i >> bit_pos) & 1:
                    # Qubit is 1
                    i0 = i ^ (1 << bit_pos)
                    # U[i] = single[1,0]*U[i0] + single[1,1]*U[i]
                    # U[i0] = single[0,0]*U[i0] + single[0,1]*U[i]
                    U_new[i] = single[1, 0] * U[i0] + single[1, 1] * U[i]
                    U_new[i0] = single[0, 0] * U[i0] + single[0, 1] * U[i]
            U = U_new
        else:
            _, control, target = gate
            # CNOT: swap rows where control bit is 1
            index = np.arange(n)
            c_bit = (index >> (qubits - 1 - control)) & 1
            t_bit_pos = qubits - 1 - target
            # Flip target bit if control is 1
            new_indices = index ^ (c_bit << t_bit_pos)
            U = U[new_indices, :]
    return U


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _h(q):
    return ['u3', q, math.pi / 2, 0, math.pi]

def _s(q):
    return ['u3', q, 0, 0, math.pi / 2]

def _sdag(q):
    return ['u3', q, 0, 0, -math.pi / 2]

def _t(q):
    return ['u3', q, math.pi / 4, 0, 0]

def _tdag(q):
    return ['u3', q, -math.pi / 4, 0, 0]


def _exact_cnot_01():
    return [['cx', 0, 1]]

def _exact_cnot_10():
    return [['cx', 1, 0]]

def _exact_iswap_2qubit():
    """iSWAP decomposition with 2 CNOTs."""
    gates = []
    gates.append(_h(0))
    gates.append(_h(1))
    gates.append(['cx', 0, 1])
    gates.append(_sdag(0))
    gates.append(_s(1))
    gates.append(['cx', 0, 1])
    gates.append(_h(0))
    gates.append(_h(1))
    return gates

def _exact_swap_2qubit():
    """SWAP decomposition with 3 CNOTs."""
    return [['cx', 0, 1], ['cx', 1, 0], ['cx', 0, 1]]

def _exact_toffoli_3qubit(c1, c2, t):
    """Toffoli decomposition with 6 CNOTs for controls c1,c2 and target t."""
    gates = []
    gates.append(_h(t))
    gates.append(['cx', c1, t])
    gates.append(_tdag(t))
    gates.append(_t(c2))
    gates.append(['cx', c2, t])
    gates.append(_t(t))
    gates.append(_tdag(c2))
    gates.append(['cx', c1, t])
    gates.append(_t(c2))
    gates.append(_tdag(t))
    gates.append(_h(t))
    gates.append(['cx', c1, c2])
    gates.append(_t(c1))
    gates.append(_tdag(c2))
    gates.append(['cx', c1, c2])
    return gates


def _is_known_gate(name, target, qubits):
    """Check if target matches a known gate and return exact decomposition."""
    name_lower = name.lower().replace('-', '').replace('_', '').replace(' ', '')
    
    if qubits == 2:
        if 'cnot' in name_lower or 'cx' in name_lower:
            ref = _build_fast(_exact_cnot_01(), 2)
            if _distance(target, ref) < 1e-10:
                return _exact_cnot_01()
            ref = _build_fast(_exact_cnot_10(), 2)
            if _distance(target, ref) < 1e-10:
                return _exact_cnot_10()
        
        if 'iswap' in name_lower:
            ref = _build_fast(_exact_iswap_2qubit(), 2)
            if _distance(target, ref) < 1e-10:
                return _exact_iswap_2qubit()
        
        if 'swap' in name_lower:
            ref = _build_fast(_exact_swap_2qubit(), 2)
            if _distance(target, ref) < 1e-10:
                return _exact_swap_2qubit()
    
    if qubits == 3:
        if 'toffoli' in name_lower or 'ccnot' in name_lower:
            for c1 in range(3):
                for c2 in range(3):
                    if c2 == c1:
                        continue
                    t = 3 - c1 - c2
                    ref = _build_fast(_exact_toffoli_3qubit(c1, c2, t), 3)
                    if _distance(target, ref) < 1e-10:
                        return _exact_toffoli_3qubit(c1, c2, t)
    
    return None


def _optimize_params(target, qubits, layout, num_params):
    """
    Optimize the parameters of the U3 gates in the layout to match the target.
    layout: list of gates where U3 gates are placeholders ['u3', q, 0, 0, 0]
    """
    
    def objective(params):
        if time.time() > _DEADLINE - 5:
            return 1.0 # Timeout penalty
        
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                # params[k], params[k+1], params[k+2] are theta, phi, lam
                gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(gate)
        
        U = _build_fast(gates, qubits)
        return _distance(target, U)

    # Initial guess: all zeros
    x0 = np.zeros(num_params)
    
    # Use L-BFGS-B for bounded optimization
    # Parameters are angles, so bounds are [-pi, pi]
    bounds = [(-math.pi, math.pi)] * num_params
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20})
        return res.x
    except Exception:
        return x0


def _generic_synthesize_2qubit(target, budget):
    """
    Synthesize a 2-qubit unitary.
    Standard decomposition: U = (A x B) CX (C x D)
    This requires 4 U3 gates (12 params) and 1 CNOT.
    If budget < 1, we can only do single qubit gates (impossible for generic 2q).
    If budget >= 1, we use 1 CNOT.
    """
    if budget < 1:
        # Fallback: just identity or best single qubit fit (unlikely to be needed)
        return []
    
    # Layout: U3(0), U3(1), CX(0,1), U3(0), U3(1)
    layout = [
        ['u3', 0, 0, 0, 0],
        ['u3', 1, 0, 0, 0],
        ['cx', 0, 1],
        ['u3', 0, 0, 0, 0],
        ['u3', 1, 0, 0, 0]
    ]
    
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    num_params = 3 * num_u3
    
    optimal_params = _optimize_params(target, 2, layout, num_params)
    
    # Construct final gate list
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], optimal_params[k], optimal_params[k+1], optimal_params[k+2]])
            k += 3
        else:
            gates.append(gate)
            
    return gates


def _generic_synthesize_3qubit(target, budget):
    """
    Synthesize a 3-qubit unitary.
    Uses a CNOT ladder structure.
    For 6 CNOTs, a common efficient layout is:
    U3s, CX(0,1), U3s, CX(1,2), U3s, CX(0,1), U3s, CX(1,2), U3s, CX(0,1), U3s, CX(1,2), U3s
    Total CNOTs: 6.
    Total U3s: 7 layers * 3 qubits = 21 U3s = 63 params.
    """
    
    # Determine number of CNOT layers based on budget
    # Max CNOTs for 3q generic is typically 6 or 7.
    # We will use min(budget, 6) CNOTs to keep it efficient, as 6 is often sufficient.
    num_cnots = min(budget, 6)
    
    # Define CNOT sequence
    # 0->1, 1->2, 0->1, 1->2, 0->1, 1->2
    cnot_seq = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
    
    layout = []
    # Initial U3s
    for q in range(3):
        layout.append(['u3', q, 0, 0, 0])
    
    for i in range(num_cnots):
        c, t = cnot_seq[i % len(cnot_seq)]
        layout.append(['cx', c, t])
        for q in range(3):
            layout.append(['u3', q, 0, 0, 0])
            
    # If we have budget for more, we can add more, but 6 is a good sweet spot.
    # If budget > 6, we might try 7, but 6 is standard for "generic" in many contexts.
    # Let's stick to 6 to ensure convergence within time.
    
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    num_params = 3 * num_u3
    
    optimal_params = _optimize_params(target, 3, layout, num_params)
    
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], optimal_params[k], optimal_params[k+1], optimal_params[k+2]])
            k += 3
        else:
            gates.append(gate)
            
    return gates


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check for known gates first
    known = _is_known_gate(name, target, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == 'cx')
        if cnot_count <= budget:
            return known
    
    # Generic synthesis
    if qubits == 2:
        return _generic_synthesize_2qubit(target, budget)
    elif qubits == 3:
        return _generic_synthesize_3qubit(target, budget)
    else:
        # Fallback for other qubit counts (not expected in task)
        return []

