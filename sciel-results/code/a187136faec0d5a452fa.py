import numpy as np
from scipy.optimize import least_squares
import time

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_(i+1) - field * sum X_i
    Basis: |b_{n-1} ... b_1 b_0>, where b_i is the state of qubit i.
    Index k corresponds to binary representation of k.
    """
    dim = 2**n
    H = np.zeros((dim, dim), dtype=float)
    
    # Z_i Z_(i+1) terms
    for i in range(n - 1):
        for k in range(dim):
            bit_i = (k >> i) & 1
            bit_ip1 = (k >> (i + 1)) & 1
            z_prod = 1.0 if bit_i == bit_ip1 else -1.0
            H[k, k] -= coupling * z_prod
            
    # X_i terms
    for i in range(n):
        for k in range(dim):
            # X_i flips bit i
            k_flip = k ^ (1 << i)
            H[k, k_flip] -= field
            
    return H

def _apply_ry(state, theta, qubit, n):
    """
    Apply Ry(theta) to a specific qubit in the state vector.
    state: 1D array of size 2**n
    theta: rotation angle
    qubit: index of the qubit (0 is LSB)
    n: number of qubits
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    # Reshape state to tensor
    state_tensor = state.reshape([2] * n)
    
    # Move the qubit axis to the front for easier manipulation
    # axes = [qubit, 0, 1, ..., qubit-1, qubit+1, ..., n-1]
    axes = [qubit] + [i for i in range(n) if i != qubit]
    state_moved = np.transpose(state_tensor, axes)
    
    # Apply rotation matrix
    # state_moved has shape (2, 2, ..., 2)
    # We want to apply R = [[c, -s], [s, c]] to the first axis
    # new_state[0, ...] = c * state_moved[0, ...] - s * state_moved[1, ...]
    # new_state[1, ...] = s * state_moved[0, ...] + c * state_moved[1, ...]
    
    new_state_moved = np.empty_like(state_moved)
    new_state_moved[0] = c * state_moved[0] - s * state_moved[1]
    new_state_moved[1] = s * state_moved[0] + c * state_moved[1]
    
    # Move axes back
    inv_axes = np.argsort(axes)
    state_tensor_new = np.transpose(new_state_moved, inv_axes)
    
    return state_tensor_new.reshape(-1)

def _apply_cz(state, qubit1, qubit2, n):
    """
    Apply CZ gate to two qubits.
    CZ = |0><0| + |1><1| on q1, q2? No, CZ is diag(1, 1, 1, -1) in basis |00>, |01>, |10>, |11>.
    It multiplies amplitude by -1 if both qubits are 1.
    """
    state_tensor = state.reshape([2] * n)
    
    # We need to iterate over all basis states or use slicing.
    # Slicing is more efficient.
    # Move qubit1 and qubit2 to the front.
    other_qubits = [i for i in range(n) if i != qubit1 and i != qubit2]
    axes = [qubit1, qubit2] + other_qubits
    state_moved = np.transpose(state_tensor, axes)
    
    # Apply -1 to the component where qubit1=1 and qubit2=1
    state_moved[1, 1] *= -1.0
    
    # Move back
    inv_axes = np.argsort(axes)
    state_tensor_new = np.transpose(state_moved, inv_axes)
    
    return state_tensor_new.reshape(-1)

def _simulate_circuit(theta, n, layers):
    """
    Simulate the circuit defined by theta (layers, n).
    Start with |0>^n.
    Layer l: Ry(theta[l]) on all qubits.
    If l < layers-1: CZ on all adjacent pairs (0,1), (1,2), ..., (n-2, n-1).
    """
    dim = 2**n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0 # |0...0>
    
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            state = _apply_ry(state, theta[l, i], i, n)
            
        # Apply CZ if not the last layer
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1, n)
                
    return state

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 110.0 # 10 second margin per call? No, total 120s for all calls?
    # The prompt says "Each evaluation must finish within 120 s".
    # The function is called once per configuration? Or once per (n, coupling, field)?
    # The task says "Return circuit parameters...". The interface is angles(n, coupling, field, layers).
    # The evaluator likely calls this function for each test case.
    # We assume each call has its own 120s budget.
    
    dim = 2**n
    H = _build_hamiltonian(n, coupling, field)
    
    # Find ground state
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_energy = eigenvalues[0]
    psi_gs = eigenvectors[:, 0]
    
    # Ensure psi_gs is real (it should be for real symmetric H)
    # eigh returns real eigenvectors for real symmetric matrices
    psi_gs = psi_gs.real
    
    # Define residual function for least_squares
    # We fit the circuit state to the ground state.
    # Since global phase is irrelevant, we can align the phase.
    # However, both are real, so we just match components.
    # To handle potential sign flips, we can minimize sum((psi_c - psi_gs)^2)
    # or sum((psi_c + psi_gs)^2) and take the min?
    # Least squares will handle the sign if we allow it, but the initial guess matters.
    # Actually, if we start near the correct state, it will converge to the correct sign.
    # If it converges to the anti-state, the energy is the same, but the "fit" is bad.
    # However, the objective in the task is "mean energy gap".
    # Energy is invariant to global phase/sign.
    # So fitting to psi_gs or -psi_gs yields the same energy.
    # We can just fit to psi_gs.
    
    def residuals(theta_flat):
        if time.time() > deadline:
            return None # Signal timeout
        
        theta = theta_flat.reshape((layers, n))
        psi_c = _simulate_circuit(theta, n, layers)
        
        # Return residuals
        return psi_c - psi_gs

    # Initial guesses
    # 1. All zeros
    # 2. All pi/2
    # 3. Random
    # 4. Based on local X expectation in GS?
    #    <X_i> in GS can be estimated.
    #    For a product state, <X_i> = sin(theta_last_i).
    #    So theta_last_i = arcsin(<X_i>_gs).
    #    Let's compute <X_i>_gs.
    
    # Compute <X_i> in ground state
    x_expectations = np.zeros(n)
    for i in range(n):
        # Sum psi[k] * psi[k_flip_i]
        # k_flip_i = k ^ (1 << i)
        # This is efficient enough for n=6
        for k in range(dim):
            k_flip = k ^ (1 << i)
            # X_i is symmetric, so <X_i> = sum_k psi[k] psi[k^flip_i]
            # Wait, X_i |k> = |k^flip_i>.
            # <psi| X_i |psi> = sum_k conj(psi[k]) psi[k^flip_i]
            # Since psi is real: sum_k psi[k] psi[k^flip_i]
            # We sum over all k. Each pair (k, k^flip) is counted twice?
            # No, k and k^flip are distinct.
            # sum_{k} psi[k] psi[k^flip] = 2 * sum_{k < k^flip} ...
            # Actually, just summing over all k is fine, it gives the correct expectation.
            x_expectations[i] += psi_gs[k] * psi_gs[k_flip]
        x_expectations[i] /= 1.0 # Normalization is 1
    
    # Clip to [-1, 1] for arcsin
    x_expectations = np.clip(x_expectations, -1.0, 1.0)
    theta_guess_4 = np.zeros((layers, n))
    theta_guess_4[-1] = np.arcsin(x_expectations)
    
    guesses = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 2),
        np.random.uniform(-np.pi, np.pi, (layers, n)),
        theta_guess_4
    ]
    
    best_theta = None
    best_cost = np.inf
    
    for guess in guesses:
        if time.time() > deadline:
            break
            
        try:
            result = least_squares(
                residuals, 
                guess.ravel(), 
                method='lm', # Levenberg-Marquardt, good for least squares
                max_nfev=1000
            )
            
            # Check if it timed out or failed
            if result.fun is None:
                continue
                
            cost = np.sum(result.fun**2)
            if cost < best_cost:
                best_cost = cost
                best_theta = result.x.reshape((layers, n))
                
        except Exception as e:
            # Fallback to other method if LM fails
            try:
                result = least_squares(
                    residuals, 
                    guess.ravel(), 
                    method='trf',
                    max_nfev=1000
                )
                if result.fun is not None:
                    cost = np.sum(result.fun**2)
                    if cost < best_cost:
                        best_cost = cost
                        best_theta = result.x.reshape((layers, n))
            except:
                pass

    if best_theta is None:
        # Fallback to seed behavior
        best_theta = np.zeros((layers, n))
        best_theta[-1] = np.pi / 2
        
    return best_theta
