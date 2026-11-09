import numpy as np
from scipy.optimize import minimize
import time

def _build_hamiltonian(n, coupling, field):
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    for i in range(n - 1):
        op = np.array([[1.0]], dtype=np.complex128)
        for q in range(n):
            op = np.kron(op, Z if q in (i, i+1) else I)
        H += -coupling * op
    
    for i in range(n):
        op = np.array([[1.0]], dtype=np.complex128)
        for q in range(n):
            op = np.kron(op, X if q == i else I)
        H += -field * op
    
    return H

def _apply_ry(state, n, q, theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    new_state = np.zeros_like(state)
    bit = 1 << (n - 1 - q)
    for idx in range(2 ** n):
        amp = state[idx]
        if amp == 0:
            continue
        if idx & bit == 0:
            new_state[idx] += c * amp
            new_state[idx | bit] += s * amp
        else:
            new_state[idx ^ bit] += -s * amp
            new_state[idx] += c * amp
    return new_state

def _apply_cz_layer(state, n):
    new_state = np.zeros_like(state)
    for idx in range(2 ** n):
        amp = state[idx]
        if amp == 0:
            continue
        phase = 1.0
        for i in range(n - 1):
            b1 = (idx >> (n - 1 - i)) & 1
            b2 = (idx >> (n - 1 - (i + 1))) & 1
            if b1 and b2:
                phase *= -1
        new_state[idx] = phase * amp
    return new_state

def _circuit_state(thetas, n):
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    layers = thetas.shape[0]
    for l in range(layers):
        for i in range(n):
            state = _apply_ry(state, n, i, thetas[l, i])
        if l < layers - 1:
            state = _apply_cz_layer(state, n)
    return state

def _energy_and_grad(thetas, n, H, E0):
    state = _circuit_state(thetas, n)
    energy = np.real(np.vdot(state, H @ state))
    
    grad = np.zeros_like(thetas)
    eps = 1e-7
    for l in range(thetas.shape[0]):
        for i in range(n):
            tp = thetas.copy()
            tp[l, i] += eps
            sp = _circuit_state(tp, n)
            ep = np.real(np.vdot(sp, H @ sp))
            
            tm = thetas.copy()
            tm[l, i] -= eps
            sm = _circuit_state(tm, n)
            em = np.real(np.vdot(sm, H @ sm))
            
            grad[l, i] = (ep - em) / (2 * eps)
    
    return energy - E0, grad

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    E0 = eigvals[0]
    
    best_theta = None
    best_energy = np.inf
    
    inits = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 4),
        np.full((layers, n), np.pi / 6),
        np.random.default_rng(0).uniform(-np.pi/2, np.pi/2, (layers, n)),
        np.random.default_rng(1).uniform(-np.pi/2, np.pi/2, (layers, n)),
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
        
        def obj(x):
            return _energy_and_grad(x.reshape(layers, n), n, H, E0)[0]
        
        def grad(x):
            return _energy_and_grad(x.reshape(layers, n), n, H, E0)[1].flatten()
        
        try:
            res = minimize(obj, init.flatten(), jac=grad, method='L-BFGS-B',
                          options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-10})
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            continue
    
    if best_theta is None:
        best_theta = np.zeros((layers, n))
    
    return best_theta
