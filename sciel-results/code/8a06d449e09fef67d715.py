"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 250.0  # Safety margin before 270s kill


def _time_left():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            # Build operator by kron
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            U = op @ U
        else:
            _, control, target = gate
            # CNOT: for each basis state, if control bit is 1, flip target bit
            # This is a permutation of rows
            idx = np.arange(n)
            cbit = (idx >> (qubits - 1 - control)) & 1
            tbit_pos = 1 << (qubits - 1 - target)
            # New row order: row i goes to row (i XOR (cbit_i * tbit_pos))
            new_idx = idx ^ (cbit * tbit_pos)
            U = U[new_idx, :]
    return U


def _distance(target, matrix):
    d = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, d)


def _known_decomposition(name, qubits, budget):
    """Try to return a known exact decomposition for standard gates."""
    if qubits == 2:
        if name == "CNOT":
            return [["cx", 0, 1]]
        elif name == "iSWAP":
            # iSWAP = |00><00| + i|01><10| + i|10><01| + |11><11|
            # Standard decomposition with 3 CNOTs
            # iSWAP = (I ⊗ H)(CX)(I ⊗ S†)(CX)(H ⊗ I)(H ⊗ I)(I ⊗ S)(I ⊗ H)(CX)(I ⊗ S†)(CX)(H ⊗ I)
            # Simpler known: iSWAP on (0,1):
            # |00> -> |00>, |01> -> i|10>, |10> -> i|01>, |11> -> |11>
            # Using 3 CNOTs:
            gates = [
                ["u3", 0, math.pi / 2, 0, -math.pi / 2],  # H on q0
                ["u3", 1, math.pi / 2, 0, -math.pi / 2],  # H on q1
                ["cx", 0, 1],
                ["u3", 1, math.pi / 2, 0, math.pi / 2],   # S on q1
                ["cx", 0, 1],
                ["u3", 1, math.pi / 2, 0, -math.pi / 2],  # S† on q1
                ["cx", 0, 1],
                ["u3", 0, math.pi / 2, 0, -math.pi / 2],  # H on q0
                ["u3", 1, math.pi / 2, 0, -math.pi / 2],  # H on q1
            ]
            return gates
        elif name == "SWAP":
            return [
                ["cx", 0, 1],
                ["cx", 1, 0],
                ["cx", 0, 1],
            ]
    elif qubits == 3:
        if name == "Toffoli":
            # Toffoli |00><00| + |01><01| + |10><10| + |111><111|
            # Standard 6-CNOT decomposition
            gates = [
                ["u3", 2, math.pi / 2, 0, -math.pi / 2],  # H on target
                ["cx", 1, 2],
                ["u3", 2, 0, 0, -math.pi / 4],            # T† on target
                ["cx", 0, 2],
                ["u3", 2, 0, 0, math.pi / 4],             # T on target
                ["cx", 1, 2],
                ["u3", 2, 0, 0, -math.pi / 4],            # T† on target
                ["cx", 0, 2],
                ["u3", 2, math.pi / 2, 0, -math.pi / 2],  # H on target
                # Additional CNOTs for full Toffoli (this is a partial; let's use a known 6-CNOT)
            ]
            # Actually the standard Toffoli uses 6 CNOTs. Let me use a correct one.
            gates = [
                ["u3", 2, math.pi / 2, 0, -math.pi / 2],
                ["cx", 1, 2],
                ["u3", 2, 0, 0, -math.pi / 4],
                ["cx", 0, 2],
                ["u3", 2, 0, 0, math.pi / 4],
                ["cx", 1, 2],
                ["u3", 2, 0, 0, -math.pi / 4],
                ["cx", 0, 2],
                ["u3", 2, math.pi / 2, 0, -math.pi / 2],
                ["u3", 1, 0, 0, -math.pi / 4],
                ["cx", 0, 1],
                ["u3", 1, 0, 0, math.pi / 4],
                ["cx", 0, 1],
            ]
            return gates
    return None


def synthesize(problem):
    """Synthesize circuit using known decompositions or coordinate descent."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known exact decompositions first
    known = _known_decomposition(name, qubits, budget)
    if known is not None:
        # Verify it's within budget
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            return known
    
    # For generic unitaries, use coordinate descent
    # Build a layout with CNOTs and u3 gates
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    n_params = 0
    layout = []
    
    # Create a reasonable circuit structure
    # Use all available CNOTs in a pattern, with u3 gates before and after each
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            n_params += 3
        layout.append(["cx", *pairs[layer % len(pairs)]])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        n_params += 3
    
    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * n_params
    best_distance = _distance(target, _build_fast(circuit(best), qubits))
    
    # Coordinate descent with adaptive step sizes
    rng = np.random.RandomState(42)
    step_sizes = [1.0] * n_params
    
    max_sweeps = 20
    for sweep in range(max_sweeps):
        if _time_left() < 5.0:
            break
        improved = False
        for i in range(n_params):
            if _time_left() < 2.0:
                break
            current = best[i]
            best_trial = current
            best_dist = best_distance
            
            # Try several perturbations
            for delta in [step_sizes[i], -step_sizes[i], 
                         0.5 * step_sizes[i], -0.5 * step_sizes[i],
                         2.0 * step_sizes[i], -2.0 * step_sizes[i]]:
                trial_val = current + delta
                values = best[:]
                values[i] = trial_val
                dist = _distance(target, _build_fast(circuit(values), qubits))
                if dist < best_dist:
                    best_dist = dist
                    best_trial = trial_val
            
            # Also try random perturbation
            trial_val = current + rng.uniform(-step_sizes[i], step_sizes[i])
            values = best[:]
            values[i] = trial_val
            dist = _distance(target, _build_fast(circuit(values), qubits))
            if dist < best_dist:
                best_dist = dist
                best_trial = trial_val
            
            if best_trial != current:
                best[i] = best_trial
                best_distance = best_dist
                improved = True
                # Reduce step size slightly
                step_sizes[i] *= 0.7
            else:
                # No improvement, reduce step size
                step_sizes[i] *= 0.5
        
        if not improved:
            # All parameters stuck, try smaller steps globally
            step_sizes = [s * 0.5 for s in step_sizes]
            if min(step_sizes) < 1e-6:
                break
    
    # Final polish with very fine steps
    if _time_left() > 3.0:
        for i in range(n_params):
            if _time_left() < 2.0:
                break
            current = best[i]
            best_dist = best_distance
            for delta in [1e-3, -1e-3, 1e-4, -1e-4, 1e-5, -1e-5]:
                values = best[:]
                values[i] = current + delta
                dist = _distance(target, _build_fast(circuit(values), qubits))
                if dist < best_dist:
                    best_dist = dist
                    best[i] = current + delta
    
    return circuit(best)
# EVOLVE-BLOCK-END
