import numpy as np
from scipy.optimize import minimize
import time

def angles(n, edges, p):
    """
    Compute optimal QAOA angles for a given graph.
    
    Args:
        n: Number of nodes.
        edges: List of edges (u, v).
        p: Number of layers (fixed to 2).
        
    Returns:
        np.array of shape (p, 2) containing (gamma, beta) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute cost vector
    num_states = 1 << n
    cost_vec = np.zeros(num_states, dtype=np.float64)
    indices = np.arange(num_states)
    
    for u, v in edges:
        bu = (indices >> u) & 1
        bv = (indices >> v) & 1
        mask = bu != bv
        cost_vec[mask] += 1
    
    # Precompute the initial state
    init_state = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    # Precompute rotation for each qubit - use a faster approach
    # For R_x(beta) on all qubits, we can use the fact that it's a product of single-qubit gates
    # The state after applying R_x(beta) to all qubits can be computed efficiently
    
    def compute_expected_cut(g1, b1, g2, b2):
        """Compute expected cut for given angles efficiently."""
        # Start with |+>^n
        # Apply cost: multiply by exp(-i*g1*cost_vec)
        # Apply R_x(b1) to all qubits
        # Apply cost: multiply by exp(-i*g2*cost_vec)
        # Apply R_x(b2) to all qubits
        # Compute <cost>
        
        # The state after all operations:
        # |psi> = R_x(b2) U_cost(g2) R_x(b1) U_cost(g1) |+>^n
        
        # Let's compute this step by step using the fact that:
        # U_cost(g) |x> = exp(-i*g*cost(x)) |x>
        # R_x(beta) = cos(beta/2) I - i sin(beta/2) X
        
        # After U_cost(g1): state[x] = exp(-i*g1*cost(x)) / sqrt(2^n)
        # After R_x(b1) on all qubits: this is a product state operation
        # R_x(b) on qubit q: maps |0> -> c|0> - i*s|1>, |1> -> -i*s|0> + c|1>
        # where c = cos(b/2), s = sin(b/2)
        
        # For a product state where each qubit is in state a|0> + b|1>,
        # after R_x(beta): new_a = c*a - i*s*b, new_b = -i*s*a + c*b
        
        # The initial state is (1/sqrt(2))^n |0...0> + ... = product of (|0>+|1>)/sqrt(2) per qubit
        # So each qubit starts in (1/sqrt(2), 1/sqrt(2))
        
        # After U_cost(g1), the state is no longer a product state due to the cost Hamiltonian
        # So we need to track the full state
        
        state = init_state.copy()
        
        # Apply cost g1
        phase1 = np.exp(-1j * g1 * cost_vec)
        state = state * phase1
        
        # Apply R_x(b1) to all qubits
        c1 = np.cos(b1 / 2.0)
        s1 = np.sin(b1 / 2.0)
        
        # For each qubit, apply the rotation
        # We can do this more efficiently using the fact that R_x is diagonal in the X basis
        # But let's use a direct approach with precomputed masks
        
        # For qubit q, the mask for bit q being 1:
        for q in range(n):
            mask_q = (np.arange(num_states) >> q) & 1
            # For states where bit q = 0: new_val = c * old_val - i*s * (val where bit q = 1)
            # For states where bit q = 1: new_val = -i*s * (val where bit q = 0) + c * old_val
            
            # Create indices with bit q flipped
            flip_mask = 1 << q
            idx0 = np.where(mask_q == 0)[0]
            idx1 = np.where(mask_q == 1)[0]
            
            # Pair up states: for each state with bit q=0, pair with same state with bit q=1
            # The flipped index: idx ^ flip_mask
            paired_idx = idx0 ^ flip_mask
            
            v0 = state[idx0]
            v1 = state[idx1]
            
            new_v0 = c1 * v0 - 1j * s1 * v1
            new_v1 = -1j * s1 * v0 + c1 * v1
            
            state[idx0] = new_v0
            state[idx1] = new_v1
        
        # Apply cost g2
        phase2 = np.exp(-1j * g2 * cost_vec)
        state = state * phase2
        
        # Apply R_x(b2) to all qubits
        c2 = np.cos(b2 / 2.0)
        s2 = np.sin(b2 / 2.0)
        
        for q in range(n):
            mask_q = (np.arange(num_states) >> q) & 1
            idx0 = np.where(mask_q == 0)[0]
            idx1 = np.where(mask_q == 1)[0]
            
            v0 = state[idx0]
            v1 = state[idx1]
            
            new_v0 = c2 * v0 - 1j * s2 * v1
            new_v1 = -1j * s2 * v0 + c2 * v1
            
            state[idx0] = new_v0
            state[idx1] = new_v1
        
        # Compute expected cut
        probs = np.abs(state) ** 2
        exp_cut = np.dot(probs, cost_vec)
        
        return exp_cut
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        g1, b1, g2, b2 = params
        # Wrap angles to [0, pi]
        g1 = g1 % np.pi
        b1 = b1 % np.pi
        g2 = g2 % np.pi
        b2 = b2 % np.pi
        return -compute_expected_cut(g1, b1, g2, b2)
    
    # Optimization with multiple strategies
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Known good starting points for QAOA p=2
    # From literature, typical optimal angles are around:
    # g1 ~ 0.3-0.5, b1 ~ 0.2-0.4, g2 ~ 0.3-0.5, b2 ~ 0.1-0.3
    initial_guesses = [
        np.array([0.35, 0.25, 0.35, 0.25]),
        np.array([0.40, 0.30, 0.40, 0.20]),
        np.array([0.30, 0.20, 0.30, 0.30]),
        np.array([0.45, 0.35, 0.45, 0.15]),
        np.array([0.25, 0.15, 0.25, 0.35]),
    ]
    
    # Add some random restarts
    np.random.seed(42)
    for _ in range(5):
        initial_guesses.append(np.random.uniform(0, np.pi, 4))
    
    for x0 in initial_guesses:
        if time.time() > deadline - 15.0:
            break
        
        try:
            # First pass: Nelder-Mead
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7})
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
            
            # Second pass: L-BFGS-B for finer convergence, starting from NM result
            if time.time() < deadline - 10.0:
                res2 = minimize(objective, res.x, method='L-BFGS-B',
                               bounds=[(0, np.pi)] * 4,
                               options={'maxiter': 100, 'gtol': 1e-8})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x.copy()
                    
        except Exception:
            pass
    
    # Final refinement: try a few more random points near the best solution
    if time.time() < deadline - 5.0:
        for _ in range(10):
            perturbation = np.random.normal(0, 0.1, 4)
            trial_params = best_params + perturbation
            trial_params = np.clip(trial_params, 0, np.pi)
            val = -objective(trial_params)
            if val > -best_val:
                best_val = -val
                best_params = trial_params
    
    # Wrap final angles to [0, pi]
    best_params = np.mod(best_params, np.pi)
    
    return best_params.reshape(p, 2)
