import numpy as np
from scipy.optimize import minimize
import time

def angles(n, edges, p):
    # Deadline: 110 seconds to be safe
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Precompute cost vector
    # cost_vec[i] is the number of cut edges for the bitstring i
    num_states = 1 << n
    cost_vec = np.zeros(num_states, dtype=np.float64)
    
    # Efficiently compute cost vector
    # For each edge (u, v), if bit u != bit v, add 1
    # We can do this by iterating edges and adding contributions
    # Or iterate all states? 2^10 = 1024. Iterating edges (max ~45) * 1024 is small.
    
    # Better: Iterate all states, check edges.
    # Or: For each edge, generate mask where bits differ.
    # Let's just iterate states for simplicity, n is small.
    
    # Actually, vectorized approach:
    # Create array of all bitstrings
    # bits = np.array([np.binary_repr(i, width=n) for i in range(num_states)], dtype=int)
    # This is slow.
    
    # Faster:
    # For each edge (u, v):
    #   mask = 1 << u | 1 << v
    #   This doesn't directly give "different".
    #   Different if (i >> u) ^ (i >> v) & 1
    
    # Let's use a loop over edges, updating cost_vec
    # Initialize cost_vec to 0
    for u, v in edges:
        # For all i, if bit u != bit v, cost_vec[i] += 1
        # We can create a boolean mask of length 2^n
        # This is 1024 booleans. 45 edges. 45 * 1024 ops. Very fast.
        
        # Generate indices where bit u != bit v
        # This is a bit tricky to vectorize without creating large arrays repeatedly.
        # Given small n, a simple loop over states might be faster in Python than complex vectorization overhead?
        # No, numpy is C.
        
        # Let's try:
        # indices = np.arange(num_states)
        # bu = (indices >> u) & 1
        # bv = (indices >> v) & 1
        # mask = bu != bv
        # cost_vec[mask] += 1
        
        indices = np.arange(num_states)
        bu = (indices >> u) & 1
        bv = (indices >> v) & 1
        mask = bu != bv
        cost_vec[mask] += 1

    # 2. Define the objective function
    def objective(params):
        # Check time
        if time.time() > deadline:
            return 1e10

        g1, b1, g2, b2 = params
        # Ensure parameters are in [0, pi] for optimization stability
        # Nelder-Mead doesn't use bounds well, so we wrap or just hope.
        # Or we can use 'L-BFGS-B' with bounds.
        # Let's stick to Nelder-Mead and rely on the landscape.
        
        # State initialization: |+>^n
        # |0> = [1, 0], |1> = [0, 1]
        # |+> = [1/sqrt(2), 1/sqrt(2)]
        # State is tensor product of |+>
        state = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
        
        # Helper to apply R_x(beta) to all qubits
        # R_x(beta) = [[cos(beta/2), -i sin(beta/2)], [-i sin(beta/2), cos(beta/2)]]
        
        def apply_rx_all(state, beta):
            # Reshape to tensor
            shape = [2] * n
            tensor = state.reshape(shape)
            
            c = np.cos(beta / 2.0)
            s = np.sin(beta / 2.0)
            # R_x = [[c, -1j*s], [-1j*s, c]]
            
            # Apply to each axis
            # We need to be careful with axis ordering in numpy.
            # Axis 0 corresponds to most significant bit?
            # In our bitstring i, bit 0 is LSB.
            # In tensor reshape, axis 0 is the first dimension.
            # If we use np.reshape, the last index varies fastest.
            # So axis n-1 is LSB (bit 0), axis 0 is MSB (bit n-1).
            # The operation is symmetric for all qubits, so order doesn't matter.
            
            for q in range(n):
                # Move axis q to front
                tensor = np.moveaxis(tensor, q, 0)
                # Reshape to (2, -1)
                orig_shape = tensor.shape
                tensor = tensor.reshape(2, -1)
                # Apply matrix
                # [[c, -1j*s], [-1j*s, c]] @ [v0, v1]
                # new_v0 = c * v0 - 1j * s * v1
                # new_v1 = -1j * s * v0 + c * v1
                v0 = tensor[0, :]
                v1 = tensor[1, :]
                new_v0 = c * v0 - 1j * s * v1
                new_v1 = -1j * s * v0 + c * v1
                tensor = np.vstack((new_v0, new_v1))
                # Reshape back
                tensor = tensor.reshape(orig_shape)
                # Move axis back
                tensor = np.moveaxis(tensor, 0, q)
                
            return tensor.reshape(num_states)

        def apply_cost(state, gamma):
            # Multiply by exp(-i * gamma * cost_vec)
            phase = np.exp(-1j * gamma * cost_vec)
            return state * phase

        # Apply layers
        # Layer 1
        state = apply_cost(state, g1)
        state = apply_rx_all(state, b1)
        # Layer 2
        state = apply_cost(state, g2)
        state = apply_rx_all(state, b2)
        
        # Calculate expected cut
        probs = np.abs(state) ** 2
        exp_cut = np.dot(probs, cost_vec)
        
        return -exp_cut

    # 3. Optimization
    # We will try a few random restarts
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Initial guesses
    # Random points in [0, pi]
    np.random.seed(42) # Fixed seed for reproducibility in this block?
    # Better to use different seeds for restarts
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            x0 = np.array([0.35, 0.25, 0.35, 0.25])
        else:
            x0 = np.random.uniform(0, np.pi, 4)
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # Return in required shape (p, 2)
    # p=2, so we have 2 gammas and 2 betas
    # best_params is [g1, b1, g2, b2]
    return best_params.reshape(p, 2)
