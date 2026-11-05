import numpy as np
import time

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio of the 
    independently simulated expected cut to the exact classical optimum.
    
    Strategy:
    1. Precompute the cut values and the exact classical optimum.
    2. Decompose the QAOA expectation value into a linear combination of basis terms.
       The unitary U = U_X(beta_1) U_C(gamma_1) U_X(beta_0) U_C(gamma_0) ...
       can be expanded. For p=2, the expectation value <H_C> is a function of 
       gamma_0, beta_0, gamma_1, beta_1.
       
       Instead of simulating the state vector every time, we can precompute the 
       "basis" contributions.
       
       Let's derive the structure for p=2.
       |psi> = U_X(b1) U_C(g1) U_X(b0) U_C(g0) |+>^n
       
       This is complex to expand analytically for arbitrary n without symbolic algebra 
       which might be slow. 
       
       Alternative "Materially Different" Strategy:
       Use the fact that the problem is small (n<=10, p=2). 
       We can use a **Global Optimization** approach with a much faster cost function 
       evaluation by precomputing the action of the operators.
       
       Actually, the most robust "different" strategy for small n is to use 
       **Differential Evolution** or **Basin Hopping** with a very fast simulator.
       
       However, the prompt asks for a DIFFERENT design family. 
       Current: Local Gradient Descent (L-BFGS-B) + Random Restarts.
       
       Let's try: **Global Grid Search with Multi-Resolution Refinement**.
       The parameter space is 4D (gamma0, beta0, gamma1, beta1).
       The period of gamma is 2*pi (actually pi for cut values if symmetric, but 2*pi is safe).
       The period of beta is 2*pi (actually pi for X rotations? Rx(beta) = -Rx(beta+pi)).
       
       If we do a coarse grid of size 10x10x10x10 = 10,000 points.
       Then refine the top 100 points with a finer grid.
       
       To make this fast, we need the cost function to be extremely fast.
       
       Let's stick to the state vector simulation but optimize it heavily using 
       precomputed Kronecker products and matrix-vector multiplication, and use 
       a **Global Search** (Grid/DE) instead of Local Gradient Descent.
       
       Key insight: For n<=10, 2^n = 1024.
       Matrix-vector product is ~1000 ops.
       We can afford ~100,000 evaluations in 10 seconds.
       
       Strategy:
       1. Fast Simulator: Precompute U_X for a set of betas? No, beta varies continuously.
          But we can precompute the structure.
          Actually, constructing U_X(beta) via Kronecker product every time is O(4^n) = 1M ops for n=10.
          That's too slow for 100k evals (100k * 1M = 10^11 ops).
          
          We need a faster simulator.
          The tensor product structure allows us to apply Rx(beta) to each qubit sequentially.
          Applying Rx(beta) to qubit i of a state vector of size 2^n:
          This can be done in O(2^n) time.
          Total for p=2 layers: 2 layers * 2 X-rotations * n qubits * O(2^n) = 4 * n * 2^n.
          For n=10: 4 * 10 * 1024 = 40,960 ops per evaluation.
          100,000 evals * 40,960 = 4 * 10^9 ops. This is feasible in ~1-2 seconds in NumPy if vectorized properly.
          
          Let's implement a fast state-vector simulator that applies single-qubit gates sequentially.
          
          Then, use **Differential Evolution** (global optimizer) or **Multi-Resolution Grid Search**.
          Differential Evolution is a stochastic global optimizer, materially different from L-BFGS-B.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Precompute cut values
    max_states = 1 << n
    cut_vals = np.zeros(max_states, dtype=np.float64)
    for s in range(max_states):
        c = 0
        for u, v in edges:
            if ((s >> u) ^ (s >> v)) & 1:
                c += 1
        cut_vals[s] = c
        
    opt_cut = np.max(cut_vals)
    if opt_cut == 0:
        return np.zeros((p, 2))

    # Initial state |+>^n
    # We represent the state as a complex vector of size 2^n.
    state_init = np.ones(max_states, dtype=np.complex128)
    state_init /= np.sqrt(max_states)

    # Precompute bit masks for efficient X-rotation application?
    # Applying Rx(beta) to qubit i:
    # Rx = [[cos(b), -i sin(b)], [-i sin(b), cos(b)]]
    # For a state vector |psi>, the amplitude at index s where bit i is 0 is a_s.
    # The amplitude at index s where bit i is 1 is a_{s^mask_i}.
    # New a_s = cos(b) * a_s - i sin(b) * a_{s^mask_i} (if bit i is 0)
    # New a_{s^mask_i} = -i sin(b) * a_s + cos(b) * a_{s^mask_i} (if bit i is 1)
    
    # We can vectorize this over all qubits? No, they are entangled.
    # But we can apply them one by one.
    
    def apply_rx(state, beta, qubit):
        """Apply Rx(beta) to a specific qubit in the state vector."""
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        
        mask = 1 << qubit
        # Create indices where the qubit is 0 and 1
        # We need to pair up indices.
        # Indices where bit qubit is 0:
        # It's easier to reshape the state into a tensor of shape (2, 2, ..., 2)
        # But reshaping is expensive.
        
        # Efficient vectorized approach:
        # We can use advanced indexing.
        # Let's precompute the pairs of indices for each qubit.
        pass

    # Precompute index pairs for each qubit to avoid overhead in the loop
    index_pairs = []
    for q in range(n):
        mask = 1 << q
        # We want to iterate over all states where bit q is 0.
        # There are 2^(n-1) such states.
        # Let's generate them once.
        # This might use memory: n * 2^(n-1) integers. For n=10, 10*512=5120 ints. Tiny.
        indices_0 = []
        indices_1 = []
        for s in range(max_states):
            if (s & mask) == 0:
                indices_0.append(s)
                indices_1.append(s | mask)
        index_pairs.append((np.array(indices_0, dtype=np.int32), np.array(indices_1, dtype=np.int32)))

    def simulate_fast(params):
        """
        params: [gamma_0, beta_0, ..., gamma_{p-1}, beta_{p-1}]
        """
        state = state_init.copy()
        gammas = params[:p]
        betas = params[p:]
        
        for k in range(p):
            g = gammas[k]
            b = betas[k]
            
            # Apply C phase: element-wise multiplication
            phases_c = np.exp(-1j * g * cut_vals)
            state = state * phases_c
            
            # Apply X rotation: U_X(b) = \bigotimes Rx(b)
            # Apply to each qubit sequentially
            for q in range(n):
                idx0, idx1 = index_pairs[q]
                c = np.cos(b)
                s = -1j * np.sin(b)
                
                a0 = state[idx0]
                a1 = state[idx1]
                
                new_a0 = c * a0 + s * a1
                new_a1 = s * a0 + c * a1
                
                state[idx0] = new_a0
                state[idx1] = new_a1
                
        probs = np.abs(state) ** 2
        expected_cut = np.sum(probs * cut_vals)
        return expected_cut / opt_cut

    def objective(params):
        return -simulate_fast(params)

    # Global Optimization Strategy: Differential Evolution
    # DE is a population-based global optimizer.
    # It is robust to local minima.
    
    from scipy.optimize import differential_evolution
    
    # Bounds: gamma in [0, pi], beta in [0, pi]
    # Note: Rx(beta) has period 2*pi, but Rx(beta+pi) = -Rx(beta).
    # The global phase doesn't matter, but relative phases do.
    # Actually, for the cut value, symmetries might reduce the space.
    # Let's use [0, pi] for both.
    
    bounds = [(0, np.pi)] * (2 * p)
    
    # We want to use as many iterations as possible within the time limit.
    # DE with population size N.
    # Cost per generation: N * (1 + 1) evaluations (approx, depends on strategy).
    # Let's benchmark.
    
    t0 = time.time()
    _ = simulate_fast(np.array([0.5]*4))
    sim_time = time.time() - t0
    
    # Estimate budget
    time_left = deadline - time.time()
    
    # Differential Evolution parameters
    # popsize: 15 is default. Let's try 20-50.
    # maxiter: We can run many iterations.
    
    # To be safe, let's do a coarse grid search first to find a good region,
    # then run DE locally? No, DE is global.
    
    # Let's just run DE with a generous maxiter and monitor time.
    # However, scipy's DE doesn't have a callback to stop early easily in older versions,
    # but we can check time inside the objective? No, that's hacky.
    
    # Better: Run DE with a fixed, moderate number of iterations,
    # and if time remains, run it again with tighter bounds around the best solution?
    # Or just run DE with maxiter set high and hope it converges or we can catch it.
    
    # Actually, let's implement a simple **Multi-Resolution Grid Search** which is 
    # deterministic and easy to time-box.
    
    # Grid Search Strategy:
    # 1. Coarse grid: 5 points per dimension (5^4 = 625 evals).
    # 2. Identify top 10 best points.
    # 3. Fine grid around each top point: 5 points per dimension in a smaller range.
    # 4. Identify top 10.
    # 5. Even finer grid.
    
    def grid_search(bounds, n_points, current_best=None):
        """
        Perform a grid search within the given bounds.
        Returns best_params, best_value.
        """
        # Create grid points
        # Use linspace
        grid_points = np.array(np.meshgrid(*[np.linspace(b[0], b[1], n_points) for b in bounds]))
        grid_points = grid_points.reshape(n_points, -1).T
        
        best_val = -np.inf
        best_p = None
        
        for i in range(len(grid_points)):
            if time.time() > deadline:
                break
            p = grid_points[i]
            val = simulate_fast(p)
            if val > best_val:
                best_val = val
                best_p = p
                
        return best_p, best_val

    # Step 1: Coarse Grid
    coarse_bounds = [(0, np.pi)] * (2 * p)
    best_params, best_val = grid_search(coarse_bounds, 6)
    
    # Step 2: Refine
    # Define a smaller range around the best found
    range_size = 0.5 * np.pi
    refine_bounds = [
        (max(0, best_params[i] - range_size), min(np.pi, best_params[i] + range_size)) 
        for i in range(2 * p)
    ]
    
    best_params, best_val = grid_search(refine_bounds, 8)
    
    # Step 3: Refine again
    range_size = 0.1 * np.pi
    refine_bounds = [
        (max(0, best_params[i] - range_size), min(np.pi, best_params[i] + range_size)) 
        for i in range(2 * p)
    ]
    
    best_params, best_val = grid_search(refine_bounds, 10)
    
    # Step 4: Final Polish with a few local steps? 
    # Or just return the grid best.
    
    gammas = best_params[:p]
    betas = best_params[p:]
    
    return np.array([gammas, betas]).T

