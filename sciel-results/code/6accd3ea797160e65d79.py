import numpy as np
import math
import time

def _simple_nn_forward(X, W1, b1, W2, b2):
    """Forward pass for a 2-layer neural network with tanh activation."""
    h = np.tanh(X @ W1 + b1)
    return h @ W2 + b2

def _simple_nn_backward(X, y_pred, y_true, W1, b1, W2, b2, lr=0.001):
    """Backward pass for a 2-layer neural network. Returns updated weights."""
    m = X.shape[0]
    
    # Output layer gradient
    d_output = (y_pred - y_true) / m  # MSE derivative
    
    # Hidden layer
    dW2 = h.T @ d_output if 'h' in locals() else None
    
    # Recompute hidden
    h = np.tanh(X @ W1 + b1)
    dW2 = h.T @ d_output
    db2 = np.sum(d_output, axis=0)
    
    # Backprop to hidden
    d_hidden = d_output @ W2.T * (1 - h**2)
    dW1 = X.T @ d_hidden
    db1 = np.sum(d_hidden, axis=0)
    
    # Update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
    
    return W1, b1, W2, b2

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0  # 110s safety margin before 120s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    n_paths = len(training_paths)
    n_times = len(training_paths[0])
    
    # Extract spot prices at each exercise date
    # training_paths[i][j] should be the spot at time step j for path i
    # We need to identify which indices correspond to exercise dates
    # Assuming n_times includes time 0, exercise dates are at indices 1, 2, ..., 12
    # Or they might be evenly spaced. Let's assume the paths are sampled at exercise dates.
    
    # Check if paths have exactly 12 or 13 points (including time 0)
    # For now, assume the last 12 points correspond to exercise dates 1-12
    
    # Prepare data: X = [S/K, t/12], y = continuation value or exercise decision
    # We'll use a hybrid approach: compute continuation values from a coarse tree,
    # then refine with neural network regression
    
    # First, compute a baseline using a moderate-resolution binomial tree
    n_steps = 600  # Reduced from 1200 for speed, still accurate
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)
    
    # Exercise steps in the tree
    exercise_steps = {i * n_steps // n_exercises: i for i in range(1, n_exercises + 1)}
    
    # Grid
    S_max = 1.5 * K
    n_grid = 50000
    S_grid = np.linspace(0.0, S_max, n_grid)
    
    # V at maturity
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Compute continuation values at exercise dates using the tree
    # Store continuation values at each exercise date as a function of S
    cont_values = {}  # {exercise_idx: array of continuation values on S_grid}
    
    for step in range(n_steps - 1, 0, -1):
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        Su = S_grid * u
        Sd = S_grid * d
        
        V_up = np.interp(Su, S_grid, V_next)
        V_down = np.interp(Sd, S_grid, V_next)
        
        continuation = df * (p * V_up + (1 - p) * V_down)
        
        V_current = np.maximum(intrinsic, continuation)
        
        if step in exercise_steps:
            idx = exercise_steps[step]
            cont_values[idx] = continuation.copy()
        
        V_next = V_current
        
        # Time check
        if step % 200 == 0 and time.time() > deadline - 30:
            break

    # Now, use neural network to learn a more flexible boundary
    # For each exercise date, train a small NN to predict whether to exercise
    # Input: [S/K], Output: 1 if exercise, 0 if hold
    
    # Use training paths to create supervised labels
    # For each path and each exercise date, we can compute the optimal decision
    # using the continuation values from the tree
    
    boundaries = np.zeros(n_exercises + 1)
    
    for ex_idx in range(1, n_exercises + 1):
        if ex_idx not in cont_values:
            # Fallback to binary search on the tree's continuation
            if ex_idx <= n_steps // n_exercises * ex_idx and ex_idx <= len(S_grid):
                pass
            boundaries[ex_idx] = K * 0.8  # Default fallback
            continue
        
        cont = cont_values[ex_idx]
        intrinsic = np.maximum(K - S_grid, 0.0)
        diff = intrinsic - cont
        
        # Find the boundary from the tree
        exercise_mask = diff >= 0
        if np.any(exercise_mask):
            idxs = np.where(exercise_mask)[0]
            last_idx = idxs[-1]
            if last_idx < n_grid - 1:
                diff0 = diff[last_idx]
                diff1 = diff[last_idx + 1]
                if diff1 != diff0:
                    t = diff0 / (diff0 - diff1)
                    boundaries[ex_idx] = S_grid[last_idx] + t * (S_grid[last_idx + 1] - S_grid[last_idx])
                else:
                    boundaries[ex_idx] = S_grid[last_idx]
            else:
                boundaries[ex_idx] = S_grid[-1]
        else:
            boundaries[ex_idx] = 0.0
        
        # Now refine with neural network using training paths
        # Create training data: for each path at this exercise date, 
        # determine if it should exercise based on the tree's boundary
        # Then train a NN to predict this decision from S
        
        # Get spot prices at this exercise date from training paths
        # Assume training_paths[i] is an array of spot prices over time
        # Exercise date ex_idx corresponds to some index in the path
        
        # Determine the index in training_paths for exercise date ex_idx
        # If paths have 13 points (time 0 to 12), then ex_idx maps to index ex_idx
        # If paths have 12 points (time 1 to 12), then ex_idx maps to index ex_idx-1
        
        if len(training_paths[0]) >= n_exercises + 1:
            path_idx = ex_idx
        else:
            path_idx = ex_idx - 1
        
        if path_idx < len(training_paths[0]):
            S_train = np.array([training_paths[i][path_idx] for i in range(n_paths)])
            
            # Create labels: 1 if S <= boundary, 0 otherwise
            y_train = (S_train <= boundaries[ex_idx]).astype(float)
            
            # Normalize S
            S_norm = S_train / K
            
            # Train a small neural network
            # Input: [S_norm], Output: probability of exercise
            np.random.seed(42)
            hidden_size = 32
            W1 = np.random.randn(1, hidden_size) * 0.1
            b1 = np.zeros(hidden_size)
            W2 = np.random.randn(hidden_size, 1) * 0.1
            b2 = np.zeros(1)
            
            X = S_norm.reshape(-1, 1)
            y = y_train.reshape(-1, 1)
            
            # Train for a few epochs
            n_epochs = 50
            lr = 0.01
            batch_size = min(256, n_paths)
            
            for epoch in range(n_epochs):
                if time.time() > deadline - 10:
                    break
                
                # Mini-batch training
                perm = np.random.permutation(n_paths)
                for start in range(0, n_paths, batch_size):
                    end = min(start + batch_size, n_paths)
                    batch_idx = perm[start:end]
                    
                    X_batch = X[batch_idx]
                    y_batch = y[batch_idx]
                    
                    # Forward
                    h = np.tanh(X_batch @ W1 + b1)
                    y_pred = h @ W2 + b2
                    y_pred = 1 / (1 + np.exp(-y_pred))  # Sigmoid for probability
                    
                    # Backward
                    m = len(batch_idx)
                    d_output = (y_pred - y_batch) / m
                    
                    dW2 = h.T @ d_output
                    db2 = np.sum(d_output, axis=0)
                    
                    d_hidden = d_output @ W2.T * (1 - h**2)
                    dW1 = X_batch.T @ d_hidden
                    db1 = np.sum(d_hidden, axis=0)
                    
                    # Update
                    W2 -= lr * dW2
                    b2 -= lr * db2
                    W1 -= lr * dW1
                    b1 -= lr * db1
            
            # After training, find the new boundary
            # The NN predicts P(exercise | S). We want the S where P = 0.5
            # Evaluate the NN on a fine grid
            S_fine = np.linspace(0, 1.5 * K, 1000)
            S_fine_norm = (S_fine / K).reshape(-1, 1)
            h_fine = np.tanh(S_fine_norm @ W1 + b1)
            prob_fine = 1 / (1 + np.exp(-(h_fine @ W2 + b2)))
            prob_fine = prob_fine.flatten()
            
            # Find where prob crosses 0.5
            cross_idx = np.where(prob_fine >= 0.5)[0]
            if len(cross_idx) > 0:
                new_boundary = S_fine[cross_idx[-1]]
                # Only update if the new boundary is reasonable
                if 0 < new_boundary < 1.2 * K:
                    boundaries[ex_idx] = new_boundary
    
    boundaries[12] = K

    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
        
        def exercise(self, current_state, time):
            if isinstance(time, (int, float, np.integer, np.floating)):
                idx = int(time)
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                return np.array(current_state) <= b
            else:
                times = np.array(time)
                times_clipped = np.clip(times, 1, 12).astype(int)
                b = np.array([self.boundaries[t] for t in times_clipped])
                return np.array(current_state) <= b
    
    return ExercisePolicy(boundaries, K)
