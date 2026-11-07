import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 320.0  # Leave 10s buffer before 330s kill

    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=float)
    n, d = X.shape

    # Normalize inputs
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    X_norm = (X - x_mean) / x_std

    # Normalize targets
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    def _time_left():
        return deadline - time.time()

    # Strategy: Use a small neural network (2 hidden layers) trained with Adam.
    # This leverages universal approximation to fit arbitrary smooth functions.
    # We will train for a fixed number of epochs or until time runs out.
    # Since data is noiseless and n=200, a small net should converge quickly.

    # Implement a simple neural network in NumPy to avoid external dependencies like PyTorch/TensorFlow
    # which might not be available or might be slow to load.
    
    class SimpleNN:
        def __init__(self, input_dim, hidden1, hidden2, lr=0.01):
            # He initialization
            self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
            self.b1 = np.zeros((1, hidden1))
            self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
            self.b2 = np.zeros((1, hidden2))
            self.W3 = np.random.randn(hidden2, 1) * np.sqrt(2.0 / hidden2)
            self.b3 = np.zeros((1, 1))
            self.lr = lr
            
            # Adam optimizer state
            self.m_W1 = np.zeros_like(self.W1)
            self.v_W1 = np.zeros_like(self.W1)
            self.m_b1 = np.zeros_like(self.b1)
            self.v_b1 = np.zeros_like(self.b1)
            self.m_W2 = np.zeros_like(self.W2)
            self.v_W2 = np.zeros_like(self.W2)
            self.m_b2 = np.zeros_like(self.b2)
            self.v_b2 = np.zeros_like(self.b2)
            self.m_W3 = np.zeros_like(self.W3)
            self.v_W3 = np.zeros_like(self.W3)
            self.m_b3 = np.zeros_like(self.b3)
            self.v_b3 = np.zeros_like(self.b3)
            self.t = 0
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8

        def _relu(self, z):
            return np.maximum(0, z)

        def _relu_derivative(self, z):
            return (z > 0).astype(float)

        def forward(self, X):
            self.z1 = X @ self.W1 + self.b1
            self.a1 = self._relu(self.z1)
            self.z2 = self.a1 @ self.W2 + self.b2
            self.a2 = self._relu(self.z2)
            self.z3 = self.a2 @ self.W3 + self.b3
            return self.z3

        def backward(self, X, y_true):
            # y_true shape (n, 1)
            m = X.shape[0]
            
            # Output layer error
            self.dz3 = self.z3 - y_true
            self.dW3 = (self.a2.T @ self.dz3) / m
            self.db3 = np.sum(self.dz3, axis=0, keepdims=True) / m
            
            # Hidden layer 2 error
            self.dz2 = (self.dz3 @ self.W3.T) * self._relu_derivative(self.z2)
            self.dW2 = (self.a1.T @ self.dz2) / m
            self.db2 = np.sum(self.dz2, axis=0, keepdims=True) / m
            
            # Hidden layer 1 error
            self.dz1 = (self.dz2 @ self.W2.T) * self._relu_derivative(self.z1)
            self.dW1 = (X.T @ self.dz1) / m
            self.db1 = np.sum(self.dz1, axis=0, keepdims=True) / m

            self._update()

        def _update(self):
            self.t += 1
            self._update_param('W1')
            self._update_param('b1')
            self._update_param('W2')
            self._update_param('b2')
            self._update_param('W3')
            self._update_param('b3')

        def _update_param(self, name):
            W = getattr(self, name)
            dW = getattr(self, 'd' + name)
            m = getattr(self, 'm_' + name)
            v = getattr(self, 'v_' + name)
            
            m = self.beta1 * m + (1 - self.beta1) * dW
            v = self.beta2 * v + (1 - self.beta2) * (dW ** 2)
            
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            W -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            setattr(self, name, W)
            setattr(self, 'm_' + name, m)
            setattr(self, 'v_' + name, v)

    # Try multiple architectures and learning rates
    # Since we have 320s and 10 equations, we have ~32s per equation.
    # We can afford to train for a few thousand steps.
    
    best_model = None
    best_loss = np.inf
    
    # Define some configurations to try
    configs = [
        (16, 16, 0.01),
        (32, 32, 0.01),
        (64, 64, 0.01),
        (32, 32, 0.001),
        (64, 64, 0.001),
    ]
    
    # We will pick the best one based on training loss (since data is noiseless, low training loss implies good generalization)
    # Or we could use a small validation split, but with only 200 points, that's risky.
    # Given the task is symbolic regression, the function is likely simple, so overfitting a small net is less of an issue if it converges.
    
    for h1, h2, lr in configs:
        if _time_left() < 2:
            break
            
        np.random.seed(42) # Reproducibility for each config
        model = SimpleNN(d, h1, h2, lr=lr)
        
        # Training loop
        max_steps = 5000
        for step in range(max_steps):
            if _time_left() < 5:
                break
                
            y_pred = model.forward(X_norm)
            loss = np.mean((y_pred - y_norm) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                # Store a copy of the weights
                best_model = {
                    'W1': model.W1.copy(),
                    'b1': model.b1.copy(),
                    'W2': model.W2.copy(),
                    'b2': model.b2.copy(),
                    'W3': model.W3.copy(),
                    'b3': model.b3.copy(),
                    'h1': h1,
                    'h2': h2
                }
            
            model.backward(X_norm, y_norm)
            
            # Early stopping if loss is very small
            if loss < 1e-8:
                break

    # If NN didn't work well, fallback to simple linear regression on polynomial features?
    # Or just use the best NN found.
    
    if best_model is None:
        # Fallback: return mean
        def predict(X_test):
            X_test = np.array(X_test, dtype=float)
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            return np.full(X_test.shape[0], y_mean)
        return predict

    W1 = best_model['W1']
    b1 = best_model['b1']
    W2 = best_model['W2']
    b2 = best_model['b2']
    W3 = best_model['W3']
    b3 = best_model['b3']

    def predict(X_test):
        X_test = np.array(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        # Normalize test data
        X_test_norm = (X_test - x_mean) / x_std
        
        # Forward pass
        z1 = X_test_norm @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        a2 = np.maximum(0, z2)
        z3 = a2 @ W3 + b3
        
        # Denormalize
        y_pred = z3.flatten() * y_std + y_mean
        
        return y_pred

    return predict
