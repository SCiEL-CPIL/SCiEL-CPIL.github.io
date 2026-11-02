import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_total = time.time()
    deadline = start_total + 140  # safety margin before 150s kill
    
    # Construct the target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose T"""
        E = T.copy()
        for r in range(U.shape[1]):
            E -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.linalg.norm(E) < 1e-9
    
    def try_find_decomposition(rank, time_limit=15, seed=42):
        start_time = time.time()
        np.random.seed(seed)
        
        # Initialize U, V, W with random values
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Scale to help convergence
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        # Adam optimizer parameters
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        lr = 0.05
        
        # Adam state
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        t = 0
        
        for step in range(5000):
            if time.time() - start_time > time_limit:
                break
            if time.time() > deadline:
                break
            
            t += 1
            E = T.copy()
            for r in range(rank):
                E -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            # Compute gradients
            Temp1 = np.einsum('abc,br->acr', E, V)
            G_U = np.einsum('acr,cr->ar', Temp1, W)
            
            Temp2 = np.einsum('abc,ar->bcr', E, U)
            G_V = np.einsum('bcr,cr->br', Temp2, W)
            
            G_W = np.einsum('acr,ar->cr', Temp1, U)
            
            # Adam update
            m_U = beta1 * m_U + (1 - beta1) * G_U
            v_U = beta2 * v_U + (1 - beta2) * G_U ** 2
            m_U_hat = m_U / (1 - beta1 ** t)
            v_U_hat = v_U / (1 - beta2 ** t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + epsilon)
            
            m_V = beta1 * m_V + (1 - beta1) * G_V
            v_V = beta2 * v_V + (1 - beta2) * G_V ** 2
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + epsilon)
            
            m_W = beta1 * m_W + (1 - beta1) * G_W
            v_W = beta2 * v_W + (1 - beta2) * G_W ** 2
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + epsilon)
            
            # Check convergence
            err = np.linalg.norm(E)
            if err < 1e-10:
                break
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check exactness
        if verify(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        else:
            return None
    
    # Try to find a lower rank decomposition
    result = None
    
    # Try rank 23 first (theoretical minimum for 3x3 is 19, but 23 is known)
    if time.time() < deadline - 30:
        result = try_find_decomposition(23, time_limit=15, seed=42)
    
    # Try rank 24
    if not result and time.time() < deadline - 30:
        result = try_find_decomposition(24, time_limit=15, seed=123)
    
    # Try rank 25
    if not result and time.time() < deadline - 30:
        result = try_find_decomposition(25, time_limit=15, seed=456)
    
    # Try rank 26
    if not result and time.time() < deadline - 20:
        result = try_find_decomposition(26, time_limit=15, seed=789)
    
    # Try rank 27 (schoolbook)
    if not result and time.time() < deadline - 10:
        result = try_find_decomposition(27, time_limit=10, seed=101)
    
    if result:
        U, V, W = result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    # Convert to lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
