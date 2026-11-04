import time
import numpy as np
from fractions import Fraction

def _construct_schoolbook():
    """
    Construct the standard schoolbook decomposition for 2x3 * 3x10.
    Rank = 60.
    """
    n, m, p = 2, 3, 10
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                
                U[idx_a, t] = 1
                V[idx_b, t] = 1
                W[idx_c, t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """
    Verify if U, V, W (lists of lists of Fractions) form a valid decomposition.
    """
    n, m, p = 2, 3, 10
    U_frac = np.array(U, dtype=object)
    V_frac = np.array(V, dtype=object)
    W_frac = np.array(W, dtype=object)
    
    # Reconstruct tensor T_recon
    # T_recon[a, b, c] = sum_t U[a,t] * V[b,t] * W[c,t]
    # We only need to check non-zero entries of the true tensor.
    # True tensor T_true[a, b, c] is 1 if a=(i,j), b=(j,k), c=(k,i)
    
    # Iterate over all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                sum_val = Fraction(0)
                for t in range(U_frac.shape[1]):
                    sum_val += U_frac[a, t] * V_frac[b, t] * W_frac[c, t]
                
                if sum_val != 1:
                    return False
                
    # Check a zero entry to ensure no spurious terms
    # a=(0,0), b=(1,0), c=(0,0) -> T[0, 10, 0] should be 0
    a = 0
    b = 1 * 10 + 0
    c = 0 * 2 + 0
    sum_val = Fraction(0)
    for t in range(U_frac.shape[1]):
        sum_val += U_frac[a, t] * V_frac[b, t] * W_frac[c, t]
    if sum_val != 0:
        return False
        
    return True

def _try_gradient_descent(rank, time_limit=10.0, seed=42):
    """
    Attempt to find a rank-decomposition using gradient descent.
    Returns dict with U, V, W as lists of Fractions if successful, else None.
    """
    n, m, p = 2, 3, 10
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Precompute T_true
    T_true = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def compute_residual_norm(U, V, W):
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.linalg.norm(T_true - T_recon)

    def compute_grads(U, V, W):
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        Residual = T_true - T_recon
        
        Grad_U = -2 * np.einsum('abc,bt,ct->at', Residual, V, W)
        Grad_V = -2 * np.einsum('abc,at,ct->bt', Residual, U, W)
        Grad_W = -2 * np.einsum('abc,at,bt->ct', Residual, U, V)
        
        return Grad_U, Grad_V, Grad_W

    # Initialize
    np.random.seed(seed)
    U = np.random.randn(dim_u, rank) * 0.1
    V = np.random.randn(dim_v, rank) * 0.1
    W = np.random.randn(dim_w, rank) * 0.1
    
    # Adam-like optimizer
    lr = 0.005
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m_u = np.zeros_like(U)
    v_u = np.zeros_like(U)
    m_v = np.zeros_like(V)
    v_v = np.zeros_like(V)
    m_w = np.zeros_like(W)
    v_w = np.zeros_like(W)
    
    start_time = time.time()
    best_loss = float('inf')
    best_params = None
    t_step = 0
    
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        loss = compute_residual_norm(U, V, W)
        if loss < best_loss:
            best_loss = loss
            best_params = (U.copy(), V.copy(), W.copy())
            
        if best_loss < 1e-6:
            break
            
        g_u, g_v, g_w = compute_grads(U, V, W)
        
        t_step += 1
        
        # Update U
        m_u = beta1 * m_u + (1 - beta1) * g_u
        v_u = beta2 * v_u + (1 - beta2) * (g_u ** 2)
        m_u_hat = m_u / (1 - beta1 ** t_step)
        v_u_hat = v_u / (1 - beta2 ** t_step)
        U -= lr * m_u_hat / (np.sqrt(v_u_hat) + eps)
        
        # Update V
        m_v = beta1 * m_v + (1 - beta1) * g_v
        v_v = beta2 * v_v + (1 - beta2) * (g_v ** 2)
        m_v_hat = m_v / (1 - beta1 ** t_step)
        v_v_hat = v_v / (1 - beta2 ** t_step)
        V -= lr * m_v_hat / (np.sqrt(v_v_hat) + eps)
        
        # Update W
        m_w = beta1 * m_w + (1 - beta1) * g_w
        v_w = beta2 * v_w + (1 - beta2) * (g_w ** 2)
        m_w_hat = m_w / (1 - beta1 ** t_step)
        v_w_hat = v_w / (1 - beta2 ** t_step)
        W -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

    if best_loss < 1e-4:
        # Attempt to round to integers/half-integers
        # Scale to make largest entry ~1
        max_val = max(np.max(np.abs(best_params[0])), np.max(np.abs(best_params[1])), np.max(np.abs(best_params[2])))
        if max_val > 0:
            U_r, V_r, W_r = best_params
            # Try rounding to nearest integer
            U_int = np.round(U_r)
            V_int = np.round(V_r)
            W_int = np.round(W_r)
            
            # Convert to Fractions and verify
            U_frac = [[Fraction(int(x)) for x in row] for row in U_int]
            V_frac = [[Fraction(int(x)) for x in row] for row in V_int]
            W_frac = [[Fraction(int(x)) for x in row] for row in W_int]
            
            if _verify_decomposition(U_frac, V_frac, W_frac):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
                
            # Try rounding to half-integers
            U_half = np.round(U_r * 2) / 2
            V_half = np.round(V_r * 2) / 2
            W_half = np.round(W_r * 2) / 2
            
            U_frac = [[Fraction(int(x*2), 2) for x in row] for row in U_half]
            V_frac = [[Fraction(int(x*2), 2) for x in row] for row in V_half]
            W_frac = [[Fraction(int(x*2), 2) for x in row] for row in W_half]
            
            if _verify_decomposition(U_frac, V_frac, W_frac):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
                
    return None

def _try_kronecker_structure(deadline):
    """
    Exploit M_{2,3,10} = M_{2,1,10} \otimes M_{1,3,1}.
    M_{1,3,1} is rank 3.
    We need to find a low rank decomposition for M_{2,1,10}.
    If rank(M_{2,1,10}) = r, then rank(M_{2,3,10}) <= 3*r.
    Schoolbook for M_{2,1,10} is rank 20. So 3*20 = 60.
    If we can find rank 15 for M_{2,1,10}, we get rank 45 for M_{2,3,10}.
    """
    n1, m1, p1 = 2, 1, 10
    dim_u1 = n1 * m1 # 2
    dim_v1 = m1 * p1 # 10
    dim_w1 = p1 * n1 # 20
    
    T_true1 = np.zeros((dim_u1, dim_v1, dim_w1))
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                a = i * m1 + j
                b = j * p1 + k
                c = k * n1 + i
                T_true1[a, b, c] = 1.0

    def compute_residual_norm1(U, V, W):
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.linalg.norm(T_true1 - T_recon)

    def compute_grads1(U, V, W):
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        Residual = T_true1 - T_recon
        Grad_U = -2 * np.einsum('abc,bt,ct->at', Residual, V, W)
        Grad_V = -2 * np.einsum('abc,at,ct->bt', Residual, U, W)
        Grad_W = -2 * np.einsum('abc,at,bt->ct', Residual, U, V)
        return Grad_U, Grad_V, Grad_W

    best_rank_full = 60
    best_result = None
    
    # Try ranks for M_{2,1,10}
    # Theoretical lower bound for M_{2,1,10} is at least 2*10/2 = 10?
    # Let's try 12, 14, 16, 18, 20.
    for r1 in [12, 14, 16, 18, 20]:
        if time.time() > deadline - 15:
            break
            
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
                
            np.random.seed(42 + attempt * 100 + r1)
            U1 = np.random.randn(dim_u1, r1) * 0.1
            V1 = np.random.randn(dim_v1, r1) * 0.1
            W1 = np.random.randn(dim_w1, r1) * 0.1
            
            lr = 0.005
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            
            m_u = np.zeros_like(U1)
            v_u = np.zeros_like(U1)
            m_v = np.zeros_like(V1)
            v_v = np.zeros_like(V1)
            m_w = np.zeros_like(W1)
            v_w = np.zeros_like(W1)
            
            start_time = time.time()
            best_loss = float('inf')
            best_params = None
            t_step = 0
            
            for it in range(3000):
                if time.time() - start_time > 10.0:
                    break
                    
                loss = compute_residual_norm1(U1, V1, W1)
                if loss < best_loss:
                    best_loss = loss
                    best_params = (U1.copy(), V1.copy(), W1.copy())
                    
                if best_loss < 1e-6:
                    break
                    
                g_u, g_v, g_w = compute_grads1(U1, V1, W1)
                t_step += 1
                
                m_u = beta1 * m_u + (1 - beta1) * g_u
                v_u = beta2 * v_u + (1 - beta2) * (g_u ** 2)
                m_u_hat = m_u / (1 - beta1 ** t_step)
                v_u_hat = v_u / (1 - beta2 ** t_step)
                U1 -= lr * m_u_hat / (np.sqrt(v_u_hat) + eps)
                
                m_v = beta1 * m_v + (1 - beta1) * g_v
                v_v = beta2 * v_v + (1 - beta2) * (g_v ** 2)
                m_v_hat = m_v / (1 - beta1 ** t_step)
                v_v_hat = v_v / (1 - beta2 ** t_step)
                V1 -= lr * m_v_hat / (np.sqrt(v_v_hat) + eps)
                
                m_w = beta1 * m_w + (1 - beta1) * g_w
                v_w = beta2 * v_w + (1 - beta2) * (g_w ** 2)
                m_w_hat = m_w / (1 - beta1 ** t_step)
                v_w_hat = v_w / (1 - beta2 ** t_step)
                W1 -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

            if best_loss < 1e-4:
                # Found a good decomposition for M_{2,1,10}
                # Now construct M_{2,3,10} using Kronecker product
                # M_{2,3,10} = M_{2,1,10} \otimes M_{1,3,1}
                # M_{1,3,1} has rank 3.
                # Let's define M_{1,3,1} decomposition explicitly.
                # A (1x3): idx 0,1,2
                # B (3x1): idx 0,1,2
                # C (1x1): idx 0
                # T_{a,b,c} = 1 if a==b.
                # Rank 3 decomposition:
                # U2: (3, 3), V2: (3, 3), W2: (1, 3)
                # U2 = I, V2 = I, W2 = [1,1,1]
                
                U2 = np.eye(3)
                V2 = np.eye(3)
                W2 = np.ones((1, 3))
                
                # Kronecker product of decompositions:
                # If T1 = sum_{t1} U1_t1 \otimes V1_t1 \otimes W1_t1
                # and T2 = sum_{t2} U2_t2 \otimes V2_t2 \otimes W2_t2
                # Then T1 \otimes T2 = sum_{t1, t2} (U1_t1 \otimes U2_t2) \otimes (V1_t1 \otimes V2_t2) \otimes (W1_t1 \otimes W2_t2)
                
                U1_r, V1_r, W1_r = best_params
                r_total = r1 * 3
                
                U_full = np.zeros((6, r_total))
                V_full = np.zeros((30, r_total))
                W_full = np.zeros((20, r_total))
                
                # Mapping of indices for Kronecker product
                # A (2x3) index (i, j) -> i from A1 (2x1), j from A2 (1x3)
                # A1 index: i. A2 index: j.
                # A_full index: i * 3 + j.
                # U_full[i*3+j, t1*3+t2] = U1[i, t1] * U2[j, t2]
                
                # B (3x10) index (j, k) -> j from B1 (1x10)? No.
                # M_{2,1,10}: A is 2x1, B is 1x10, C is 2x10.
                # M_{1,3,1}: A is 1x3, B is 3x1, C is 1x1.
                # M_{2,3,10}: A is 2x3, B is 3x10, C is 2x10.
                # 
                # Tensor product:
                # A_full[i, j] = A1[i, 0] * A2[0, j]
                # B_full[j, k] = B1[0, k] * B2[j, 0]
                # C_full[i, k] = C1[i, k] * C2[0, 0]
                
                # So:
                # U1 corresponds to A1 (2x1). Index i.
                # U2 corresponds to A2 (1x3). Index j.
                # U_full index (i, j) -> i*3 + j.
                # U_full[i*3 + j, t1*3 + t2] = U1[i, t1] * U2[j, t2]
                
                # V1 corresponds to B1 (1x10). Index k.
                # V2 corresponds to B2 (3x1). Index j.
                # V_full index (j, k) -> j*10 + k.
                # V_full[j*10 + k, t1*3 + t2] = V1[k, t1] * V2[j, t2]
                
                # W1 corresponds to C1 (2x10). Index (i, k).
                # W2 corresponds to C2 (1x1). Index (0, 0).
                # W_full index (k, i) -> k*2 + i. (Trace convention W[k*n+i])
                # W_full[k*2 + i, t1*3 + t2] = W1[k*2 + i, t1] * W2[0, t2]
                
                for t1 in range(r1):
                    for t2 in range(3):
                        t_full = t1 * 3 + t2
                        
                        # U
                        for i in range(2):
                            for j in range(3):
                                idx_u = i * 3 + j
                                U_full[idx_u, t_full] = U1_r[i, t1] * U2[j, t2]
                                
                        # V
                        for j in range(3):
                            for k in range(10):
                                idx_v = j * 10 + k
                                V_full[idx_v, t_full] = V1_r[k, t1] * V2[j, t2]
                                
                        # W
                        for k in range(10):
                            for i in range(2):
                                idx_w = k * 2 + i
                                W_full[idx_w, t_full] = W1_r[k * 2 + i, t1] * W2[0, t2]
                                
                # Verify the full decomposition
                U_frac = [[Fraction(int(round(x * 1e6)), 1000000) for x in row] for row in U_full]
                V_frac = [[Fraction(int(round(x * 1e6)), 1000000) for x in row] for row in V_full]
                W_frac = [[Fraction(int(round(x * 1e6)), 1000000) for x in row] for row in W_full]
                
                # Simplify fractions
                U_frac = [[x.limit() for x in row] for row in U_frac]
                V_frac = [[x.limit() for x in row] for row in V_frac]
                W_frac = [[x.limit() for x in row] for row in W_frac]
                
                if _verify_decomposition(U_frac, V_frac, W_frac):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
                    
    return best_result

def construct():
    deadline = time.time() + 140
    
    # Fallback
    schoolbook = _construct_schoolbook()
    best_result = {
        "U": schoolbook["U"].tolist(),
        "V": schoolbook["V"].tolist(),
        "W": schoolbook["W"].tolist()
    }
    
    # Try Kronecker structure first, as it is a structural insight
    # that might find a lower rank more reliably than random GD on the full space.
    result = _try_kronecker_structure(deadline)
    if result is not None:
        return result
        
    # If Kronecker fails or doesn't improve, try standard GD on full space
    # We only have limited time left.
    if time.time() < deadline - 20:
        for rank in [30, 35, 40, 45, 50, 55]:
            if time.time() > deadline - 10:
                break
            result = _try_gradient_descent(rank, time_limit=10.0, seed=42)
            if result is not None:
                return result
                
    return best_result

if __name__ == "__main__":
    construct()
