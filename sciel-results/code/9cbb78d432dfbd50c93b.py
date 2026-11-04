import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p

    # Construct target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    def compute_error(U, V, W):
        R = U.shape[1]
        X = np.zeros((nm, mp, pn))
        for r in range(R):
            X += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.sum((X - T)**2)

    def compute_gradients(U, V, W):
        R = U.shape[1]
        X = np.zeros((nm, mp, pn))
        for r in range(R):
            X += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        E = X - T
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        for r in range(R):
            grad_U[:, r] = 2.0 * np.einsum('abc,b,c->a', E, V[:, r], W[:, r])
            grad_V[:, r] = 2.0 * np.einsum('abc,a,c->b', E, U[:, r], W[:, r])
            grad_W[:, r] = 2.0 * np.einsum('abc,a,b->c', E, U[:, r], V[:, r])
        return grad_U, grad_V, grad_W

    best_rank = schoolbook_rank
    best_U = np.zeros((nm, best_rank))
    best_V = np.zeros((mp, best_rank))
    best_W = np.zeros((pn, best_rank))
    
    # Schoolbook init
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                best_U[alpha, col] = 1.0
                best_V[beta, col] = 1.0
                best_W[gamma, col] = 1.0
                col += 1

    start_time = time.time()
    deadline = start_time + 130

    # Search for lower ranks
    # We try ranks from 110 down to 50 in steps of 5
    # If we find one, we try to go lower
    for target_rank in range(110, 49, -5):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue

        found = False
        # Try 3 restarts per rank
        for restart in range(3):
            if time.time() > deadline:
                break
            np.random.seed(123 + restart * 7 + target_rank)
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5

            lr = 0.01
            momentum = 0.9
            v_U = np.zeros_like(U)
            v_V = np.zeros_like(V)
            v_W = np.zeros_like(W)
            
            best_loss = compute_error(U, V, W)
            patience = 0
            max_patience = 200
            max_steps = 3000

            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                loss = compute_error(U, V, W)
                
                if loss < 1e-9:
                    X = np.zeros((nm, mp, pn))
                    for r in range(target_rank):
                        X += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                    if np.allclose(X, T, atol=1e-8):
                        best_rank = target_rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        found = True
                        break
                        
                if loss < best_loss * 0.99:
                    best_loss = loss
                    patience = 0
                else:
                    patience += 1
                    
                if patience > max_patience:
                    break
                    
                grad_U, grad_V, grad_W = compute_gradients(U, V, W)
                
                v_U = momentum * v_U + (1 - momentum) * grad_U
                v_V = momentum * v_V + (1 - momentum) * grad_V
                v_W = momentum * v_W + (1 - momentum) * grad_W
                
                U -= lr * v_U
                V -= lr * v_V
                W -= lr * v_W

            if found:
                break

    def to_list(mat):
        return [[round(float(x), 12) for x in row] for row in mat]

    U_out = to_list(best_U)
    V_out = to_list(best_V)
    W_out = to_list(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}
