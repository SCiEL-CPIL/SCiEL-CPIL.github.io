import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
    
    # Precompute target tensor
    # Shape: (n*m, m*p, p*n)
    T_tgt = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_tgt[a, b, c] = 1.0

    def validate(U, V, W, tol=1e-8):
        # U, V, W are numpy arrays
        T = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.all(np.abs(T - T_tgt) < tol)

    def try_rank(r, max_seeds=50, max_iters=1000, lr=0.01):
        for seed in range(max_seeds):
            if time.time() > deadline:
                return None
            np.random.seed(seed)
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Adam optimizer parameters
            mU = np.zeros_like(U)
            vU = np.zeros_like(U)
            mV = np.zeros_like(V)
            vV = np.zeros_like(V)
            mW = np.zeros_like(W)
            vW = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t = 0
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Compute T and Error
                T = np.einsum('ak,bk,ck->abc', U, V, W)
                E = T - T_tgt
                
                # Check convergence
                if np.max(np.abs(E)) < 1e-7:
                    # Polish
                    for _ in range(10):
                        T = np.einsum('ak,bk,ck->abc', U, V, W)
                        E = T - T_tgt
                        dU = np.einsum('abc,bk,ck->ak', E, V, W)
                        dV = np.einsum('abc,ak,ck->bk', E, U, W)
                        dW = np.einsum('abc,ak,bk->ck', E, U, V)
                        U -= 0.01 * dU
                        V -= 0.01 * dV
                        W -= 0.01 * dW
                    if validate(U, V, W, tol=1e-8):
                        return U, V, W
                    else:
                        break

                # Compute Gradients
                dU = np.einsum('abc,bk,ck->ak', E, V, W)
                dV = np.einsum('abc,ak,ck->bk', E, U, W)
                dW = np.einsum('abc,ak,bk->ck', E, U, V)
                
                # Adam Update
                t += 1
                mU = beta1 * mU + (1 - beta1) * dU
                vU = beta2 * vU + (1 - beta2) * (dU ** 2)
                mU_hat = mU / (1 - beta1 ** t)
                vU_hat = vU / (1 - beta2 ** t)
                U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
                
                mV = beta1 * mV + (1 - beta1) * dV
                vV = beta2 * vV + (1 - beta2) * (dV ** 2)
                mV_hat = mV / (1 - beta1 ** t)
                vV_hat = vV / (1 - beta2 ** t)
                V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
                
                mW = beta1 * mW + (1 - beta1) * dW
                vW = beta2 * vW + (1 - beta2) * (dW ** 2)
                mW_hat = mW / (1 - beta1 ** t)
                vW_hat = vW / (1 - beta2 ** t)
                W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
                # Occasionally check for convergence in loop
                if it % 10 == 0 and np.max(np.abs(E)) < 1e-5:
                     # Try a quick polish
                     for _ in range(5):
                         T = np.einsum('ak,bk,ck->abc', U, V, W)
                         E = T - T_tgt
                         dU = np.einsum('abc,bk,ck->ak', E, V, W)
                         dV = np.einsum('abc,ak,ck->bk', E, U, W)
                         dW = np.einsum('abc,ak,bk->ck', E, U, V)
                         U -= 0.01 * dU
                         V -= 0.01 * dV
                         W -= 0.01 * dW
                     if validate(U, V, W, tol=1e-7):
                         return U, V, W

        return None

    # Try to find lower rank decompositions
    # Start from 29 down to 25
    for r in range(29, 24, -1):
        if time.time() > deadline:
            break
        res = try_rank(r, max_seeds=20, max_iters=2000, lr=0.05)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
