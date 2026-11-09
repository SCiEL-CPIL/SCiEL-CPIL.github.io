import numpy as np
import random
import time

def _find_block_decomposition(n, m, p, target_rank, deadline):
    """
    Find a rank-target_rank decomposition for n x m x p.
    Uses gradient descent on the loss function.
    """
    # Initialize
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    # Scale to avoid huge gradients
    U /= np.linalg.norm(U) * np.sqrt(n*m)
    V /= np.linalg.norm(V) * np.sqrt(m*p)
    W /= np.linalg.norm(W) * np.sqrt(p*n)
    
    # Loss function
    def loss_and_grad():
        # Compute T = einsum('imc,jpc,knc->ijk', U, V, W)
        # For n=2, m=3, p=3, this is 2x3x3
        T = np.einsum('imc,jpc,knc->ijk', U, V, W)
        loss = np.sum((T - 1.0)**2)
        
        # Gradients
        # dL/dU = 2 * (T - 1) * V * W
        # T_ijk = sum_c U_imc V_jpc W_knc
        # dT_ijk/dU_imc = V_jpc W_knc
        # dL/dU_imc = sum_j,k 2(T_ijk - 1) V_jpc W_knc
        # This can be computed efficiently.
        
        # Reshape T to (n, m, p)
        # grad_U = 2 * einsum('ijk,jpc,knc->imc', T-1, V, W)
        grad_U = 2.0 * np.einsum('ijk,jpc,knc->imc', T - 1.0, V, W)
        grad_V = 2.0 * np.einsum('ijk,imc,knc->jpc', T - 1.0, U, W)
        grad_W = 2.0 * np.einsum('ijk,imc,jpc->knc', T - 1.0, U, V)
        
        return loss, (grad_U, grad_V, grad_W)
    
    # Simple gradient descent
    lr = 0.01
    for step in range(1000):
        if time.time() > deadline:
            break
        loss, (gU, gV, gW) = loss_and_grad()
        
        if loss < 1e-6:
            break
        
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Periodic restart or scaling?
        if step % 100 == 0:
            # Check for convergence to integers/halves
            pass
            
    # Final check
    T = np.einsum('imc,jpc,knc->ijk', U, V, W)
    if np.allclose(T, 1.0, atol=1e-3):
        # Snap to halves
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        T = np.einsum('imc,jpc,knc->ijk', U, V, W)
        if np.allclose(T, 1.0, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try to find a rank-14 decomposition for 2x3x3
    # If found, use it 3 times for 2x3x9 -> rank 42
    block_result = _find_block_decomposition(2, 3, 3, 14, deadline - 5.0)
    
    if block_result is not None:
        # Construct 2x3x9 decomposition
        U_full = np.zeros((n * m, 42))
        V_full = np.zeros((m * p, 42))
        W_full = np.zeros((p * n, 42))
        
        U_block = np.array(block_result['U']) # 6x14
        V_block = np.array(block_result['V']) # 9x14
        W_block = np.array(block_result['W']) # 6x14
        
        for b in range(3):
            U_full[:, b*14:(b+1)*14] = U_block
            # V needs to be mapped: V_block is 9x14 for 3x3.
            # V_full is 27x42.
            # Block b corresponds to k in [b*3, b*3+2]
            V_full[b*3:(b+1)*3, b*14:(b+1)*14] = V_block
            # W_full is 18x42.
            # W_block is 6x14. Indices (k,i) for k in 0..2.
            # W_full indices (k,i) for k in 0..8.
            W_full[b*3:(b+1)*3, b*14:(b+1)*14] = W_block
            
        # Verify
        if _verify(U_full, V_full, W_full, n, m, p, 42):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
            
    # Fallback to schoolbook
    return _schoolbook()
