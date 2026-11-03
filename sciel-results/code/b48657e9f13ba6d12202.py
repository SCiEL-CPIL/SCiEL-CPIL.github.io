"""Optimized program for second autocorrelation inequality."""
import numpy as np
import time

N = 50

def _autoconv(h):
    """Compute autoconvolution h*h (linear convolution)."""
    return np.convolve(h, h, mode='full')

def _objective_and_grad(h):
    """Compute objective and its gradient. Returns (score, grad)."""
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0, np.zeros(N)
    l2_sq = np.dot(h, h)
    if l2_sq <= 1e-15:
        return 0.0, np.zeros(N)
    conv = _autoconv(h)
    sup = np.max(np.abs(conv))
    if sup <= 1e-15:
        return 0.0, np.zeros(N)
    
    score = l2_sq / (l1 * sup)
    
    # Gradient of score w.r.t. h
    # score = l2_sq / (l1 * sup)
    # d(score)/d(h_j) = (2*h_j * l1 * sup - l2_sq * (d(l1*sup)/d(h_j))) / (l1 * sup)^2
    # = (2*h_j / (l1 * sup)) - (l2_sq / (l1 * sup)^2) * d(l1*sup)/d(h_j)
    # = (2*h_j / (l1 * sup)) - (score / (l1 * sup)) * d(l1*sup)/d(h_j)
    # = (1/(l1*sup)) * (2*h_j - score * d(l1*sup)/d(h_j))
    
    # d(l1)/d(h_j) = 1
    # d(sup)/d(h_j) = subgradient: use the index where sup is achieved
    idx_max = np.argmax(np.abs(conv))
    
    # d(conv[idx_max])/d(h_j): 
    # conv[m] = sum_{i} h[i]*h[m-i] for valid i
    # d(conv[m])/d(h_j) = h[m-j] + h[j-m] if m-j and j-m are both in [0,N-1] and m-j != j-m
    #                     = 2*h[m-j] if m-j == j-m (i.e., m=2j)
    # Actually: conv[m] = sum_{i=max(0,m-N+1)}^{min(m,N-1)} h[i]*h[m-i]
    # d/dh_j: 
    #   If j is in the valid range for i: term h[j]*h[m-j], derivative w.r.t. h[j] is h[m-j]
    #   If m-j is in the valid range for i and m-j != j: term h[m-j]*h[j], derivative is h[j]
    #   If m-j == j (m=2j): single term h[j]*h[j], derivative is 2*h[j]
    
    d_sup = np.zeros(N)
    m = idx_max
    for j in range(N):
        k = m - j
        if 0 <= k < N:
            if k == j:
                d_sup[j] += 2.0 * h[j]
            else:
                d_sup[j] += h[k]
                # Also check if j appears as the second index
                # When i = k = m-j, term is h[k]*h[m-k] = h[m-j]*h[j], d/dh[j] = h[m-j]
                # But we already counted this? No:
                # In the sum over i, when i=j: term h[j]*h[m-j], d/dh[j] = h[m-j]
                # When i=m-j (if m-j != j): term h[m-j]*h[j], d/dh[j] = h[m-j]
                # So total contribution is h[m-j] + h[m-j] = 2*h[m-j] if j != m-j
                # Wait, that's not right. Let me re-derive.
                # conv[m] = sum_i h[i] * h[m-i]
                # d(conv[m])/d(h[j]) = sum_i [delta(i,j)*h[m-i] + h[i]*delta(m-i,j)]
                # = h[m-j] (if 0<=m-j<N) + h[m-j] (if 0<=m-j<N and m-j != j, from second term with i=m-j)
                # If m-j == j: only one term in the sum contributes (i=j), and d(h[j]*h[j])/d(h[j]) = 2*h[j]
                # If m-j != j: two terms contribute: i=j gives h[m-j], i=m-j gives h[j]
                #   So total = h[m-j] + h[j]... no wait, h[i]*delta(m-i,j) with i=m-j gives h[m-j]*1 = h[m-j]
                #   And delta(i,j)*h[m-i] with i=j gives 1*h[m-j] = h[m-j]
                #   So total = h[m-j] + h[m-j] = 2*h[m-j]
                # Hmm, that doesn't seem right either. Let me just be careful.
                pass
    
    # Let me just compute d_sup numerically but more efficiently
    # Actually, let me use a different approach: compute the gradient via the chain rule more carefully
    # using the fact that we can compute d(conv[m])/d(h) as a vector efficiently
    
    # d(conv[m])/d(h_j) for all j:
    # For each j, the contribution is:
    #   h[m-j] if 0 <= m-j < N (from the i=j term)
    #   + h[j] if 0 <= m-j < N and m-j != j (from the i=m-j term, where h[i]*h[m-i] = h[m-j]*h[j])
    #   Wait, I'm confusing myself. Let me just do:
    
    d_conv = np.zeros(N)
    for j in range(N):
        k = m - j
        if 0 <= k < N:
            # From term i=j: h[j]*h[k], d/dh[j] = h[k]
            d_conv[j] += h[k]
            # From term i=k (if k != j): h[k]*h[m-k] = h[k]*h[j], d/dh[j] = h[k]
            if k != j and 0 <= k < N:
                d_conv[j] += h[k]
    
    # Now d(l1*sup)/d(h_j) = d(l1)/d(h_j) * sup + l1 * d(sup)/d(h_j)
    # = sup + l1 * d_conv[j] (using d_conv as subgradient for d(sup))
    
    # Actually sup = |conv[m]| and since conv values could be negative, but h >= 0 so conv >= 0
    # So sup = conv[m] and d(sup)/d(h_j) = d_conv[j]
    
    d_l1_sup = sup + l1 * d_conv
    
    grad = (2.0 * h * l1 * sup - l2_sq * d_l1_sup) / (l1 * sup) ** 2
    
    return score, grad

def _project_simplex(v, target_sum=1.0):
    """Project v onto {x >= 0, sum(x) = target_sum}."""
    v = np.maximum(v, 0)
    s = np.sum(v)
    if s <= 1e-15:
        return np.full(N, target_sum / N)
    return v * (target_sum / s)

def construct():
    deadline = time.time() + 140.0
    np.random.seed(42)
    
    best_h = np.ones(N) / N
    best_score = _objective_and_grad(best_h)[0]
    
    initializations = []
    initializations.append(np.ones(N) / N)
    
    for width in [2, 3, 5, 7, 10, 15]:
        h = np.zeros(N)
        center = N // 2
        start = max(0, center - width // 2)
        end = min(N, start + width)
        h[start:end] = 1.0 / (end - start)
        initializations.append(h.copy())
    
    x = np.linspace(-1, 1, N)
    h = np.maximum(np.cos(np.pi * x / 2), 0)
    h /= np.sum(h)
    initializations.append(h)
    
    x = np.linspace(-3, 3, N)
    h = np.exp(-x**2)
    h /= np.sum(h)
    initializations.append(h)
    
    for _ in range(15):
        h = np.random.dirichlet(np.ones(N))
        initializations.append(h)
    
    for init_h in initializations:
        if time.time() > deadline:
            break
        
        h = _project_simplex(init_h, 1.0)
        current_score, _ = _objective_and_grad(h)
        
        step = 0.05
        for iteration in range(300):
            if time.time() > deadline:
                break
            
            if current_score > best_score:
                best_score = current_score
                best_h = h.copy()
            
            _, grad = _objective_and_grad(h)
            grad_tangent = grad - np.mean(grad)
            
            h_new = _project_simplex(h + step * grad_tangent, 1.0)
            new_score, _ = _objective_and_grad(h_new)
            
            if new_score > current_score:
                h = h_new
                current_score = new_score
                step = min(step * 1.3, 0.3)
            else:
                step *= 0.5
                if step < 1e-9:
                    break
    
    # Local refinement
    if time.time() < deadline - 5:
        h = best_h.copy()
        current_score = best_score
        step = 0.01
        for iteration in range(200):
            if time.time() > deadline:
                break
            _, grad = _objective_and_grad(h)
            grad_tangent = grad - np.mean(grad)
            h_new = _project_simplex(h + step * grad_tangent, 1.0)
            new_score, _ = _objective_and_grad(h_new)
            if new_score > current_score:
                h = h_new
                current_score = new_score
                step = min(step * 1.1, 0.1)
            else:
                step *= 0.5
                if step < 1e-10:
                    break
        if current_score > best_score:
            best_score = current_score
            best_h = h
    
    return list(best_h)
