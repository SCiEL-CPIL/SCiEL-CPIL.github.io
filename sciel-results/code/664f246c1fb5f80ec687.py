import numpy as np

N = 50

def construct():
    def evaluate(h):
        norm1 = np.sum(h)
        if norm1 == 0:
            return 0.0
        norm2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv == 0:
            return 0.0
        return norm2_sq / (norm1 * max_conv)

    def optimize_weights(support, iterations=100):
        h = np.zeros(N)
        h[support] = 1.0 / len(support)
        
        best_score = evaluate(h)
        best_h = h.copy()
        
        for _ in range(iterations):
            conv = np.convolve(h, h, mode='full')
            k_max = np.argmax(conv)
            peak_val = conv[k_max]
            
            # Identify indices contributing to the peak
            # h[i] * h[k_max - i]
            contributors = set()
            for i in support:
                j = k_max - i
                if 0 <= j < N and h[j] > 0:
                    contributors.add(i)
                    contributors.add(j)
            
            if not contributors:
                break
                
            # Reduce weights of contributors
            # Use a soft penalty
            for idx in contributors:
                h[idx] *= 0.95
                
            # Normalize
            norm = np.sum(h)
            if norm > 0:
                h /= norm
            
            score = evaluate(h)
            if score > best_score:
                best_score = score
                best_h = h.copy()
                
        return best_h, best_score

    best_overall_score = 0.0
    best_overall_h = np.zeros(N)
    
    # Try different support sizes
    for k_size in [8, 9, 10, 11]:
        for trial in range(20):
            # Random support
            support = np.random.choice(N, k_size, replace=False)
            h_opt, score = optimize_weights(support, iterations=50)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_h = h_opt
                
    return list(best_overall_h)
