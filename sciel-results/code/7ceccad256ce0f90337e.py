import numpy as np
import time

def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    start_time = time.time()
    TIME_LIMIT = 300.0
    MARGIN = 5.0
    
    def time_remaining():
        return TIME_LIMIT - MARGIN - (time.time() - start_time)
    
    # 1. Construct Target Matrix T
    # T[i, j] represents how "complementary" item i and item j are.
    # High value if d[i] + d[j] is close to capacity (good fit) but not over.
    # We use a Gaussian kernel around the ideal sum (capacity).
    # Ideal sum is capacity. If sum is slightly less, it's good. If much less, waste is high.
    # Let's define target as exp(-((d_i + d_j - capacity)^2) / sigma^2) for valid pairs.
    # For invalid pairs (d_i + d_j > capacity), target is 0.
    
    pair_sums = d[:, None] + d[None, :]
    valid_mask = pair_sums <= capacity
    
    # Sigma controls how strict the "complementarity" is.
    # If sigma is small, only pairs very close to capacity are good.
    # If sigma is large, more pairs are considered good.
    # A reasonable sigma might be around 10-20 for capacity 150.
    sigma = 15.0
    
    T = np.zeros((n, n), dtype=float)
    T[valid_mask] = np.exp(-((pair_sums[valid_mask] - capacity) ** 2) / (2 * sigma ** 2))
    
    # Ensure diagonal is 0 (item cannot be paired with itself in this context for heuristic)
    np.fill_diagonal(T, 0)
    
    # 2. Autoencoder to learn embeddings
    # Architecture: Input (n) -> Hidden (k) -> Output (n)
    # We want to find embeddings E (n x k) such that E @ E.T approximates T.
    # This is essentially learning a low-rank approximation of T that preserves the structure.
    # We can use gradient descent on the reconstruction loss.
    
    latent_dim = 16  # Low dimensionality
    
    # Initialize weights
    # W1: (n, latent_dim) - Encoder
    # W2: (latent_dim, n) - Decoder
    # We can simplify by just optimizing E directly if we assume symmetric structure,
    # but a standard AE is safer. However, since T is symmetric, we can just optimize E
    # such that E @ E.T ~ T. This is like Factor Analysis.
    
    # Let's use a simple factorization approach with gradient descent.
    # Minimize || E @ E.T - T ||^2
    # Gradient w.r.t E: 2 * (E @ E.T - T) @ E
    
    # Initialize E with small random values
    rng = np.random.default_rng(42)
    E = rng.normal(0, 0.1, (n, latent_dim))
    
    lr = 0.01
    max_iter = 1000
    
    # Precompute for speed
    for i in range(max_iter):
        if time_remaining() < 10.0:
            break
            
        # Forward pass
        S = E @ E.T
        error = S - T
        
        # Gradient
        grad = 2 * error @ E
        
        # Update
        E -= lr * grad
        
        # Optional: Regularization to prevent explosion
        E *= 0.999
        
        # Check convergence
        if i % 100 == 0 and i > 0:
            loss = np.sum(error ** 2)
            if loss < 1e-6:
                break
        
    # 3. Compute Heuristic from Embeddings
    # Heuristic H[i, j] should be high if embeddings are similar.
    # Cosine similarity is a good choice.
    
    # Normalize embeddings
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    E_norm = E / norms
    
    # Cosine similarity matrix
    H = E_norm @ E_norm.T
    
    # Ensure H is symmetric and non-negative
    H = (H + H.T) / 2.0
    
    # Clip negative values to 0
    H = np.maximum(H, 0)
    
    # Apply validity mask: if pair is invalid, set to 1e-9
    H[~valid_mask] = 1e-9
    
    # Scale heuristic to a reasonable range
    # The ACO uses pheromone * heuristic. Pheromone starts at 1.
    # Heuristic values should be O(1) or slightly larger.
    max_h = np.max(H)
    if max_h > 0:
        H = H / max_h * 10.0
    
    # Final safety checks
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
