import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit
    
    d = np.array(distance_matrix, dtype=float)
    coords = np.array(coordinates, dtype=float)
    demands = np.array(demands, dtype=float)
    n = d.shape[0]
    
    # 1. Base distance heuristic
    # Power 4.0 was found to be good in the current best program
    heuristic = np.zeros_like(d)
    mask = d > 0
    heuristic[mask] = 1.0 / (d[mask] ** 4.0)
    
    # 2. Angular coherence (from current best)
    depot = coords[0]
    # Calculate angles of customers relative to depot
    # Avoid division by zero or atan2(0,0) if a customer is exactly at depot (unlikely in uniform square)
    dx = coords[1:, 0] - depot[0]
    dy = coords[1:, 1] - depot[1]
    angles = np.arctan2(dy, dx)
    angles = angles % (2 * np.pi)
    
    # Circular angular distance
    angle_diff = np.abs(angles[:, np.newaxis] - angles[np.newaxis, :])
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    
    # Smooth coherence score
    sigma_angle = 0.7
    coherence_angle = np.exp(-angle_diff / sigma_angle)
    
    # Full matrix for angular coherence
    full_coherence_angle = np.ones((n, n))
    full_coherence_angle[1:, 1:] = coherence_angle
    
    # 3. Spectral Clustering Affinity
    # We apply this to the customer subgraph (indices 1 to n-1)
    # The idea is to find natural clusters and boost edges within them.
    D_c = d[1:, 1:]
    n_c = n - 1
    
    if n_c > 1:
        # Similarity matrix: Gaussian kernel
        # Estimate sigma from median distance or fixed value
        # Distances in unit square are typically < 0.7
        # Let's use a fixed sigma that captures local structure
        sigma_sim = 0.2 
        # To avoid overflow/underflow, we can normalize D_c
        # But exp(-D^2/sigma^2) is fine for small D
        S = np.exp(-(D_c ** 2) / (2 * sigma_sim ** 2))
        
        # Ensure symmetry and no self-loops for Laplacian calculation
        np.fill_diagonal(S, 0)
        
        # Graph Laplacian L = D - S
        # D is diagonal matrix of degrees
        degrees = S.sum(axis=1)
        L = np.diag(degrees) - S
        
        # Find eigenvalues and eigenvectors
        # We only need the Fiedler vector (2nd smallest eigenvalue)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Sort is already done by eigh for symmetric matrices (ascending)
            # The first eigenvector is constant (eigenvalue 0)
            # The second is the Fiedler vector
            fiedler_vec = eigvecs[:, 1]
            
            # Spectral affinity based on difference in Fiedler values
            # Nodes with similar Fiedler values are in the same cluster
            fv_diff = np.abs(fiedler_vec[:, np.newaxis] - fiedler_vec[np.newaxis, :])
            
            # Width of the affinity kernel
            # Fiedler values are typically in range [-1, 1] or similar scale
            # Let's normalize Fiedler vector to have unit variance for stability
            fv_std = np.std(fiedler_vec)
            if fv_std > 1e-9:
                fiedler_vec_norm = fiedler_vec / fv_std
            else:
                fiedler_vec_norm = fiedler_vec
            
            fv_diff_norm = np.abs(fiedler_vec_norm[:, np.newaxis] - fiedler_vec_norm[np.newaxis, :])
            
            # Affinity: exp(-diff^2 / tau^2)
            # tau controls how "wide" the cluster is considered
            tau_spec = 0.8
            coherence_spec = np.exp(-(fv_diff_norm ** 2) / (2 * tau_spec ** 2))
            
            # Full matrix for spectral coherence
            full_coherence_spec = np.ones((n, n))
            full_coherence_spec[1:, 1:] = coherence_spec
            
        except np.linalg.LinAlgError:
            # Fallback if eigen decomposition fails
            full_coherence_spec = np.ones((n, n))
    else:
        full_coherence_spec = np.ones((n, n))
        
    # 4. Combine Heuristics
    # We want to boost edges that are short, angularly coherent, and spectrally coherent.
    # Multiplicative combination of "quality" scores.
    # Base is distance.
    # Boost factor = w_a * AngleCoherence + w_s * SpecCoherence + w_0 * 1
    
    w_angle = 0.4
    w_spec = 0.3
    w_base = 1.0 - w_angle - w_spec # 0.3
    
    # Ensure coherence values are positive (they are, being exp)
    # Normalize coherence to [0,1] range roughly? They are already in [0,1]
    
    boost_factor = w_base * 1.0 + w_angle * full_coherence_angle + w_spec * full_coherence_spec
    
    # Apply boost to distance heuristic
    heuristic *= boost_factor
    
    # 5. Demand-aware boost (from current best)
    # Favor destinations with higher demand to pack trucks
    demand_boost = 1.0 + 0.1 * (demands / capacity)
    heuristic *= demand_boost[np.newaxis, :]
    
    # Ensure all values are positive
    heuristic[heuristic <= 0] = 1e-9
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    return heuristic
