"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    n = len(mu)
    deadline = time.time() + 280  # safety margin before 300s limit
    
    # Base EI scores
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle edge cases
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Look-ahead: simulate greedy selection of next 5 points
    # Score each candidate by its expected contribution to batch minimum
    # 
    # Strategy: For each candidate i, estimate how much it reduces the
    # expected minimum if selected now, considering that nearby points
    # will have reduced uncertainty after seeing i's value.
    #
    # Approximation: The information gain from selecting point i reduces
    # the effective sigma of nearby points. We estimate the batch-aware
    # score as EI(i) + expected reduction in EI of nearby points due to
    # correlated information.
    
    # Use a simple local correlation model: points close in mu-space
    # (proxy for input-space proximity) share information
    # Since we don't have coordinates, use mu and sigma similarity
    
    # For efficiency, only do detailed look-ahead for top candidates
    top_k = min(200, n)
    top_idx = np.argpartition(-ei, top_k)[:top_k]
    
    # Initialize batch-aware scores
    batch_scores = ei.copy()
    
    if time.time() > deadline - 30:
        return batch_scores
    
    # For top candidates, compute look-ahead contribution
    # Approximate: if we select candidate i, the expected improvement
    # from the best of the remaining top candidates decreases because
    # their sigma effectively reduces (information gain)
    
    # Simple model: information gain from selecting i reduces sigma
    # of candidates with similar (mu, sigma) profile
    # Use a kernel over (mu, sigma) space
    
    mu_top = mu[top_idx]
    sigma_top = sigma[top_idx]
    
    for i in range(len(top_idx)):
        if time.time() > deadline - 20:
            break
        
        idx_i = top_idx[i]
        mu_i = mu[idx_i]
        sigma_i = sigma[idx_i]
        
        # Compute similarity to other top candidates
        # Use difference in mu and relative difference in sigma
        dm = np.abs(mu_top - mu_i)
        ds = np.abs(sigma_top - sigma_i) / (sigma_top + sigma_i + 1e-9)
        
        # Gaussian kernel over combined distance
        dist = dm * 2.0 + ds * 0.5
        similarity = np.exp(-0.5 * (dist / 0.3) ** 2)
        
        # Information gain: selecting i reduces uncertainty of similar points
        # Effective sigma reduction for point j: sigma_j * (1 - alpha * sim_ij)
        alpha = 0.3  # strength of information sharing
        
        # Compute EI for each top candidate j under reduced sigma
        # if i is selected
        sigma_reduced = sigma_top * (1 - alpha * similarity)
        sigma_reduced = np.maximum(sigma_reduced, 1e-6)
        
        z_j = (f_best - mu_top) / (sigma_reduced + 1e-9)
        ei_j_reduced = (f_best - mu_top) * norm.cdf(z_j) + sigma_reduced * norm.pdf(z_j)
        ei_j_reduced = np.where(np.isfinite(ei_j_reduced), ei_j_reduced, 0.0)
        
        # The value of selecting i: its own EI plus the expected best
        # EI among remaining candidates (which is reduced due to info gain)
        # But we want to score i by how much it contributes to the batch
        # 
        # Contribution = EI(i) + max(EI_j for j != i under reduced sigma)
        #                - max(EI_j for j != i under original sigma)
        # 
        # This is complex. Simpler: score i by EI(i) + expected reduction
        # in the best remaining EI due to information gain
        
        ei_original_best = np.max(np.delete(ei[top_idx], i))
        ei_reduced_best = np.max(np.delete(ei_j_reduced, i))
        
        # The information gain benefit: if reduced best is lower, it means
        # we've already "used up" some improvement, so i's marginal value is
        # its own EI. If reduced best is similar, i adds value.
        
        # Batch-aware score: weight EI by how unique i's information is
        # High similarity to others => less unique => lower batch score
        uniqueness = 1.0 - np.mean(similarity)  # average similarity to others
        batch_scores[idx_i] = ei[idx_i] * (0.5 + 0.5 * uniqueness)
    
    return batch_scores
# EVOLVE-BLOCK-END
