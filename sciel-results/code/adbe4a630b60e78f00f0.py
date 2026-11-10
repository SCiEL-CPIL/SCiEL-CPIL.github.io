import time
import numpy as np

T_START = time.perf_counter()
LIMIT = 300.0
MARGIN = 5.0

def _deadline_left():
    return (T_START + LIMIT) - time.perf_counter()

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure positive
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    cap = 25.0
    
    # Per-constraint ratios
    ratios = prizes / weights  # shape (5, 100)
    
    # Soft-min via log-sum-exp (smoother surrogate for min)
    # softmin(x) = -1/beta * log(sum(exp(-beta * x)))
    beta = 10.0
    neg_beta_ratios = -beta * ratios
    # Use log-sum-exp trick for numerical stability
    max_neg = np.max(neg_beta_ratios, axis=0)
    lse = max_neg + np.log(np.sum(np.exp(neg_beta_ratios - max_neg), axis=0))
    soft_min = -lse / beta
    
    # Also compute a slack bonus: items with smaller max weight are more flexible
    max_weight = weights.max(axis=0)  # shape (100,)
    slack = 1.0 / (max_weight + 1e-9)
    
    # Base heuristic: soft_min efficiency * slack bonus
    heuristic = soft_min * slack
    
    # --- Co-occurrence penalty from greedy packing ---
    # Run a greedy pack to identify selected items and rejected items
    # Then penalize items that share high weight correlation with top-selected items
    
    # Greedy order: by soft_min * slack (current heuristic)
    order = np.argsort(-heuristic, kind='stable')
    
    rem = np.full(n_constraints, cap, dtype=np.float64)
    selected = []
    rejected = []
    
    for i in order:
        if np.all(weights[:, i] <= rem):
            selected.append(i)
            rem -= weights[:, i]
        else:
            rejected.append(i)
    
    # Compute weight correlation matrix (only for selected items vs all items)
    # We want to penalize items that are "similar" in weight profile to selected items
    # because they compete for the same resources
    
    if len(selected) > 0:
        # Weight profiles of selected items: shape (n_selected, n_constraints)
        sel_weights = weights[:, selected]
        
        # For each item, compute its "conflict score" with selected items
        # High conflict = high correlation in weight usage
        # We use a simple dot-product-based similarity (cosine-like)
        
        # Normalize weight vectors
        norms_all = np.linalg.norm(weights, axis=0, keepdims=True)
        norms_all = np.maximum(norms_all, 1e-9)
        w_norm = weights / norms_all  # shape (n_constraints, n_items)
        
        norms_sel = np.linalg.norm(sel_weights, axis=0, keepdims=True)
        norms_sel = np.maximum(norms_sel, 1e-9)
        sel_norm = sel_weights / norms_sel  # shape (n_constraints, n_selected)
        
        # Similarity: shape (n_items, n_selected)
        sim = w_norm.T @ sel_norm  # shape (n_items, n_selected)
        
        # Conflict score: mean similarity with selected items
        # Items that are similar to selected items are more likely to conflict
        conflict_score = np.mean(sim, axis=1)  # shape (n_items,)
        
        # Penalize items with high conflict score
        # The penalty should be stronger for items that were rejected
        # But we apply a soft penalty to all items to guide the ant colony
        
        # Scale conflict score to [0, 1]
        c_min = conflict_score.min()
        c_max = conflict_score.max()
        if c_max > c_min:
            conflict_norm = (conflict_score - c_min) / (c_max - c_min)
        else:
            conflict_norm = np.zeros(n_items)
        
        # Apply penalty: reduce heuristic for items with high conflict
        # penalty_factor = 1 - alpha * conflict_norm
        alpha = 0.3  # Penalty strength
        penalty_factor = 1.0 - alpha * conflict_norm
        penalty_factor = np.maximum(penalty_factor, 0.1)  # Keep at least 10% of original
        
        heuristic = heuristic * penalty_factor
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
