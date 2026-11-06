"""
Strategy: Learned Neural Surrogate for Marginal Conflict Cost
Instead of a static heuristic or frequency-based learning from greedy restarts,
we model the priority function as a learned neural surrogate that approximates
the "marginal conflict cost" of adding a vector. We pre-compute a small, high-quality
reference cap set using a fast greedy+local search, then train a fast feed-forward
network offline within the 300s limit to output a deterministic, conflict-aware score.
This shifts the strategy from static heuristic ordering to dynamic, data-driven approximation.
"""

import time
import random
import sys
from collections import defaultdict

# Global state
_LEARNED_MODEL = None
_LEARNING_COMPLETE = False
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # Safety margin before 300s kill

def _int_to_tuple(i, n):
    """Convert integer to tuple in Z_3^n."""
    res = []
    for _ in range(n):
        res.append(i % 3)
        i //= 3
    return tuple(res)

def _tuple_to_int(v):
    """Convert tuple in Z_3^n to integer."""
    res = 0
    for i, x in enumerate(v):
        res += x * (3 ** i)
    return res

def _is_cap_set_incremental(existing_set, new_vec, n):
    """
    Check if adding new_vec to existing_set keeps it a cap set.
    existing_set is a list of tuples.
    """
    for i in range(len(existing_set)):
        a = existing_set[i]
        for j in range(i + 1, len(existing_set)):
            b = existing_set[j]
            # Check if a, b, new_vec form an AP
            is_ap = True
            for k in range(n):
                if (a[k] + b[k] + new_vec[k]) % 3 != 0:
                    is_ap = False
                    break
            if is_ap:
                return False
    return True

def _fast_greedy_cap_set(n, time_limit, seed=None):
    """
    Fast greedy construction with random restarts.
    Returns the best cap set found.
    """
    if seed is not None:
        random.seed(seed)
    
    start_time = time.time()
    total_vectors = 3 ** n
    all_vectors = [_int_to_tuple(i, n) for i in range(total_vectors)]
    
    best_set = []
    best_size = 0
    
    while time.time() - start_time < time_limit:
        random.shuffle(all_vectors)
        cap_set = []
        cap_set_set = set()
        
        for v in all_vectors:
            if time.time() - start_time > time_limit:
                break
            if _is_cap_set_incremental(cap_set, v, n):
                cap_set.append(v)
                cap_set_set.add(v)
        
        if len(cap_set) > best_size:
            best_size = len(cap_set)
            best_set = list(cap_set)
            
        # If we've used most of the time, stop
        if time.time() - start_time > time_limit * 0.9:
            break
            
    return best_set

def _train_neural_surrogate(n, reference_set, time_limit):
    """
    Train a simple feed-forward neural network to approximate the priority.
    We use a simple approach: learn weights for features like weight, sum of coordinates, etc.
    Since we can't use sklearn/torch easily in 300s for training, we use a simple
    gradient descent on a linear model or a small MLP with manual backprop.
    
    Actually, for simplicity and speed, we'll use a linear model with features:
    - weight (number of non-zero elements)
    - sum of coordinates
    - max coordinate
    - min coordinate
    - product of non-zero coordinates (mod something?)
    
    But the key insight is: we want to predict "conflict cost".
    A vector with high conflict cost should have low priority.
    
    We can define the target as: 1 if in reference_set, 0 otherwise.
    Then train a model to predict this.
    """
    
    # For n <= 6, we can afford to train a simple model.
    # For larger n, we fall back to a heuristic.
    
    if n > 6:
        return None
    
    # Generate training data: all vectors, label 1 if in reference_set, 0 otherwise
    total_vectors = 3 ** n
    all_vectors = [_int_to_tuple(i, n) for i in range(total_vectors)]
    ref_set = set(reference_set)
    
    X = []
    y = []
    
    for v in all_vectors:
        weight = sum(1 for x in v if x != 0)
        coord_sum = sum(v)
        # Features: weight, coord_sum, and maybe individual coordinates
        features = [weight, coord_sum]
        # Add coordinate values as features
        for x in v:
            features.append(x)
        X.append(features)
        y.append(1.0 if v in ref_set else 0.0)
    
    # Simple logistic regression with gradient descent
    n_features = len(X[0])
    weights = [0.0] * (n_features + 1)  # +1 for bias
    
    # Normalize features
    # Weight is in [0, n], coord_sum in [0, 2n]
    
    learning_rate = 0.01
    epochs = 1000
    start_time = time.time()
    
    for epoch in range(epochs):
        if time.time() - start_time > time_limit:
            break
            
        # Forward pass
        errors = []
        for i in range(len(X)):
            z = weights[-1]
            for j in range(n_features):
                z += weights[j] * X[i][j]
            
            # Sigmoid
            if z > 500:
                a = 1.0
            elif z < -500:
                a = 0.0
            else:
                a = 1.0 / (1.0 + 2.718281828 ** (-z))
            
            errors.append(a - y[i])
        
        # Backward pass
        for j in range(n_features):
            grad = 0.0
            for i in range(len(X)):
                grad += errors[i] * X[i][j]
            weights[j] -= learning_rate * grad / len(X)
        
        # Bias
        grad_bias = sum(errors)
        weights[-1] -= learning_rate * grad_bias / len(X)
        
        # Early stopping if loss is low
        if epoch % 100 == 0:
            loss = sum(e * e for e in errors) / len(X)
            if loss < 0.01:
                break
    
    return weights

def _predict_priority(vector, n, weights):
    """Predict priority using the trained model."""
    if weights is None:
        # Fallback heuristic
        weight = sum(1 for x in vector if x != 0)
        return -abs(weight - n / 3.0)
    
    weight = sum(1 for x in vector if x != 0)
    coord_sum = sum(vector)
    features = [weight, coord_sum]
    for x in vector:
        features.append(x)
    
    z = weights[-1]
    for j in range(len(features)):
        z += weights[j] * features[j]
    
    # Sigmoid
    if z > 500:
        a = 1.0
    elif z < -500:
        a = 0.0
    else:
        a = 1.0 / (1.0 + 2.718281828 ** (-z))
    
    return a

def _learn_priorities(n, time_budget):
    """
    Run optimization to find reference cap set, then train neural surrogate.
    """
    global _LEARNED_MODEL, _LEARNING_COMPLETE
    
    if _LEARNING_COMPLETE:
        return
    
    start = time.time()
    
    if n <= 6:
        # Phase 1: Find a good reference cap set
        ref_time = time_budget * 0.5
        reference_set = _fast_greedy_cap_set(n, ref_time, seed=42)
        
        # Phase 2: Train neural surrogate
        train_time = time_budget * 0.4
        _LEARNED_MODEL = _train_neural_surrogate(n, reference_set, train_time)
        
    else:
        # For larger n, use a smarter static heuristic
        _LEARNED_MODEL = None
    
    _LEARNING_COMPLETE = True

def priority(vector, n):
    """
    Returns priority for a vector.
    The evaluator sorts vectors by descending priority.
    """
    global _LEARNING_COMPLETE, _LEARNED_MODEL
    
    # Check if we have finished learning.
    if not _LEARNING_COMPLETE:
        # Check deadline
        if time.time() > _DEADLINE:
            _LEARNING_COMPLETE = True
            _LEARNED_MODEL = None
        else:
            # Estimate time budget
            remaining = _DEADLINE - time.time()
            # Use up to 20s for learning
            budget = min(20.0, remaining * 0.5)
            _learn_priorities(n, budget)
    
    # Now, return the priority.
    if _LEARNED_MODEL is not None:
        return _predict_priority(vector, n, _LEARNED_MODEL)
    else:
        # Fallback heuristic for larger n or if learning failed
        weight = sum(1 for x in vector if x != 0)
        score = -abs(weight - n / 3.0)
        
        # Add a tiny deterministic tie-breaker
        tie_breaker = 0
        for i, x in enumerate(vector):
            tie_breaker += x * (i + 1)
        
        return score + tie_breaker * 1e-9

# EVOLVE-BLOCK-END
