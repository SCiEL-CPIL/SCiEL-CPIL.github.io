import time
import random
import math
import numpy as np
from itertools import combinations

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def generate_smooth_numbers(max_val=10000, max_prime=7):
    """Generate smooth numbers up to max_val with prime factors <= max_prime."""
    smooths = set()
    primes = [2, 3, 5, 7]
    
    def gen(idx, current):
        if current > max_val:
            return
        smooths.add(current)
        if idx < len(primes):
            p = primes[idx]
            for i in range(1, 15):
                gen(idx + 1, current * p)
                current *= p
                if current > max_val:
                    break
    
    gen(0, 1)
    return sorted(smooths)

def build_collision_graph(pool):
    """
    Build a weighted graph where edge weights represent collision density
    between pairs of candidates. Higher weight = more likely to produce
    colliding sums/products when both are in the set.
    
    Returns adjacency matrix and degree matrix.
    """
    n = len(pool)
    W = np.zeros((n, n))
    
    for i in range(n):
        a = pool[i]
        for j in range(i + 1, n):
            b = pool[j]
            
            # Compute collision potential between a and b
            weight = 0.0
            
            # Sum collision: a+b collides with other sums if there exist
            # x,y in pool such that x+y = a+b with {x,y} != {a,b}
            # Approximate by checking if a+b can be formed by other pairs
            # Use a heuristic based on multiplicative structure
            
            ratio = max(a, b) / min(a, b)
            gcd_ab = math.gcd(a, b)
            
            # Base weight: smooth numbers have more factorizations
            base = 1.0
            
            # Close values create more sum collisions
            if ratio < 1.3:
                base += 3.0
            elif ratio < 1.5:
                base += 2.0
            elif ratio < 2.0:
                base += 1.0
            
            # Multiplicative relationships create product collisions
            if a % b == 0 or b % a == 0:
                base += 4.0
            elif gcd_ab > 1:
                base += 2.0
            
            # Both being powers of 2 or 3 creates strong structure
            def is_power_of(x, p):
                while x % p == 0 and x > 1:
                    x //= p
                return x == 1
            
            if is_power_of(a, 2) and is_power_of(b, 2):
                base += 3.0
            elif is_power_of(a, 3) and is_power_of(b, 3):
                base += 3.0
            elif is_power_of(a, 2) and is_power_of(b, 3):
                base += 2.0
            
            W[i, j] = base
            W[j, i] = base
    
    return W

def spectral_select(W, k, seed=42):
    """
    Use spectral methods to select k nodes with highest internal edge density.
    
    1. Compute the Laplacian L = D - W
    2. Find the k eigenvectors corresponding to the smallest eigenvalues
    3. Cluster using k-means on these eigenvectors
    4. Return the cluster with highest internal density
    """
    n = W.shape[0]
    if n <= k:
        return list(range(n))
    
    # Compute degree matrix
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback: return first k nodes
        return list(range(k))
    
    # Use the k smallest eigenvectors (excluding the trivial one)
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx[:k]]
    
    # Normalize rows for k-means
    norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = eigenvectors / norms
    
    # Simple k-means
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    
    for _ in range(50):  # k-means iterations
        # Assign to nearest center
        dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        
        # Update centers
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centers[c] = X[mask].mean(axis=0)
    
    # Find the cluster with highest internal edge density
    best_cluster = None
    best_density = -1
    
    for c in range(k):
        mask = labels == c
        if np.sum(mask) < k:
            continue
        
        cluster_nodes = np.where(mask)[0]
        if len(cluster_nodes) >= k:
            # Take top k by degree within cluster
            sub_W = W[np.ix_(cluster_nodes, cluster_nodes)]
            degrees = sub_W.sum(axis=1)
            top_k_idx = np.argsort(degrees)[-k:]
            selected = cluster_nodes[top_k_idx]
            
            # Compute internal density
            sub_selected = W[np.ix_(selected, selected)]
            density = sub_selected.sum() / (k * (k - 1) + 1e-10)
            
            if density > best_density:
                best_density = density
                best_cluster = selected
    
    if best_cluster is None:
        # Fallback: select by degree
        degrees = W.sum(axis=1)
        return list(np.argsort(degrees)[-k:])
    
    return list(best_cluster)

def greedy_density_select(W, k):
    """
    Greedy selection: iteratively add the node that maximizes
    internal edge density with already-selected nodes.
    """
    n = W.shape[0]
    selected = []
    remaining = set(range(n))
    
    # Start with the node with highest degree
    degrees = W.sum(axis=1)
    start = np.argmax(degrees)
    selected.append(start)
    remaining.remove(start)
    
    for _ in range(k - 1):
        best_node = None
        best_score = -1
        
        for node in remaining:
            # Score: sum of weights to already-selected nodes
            score = sum(W[node, s] for s in selected)
            if score > best_score:
                best_score = score
                best_node = node
        
        if best_node is None:
            break
        
        selected.append(best_node)
        remaining.remove(best_node)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Generate smooth numbers as candidate pool
    smooth_pool = generate_smooth_numbers(10000, 7)
    
    # Ensure we have enough candidates
    if len(smooth_pool) < 50:
        for i in range(1, 30):
            if i not in smooth_pool:
                smooth_pool.append(i)
        smooth_pool = sorted(smooth_pool)
    
    # Limit pool size for computational tractability
    pool_size = min(len(smooth_pool), 150)
    pool = smooth_pool[:pool_size]
    
    # Build collision graph
    W = build_collision_graph(pool)
    
    k_target = 20
    best_set = None
    best_score = float('inf')
    
    # Phase 1: Spectral selection
    spectral_indices = spectral_select(W, k_target, seed=42)
    spectral_set = [pool[i] for i in spectral_indices]
    spectral_score = compute_metrics(spectral_set)
    best_score = spectral_score
    best_set = spectral_set[:]
    
    # Phase 2: Greedy density selection
    greedy_indices = greedy_density_select(W, k_target)
    greedy_set = [pool[i] for i in greedy_indices]
    greedy_score = compute_metrics(greedy_set)
    
    if greedy_score < best_score:
        best_score = greedy_score
        best_set = greedy_set[:]
    
    # Try multiple spectral seeds for diversity
    if time.time() < deadline - 30.0:
        for seed in [0, 1, 2, 3, 4]:
            if time.time() > deadline - 25.0:
                break
            
            indices = spectral_select(W, k_target, seed=seed)
            candidate_set = [pool[i] for i in indices]
            score = compute_metrics(candidate_set)
            
            if score < best_score:
                best_score = score
                best_set = candidate_set[:]
    
    # Phase 3: Local search refinement
    current_set = best_set[:]
    current_score = best_score
    
    # Simulated annealing
    T_start = 5.0
    T_end = 0.1
    num_steps = 100000
    
    step = 0
    while step < num_steps:
        if time.time() > deadline:
            break
        
        step += 1
        T = T_start * (T_end / T_start) ** (step / num_steps)
        
        idx = random.randint(0, 19)
        old_val = current_set[idx]
        
        new_val = random.choice(pool)
        
        if new_val in current_set:
            continue
        
        current_set[idx] = new_val
        new_score = compute_metrics(current_set)
        
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            current_set[idx] = old_val
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(20):
                old_val = best_set[i]
                for new_val in pool:
                    if new_val in best_set:
                        continue
                    best_set[i] = new_val
                    s = compute_metrics(best_set)
                    if s < best_score:
                        best_score = s
                        improved = True
                    else:
                        best_set[i] = old_val
                best_set[i] = old_val
    
    return sorted(best_set)
