import time
import math
import random
import numpy as np
from itertools import product
from collections import deque

def compute_score(U):
    U = tuple(sorted(set(U)))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def fast_score(U):
    """Fast scoring using numpy for vectorized set size computation."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U or len(U) < 2:
        return -1.0
    max_u = int(U.max())
    if max_u == 0:
        return -1.0
    
    n = len(U)
    # For very small n, numpy overhead might be comparable to loop, but n^2 is small.
    # For n > 50, n^2 is 2500, which is fine.
    # Let's always use numpy for consistency and speed on modern CPUs.
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    
    if n_sums == 0:
        return -1.0
        
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def local_search(U, deadline, max_steps=1000, step_range=5):
    """
    Local search to improve the score of U.
    Tries to move each element by a small amount.
    """
    U = sorted(set(U))
    if 0 not in U:
        U.insert(0, 0)
    U = sorted(U)
    
    current_score = fast_score(U)
    if current_score <= 0:
        return U, current_score

    n = len(U)
    
    # Fixed 0, others can move
    # We will try to adjust elements U[1:]
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        improved = False
        # Shuffle order of indices to avoid bias
        indices = list(range(1, n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            old_val = U[i]
            # Try moving in a window around old_val
            # The window size can depend on the magnitude of old_val
            # But for small sets, values are small.
            low = max(0, old_val - step_range)
            high = old_val + step_range
            
            # Ensure we don't collide with other elements
            # We check validity after constructing new_U
            
            best_new_val = old_val
            best_new_score = current_score
            
            # Sample some candidates in the range, plus exact neighbors
            candidates = set()
            candidates.add(old_val - 1)
            candidates.add(old_val + 1)
            if step_range > 1:
                # Random samples
                for _ in range(10):
                    candidates.add(random.randint(low, high))
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                if new_val < 0:
                    continue
                
                new_U = U[:i] + [new_val] + U[i+1:]
                # Check for duplicates
                if len(set(new_U)) != len(new_U):
                    continue
                # Check sorted order is maintained (since we are replacing one element,
                # we need to ensure it doesn't violate ordering if we assume U is sorted.
                # Actually, we just sort it for the score function, but for local search
                # efficiency, we should keep it sorted.
                # If new_val is not in the right place, the set is still valid, just unsorted.
                # fast_score sorts it.
                
                s = fast_score(new_U)
                if s > best_new_score:
                    best_new_score = s
                    best_new_val = new_val
            
            if best_new_val != old_val:
                U[i] = best_new_val
                # Re-sort U to maintain invariant for next iterations?
                # Not strictly necessary if fast_score sorts, but helps if we relied on order.
                # However, indices in the next iteration would be wrong if we swap values.
                # So we must re-sort U and then we lose the index mapping?
                # No, we are iterating over indices of the *current* U.
                # If we change U[i], the set changes.
                # Let's just sort U after an update.
                U = sorted(U)
                current_score = best_new_score
                improved = True
                # Restart the index loop? Or continue?
                # Continuing is fine, but indices might point to different values now.
                # To be safe, break and restart the outer step loop?
                # No, that's too slow.
                # Just continue. The indices are from the pre-update state.
                # This is a "steepest ascent" with stale indices.
                # It's a heuristic. It works well in practice.
        
        if not improved:
            # Try a larger step range?
            step_range += 2
            if step_range > 20:
                break
                
    return U, current_score

def generate_random_set(n, max_val, deadline):
    """Generate a random set of size n containing 0."""
    if n == 1:
        return [0]
    # Generate n-1 distinct non-zero integers
    # Use a power-law distribution to favor small numbers?
    # Or just uniform?
    # Small sets with small max are better.
    
    # Strategy: Pick 0, then pick n-1 numbers from 1..max_val
    # Ensure distinctness.
    if max_val < n:
        return list(range(n))
        
    S = {0}
    while len(S) < n:
        # Bias towards smaller numbers
        # x = int(random.random() ** 2 * max_val)
        x = random.randint(1, max_val)
        S.add(x)
    return sorted(S)

def generate_ldpc_inspired_set(num_codewords, max_weight, deadline):
    """
    Generate a set inspired by LDPC code structure.
    
    The idea: select binary vectors with controlled weight and spacing
    to minimize sum collisions while maximizing difference diversity.
    
    We map binary codewords to integers via positional encoding.
    """
    if time.time() > deadline:
        return [0, 1]
    
    # Start with 0
    S = {0}
    
    # Generate binary vectors with controlled properties
    # Each vector has weight between 1 and max_weight
    # We want high girth: no small cycles in the bipartite graph
    
    # Use a greedy approach: add vectors that maximize minimum distance
    # from existing vectors (in terms of their integer representations)
    
    vectors = []
    
    # First vector: single 1
    v1 = [0] * max_weight
    v1[0] = 1
    vectors.append(v1)
    
    # Add more vectors with controlled overlap
    for _ in range(num_codewords - 1):
        if time.time() > deadline:
            break
            
        # Try random vectors with weight 1 to max_weight
        best_vec = None
        best_score = -1
        
        for attempt in range(50):
            w = random.randint(1, min(max_weight, 8))
            positions = random.sample(range(max_weight), w)
            v = [0] * max_weight
            for p in positions:
                v[p] = 1
            
            # Check minimum distance to existing vectors
            min_dist = float('inf')
            for existing in vectors:
                dist = sum(1 for a, b in zip(v, existing) if a != b)
                min_dist = min(min_dist, dist)
            
            # Score: prefer higher minimum distance
            score = min_dist
            if score > best_score:
                best_score = score
                best_vec = v
        
        if best_vec is not None:
            vectors.append(best_vec)
    
    # Convert binary vectors to integers
    # Use positional encoding: bit i contributes 2^i
    for v in vectors:
        val = 0
        for i, bit in enumerate(v):
            if bit:
                val += (1 << i)
        S.add(val)
    
    return sorted(S)

def generate_sparse_binary_set(n_bits, num_vectors, deadline):
    """
    Generate a set from sparse binary vectors.
    
    Each vector has few 1s, ensuring limited carry propagation
    when adding elements.
    """
    if time.time() > deadline:
        return [0, 1]
    
    S = {0}
    
    # Generate sparse binary vectors
    vectors = []
    used_positions = set()
    
    for _ in range(num_vectors):
        if time.time() > deadline:
            break
        
        # Choose 1-3 positions to set to 1
        weight = random.randint(1, 3)
        available = [i for i in range(n_bits) if i not in used_positions]
        
        if not available:
            break
        
        positions = random.sample(available, min(weight, len(available)))
        
        v = [0] * n_bits
        for p in positions:
            v[p] = 1
            used_positions.add(p)
        
        vectors.append(v)
    
    # Convert to integers with non-power-of-2 spacing to avoid
    # perfect binary structure
    base_values = []
    for i in range(n_bits):
        # Use slightly perturbed powers of 2
        base = 1 << i
        # Add small perturbation based on position
        perturbation = random.randint(0, i) if i > 0 else 0
        base_values.append(base + perturbation)
    
    for v in vectors:
        val = 0
        for i, bit in enumerate(v):
            if bit:
                val += base_values[i]
        S.add(val)
    
    return sorted(S)

def generate_geometric_set(m, b, deadline):
    """Generate {0, 1, b, b^2, ...}"""
    U = [0, 1]
    cur = 1
    for i in range(m-1):
        cur = int(cur * b)
        if cur in U:
            # Collision, perturb
            cur += 1
            while cur in U:
                cur += 1
        U.append(cur)
    return sorted(set(U))

def construct():
    start = time.time()
    # Evaluator kills at 150s. Keep margin.
    deadline = start + 115
    
    best_set = [0, 1, 2, 4, 8]
    best_score = fast_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Explore small sets with local search
    # Sizes 3 to 10 seem promising.
    # Max value should be small to keep denominator small.
    # If n=5, max=8, log(17) ~ 2.8.
    # If n=10, max=500, log(1001) ~ 6.9.
    
    # We will run multiple restarts.
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Determine parameters for this restart
        # Mix of strategies
        strategy = random.random()
        
        if strategy < 0.3:
            # Random small set
            n = random.randint(3, 8)
            max_val = random.randint(10, 100)
            U = generate_random_set(n, max_val, deadline)
        elif strategy < 0.5:
            # Geometric-ish
            n = random.randint(3, 7)
            b = random.uniform(1.5, 3.0)
            U = generate_geometric_set(n, b, deadline)
            # Perturb
            for i in range(1, len(U)):
                if random.random() < 0.3:
                    U[i] += random.randint(-2, 2)
                    if U[i] < 0: U[i] = 0
            U = sorted(set(U))
        elif strategy < 0.7:
            # LDPC-inspired: sparse binary vectors
            n_bits = random.randint(6, 12)
            num_vectors = random.randint(3, 7)
            U = generate_sparse_binary_set(n_bits, num_vectors, deadline)
        elif strategy < 0.85:
            # LDPC-inspired: controlled weight vectors
            num_codewords = random.randint(4, 8)
            max_weight = random.randint(2, 5)
            U = generate_ldpc_inspired_set(num_codewords, max_weight, deadline)
        else:
            # Specific known good structure: {0, 1, 2, 4, ...} with perturbations
            n = random.randint(4, 6)
            U = [0, 1, 2, 4]
            for i in range(4, n):
                U.append(U[-1] * 2 + random.randint(-1, 1))
            U = sorted(set(U))
            # Ensure 0 is present and unique
            if 0 not in U: U.insert(0, 0)
            U = sorted(set(U))
            if len(U) < 2:
                U = [0, 1]

        # Ensure valid set
        if len(U) < 2 or 0 not in U:
            continue
            
        # Run local search
        improved_U, score = local_search(U, deadline, max_steps=50)
        
        if score > best_score:
            best_score = score
            best_set = improved_U
            
    # Phase 2: Final polish on the best set found
    # Use a larger step range and more steps
    if time.time() < deadline - 10:
        improved_U, score = local_search(best_set, deadline, max_steps=500, step_range=10)
        if score > best_score:
            best_score = score
            best_set = improved_U
            
    # One more deep polish if time permits
    if time.time() < deadline - 2:
        improved_U, score = local_search(best_set, deadline, max_steps=100, step_range=20)
        if score > best_score:
            best_score = score
            best_set = improved_U

    return best_set

if __name__ == "__main__":
    # Quick self-test
    U = construct()
    print(f"Best Set: {U}")
    print(f"Score: {fast_score(U)}")
