import numpy as np
import time
import random

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    h is a 1D array of non-negative floats.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    hh = np.convolve(h, h, mode='full')
    hh_max = np.max(hh)
    if hh_max < 1e-15:
        return 0.0
    return l2_sq / (l1 * hh_max)

def find_sidon_set(N, max_time=10.0):
    """
    Find a large Sidon set in range(N) using greedy heuristics.
    """
    best_set = []
    start_time = time.time()
    
    def is_sidon(S, k):
        """Check if k can be added to S while maintaining Sidon property."""
        # New sums are s+k for s in S, and k+k
        # We need to check if any of these collide with existing sums
        # Or if k+k collides
        # Efficient check:
        # 1. Check if k+k is in existing sums (sums of pairs from S)
        # 2. Check if any s+k is in existing sums
        # Note: Sums in Sidon set are usually defined as sums of distinct elements or all pairs.
        # Standard definition: All sums x_i + x_j (i <= j) are distinct.
        # Let's maintain a set of all sums i+j for i,j in S, i<=j.
        
        # This function is called frequently, so let's optimize.
        # We'll pass the set of sums as well to avoid recomputation.
        pass 

    # Better approach: Maintain the set of sums explicitly.
    def greedy_sidon(order):
        S = []
        sums = set()
        for k in order:
            # Check if adding k is valid
            valid = True
            # 1. Check k+k
            if k + k in sums:
                valid = False
            # 2. Check s+k for all s in S
            if valid:
                for s in S:
                    if s + k in sums:
                        valid = False
                        break
            
            if valid:
                S.append(k)
                # Add new sums
                sums.add(k + k)
                for s in S:
                    sums.add(s + k)
        
        return S

    # Try standard order
    best_set = greedy_sidon(list(range(N)))
    
    # Try random restarts
    for _ in range(100):
        if time.time() - start_time > max_time / 2:
            break
        order = list(range(N))
        random.shuffle(order)
        S = greedy_sidon(order)
        if len(S) > len(best_set):
            best_set = S
            
    # Local search: try to swap elements to increase size?
    # Or just try to add more if possible?
    # Since it's a set, we can't add if full.
    # We can try to remove an element and add two? Complex.
    # For now, greedy is usually good enough for N=50.
    
    return best_set

def optimize_weights(support, max_time=10.0):
    """
    Optimize the weights on the given support to maximize the ratio.
    Uses projected gradient ascent on the log-weights to ensure non-negativity and sum-to-1 (optional, ratio is scale invariant).
    Actually, since ratio is scale invariant, we can fix sum(weights) = 1.
    """
    K = len(support)
    if K == 0:
        return None
    
    # Initialize weights uniformly
    w = np.ones(K) / K
    
    # Gradient ascent on the ratio R(w)
    # R(w) = ||w||_2^2 / (||w||_1 * ||w*w||_inf)
    # Let S = ||w||_1. Fix S=1.
    # R(w) = sum(w_i^2) / max_k (sum_{i+j=k} w_i w_j)
    
    # To compute gradients, we need to know which index k achieves the max convolution value.
    # If multiple, it's non-smooth. We can use the subgradient or just pick one.
    # Let's use a simple iterative method:
    # 1. Compute conv
    # 2. Find max index k_max
    # 3. Compute gradient of numerator and denominator w.r.t w
    # 4. Update w in direction of increasing ratio
    # 5. Project back to simplex (non-negative, sum=1)
    
    h_full = np.zeros(N)
    h_full[support] = w
    
    best_w = w.copy()
    best_score = compute_ratio(h_full)
    start_time = time.time()
    
    lr = 0.1
    decay = 0.95
    
    for step in range(1000):
        if time.time() - start_time > max_time:
            break
        
        # Compute convolution
        conv = np.convolve(h_full, h_full, mode='full')
        max_val = np.max(conv)
        # Find indices where max is achieved (handle non-smoothness by averaging gradients or picking first)
        # For simplicity, pick the first max
        k_max = np.argmax(conv)
        
        # Numerator: N = sum w_i^2
        # dN/dw_i = 2 w_i
        
        # Denominator: D = conv[k_max] = sum_{i+j=k_max} w_i w_j
        # Note: support indices are in [0, N-1].
        # conv index k corresponds to sum of h[i] h[j] where i+j = k.
        # In our support, let support indices be s_0, ..., s_{K-1}.
        # We need to find pairs (a, b) in support such that s_a + s_b = k_max.
        
        # Gradient of D w.r.t w_i:
        # dD/dw_i = sum_{j: s_i + s_j = k_max} w_j
        # Note: if s_i + s_i = k_max, the term is 2*w_i (derivative of w_i^2 is 2w_i)
        # Wait, conv[k] = sum_{i=0}^N h[i] h[k-i].
        # If h is sparse, only non-zero terms matter.
        # Let's compute gradient numerically or analytically.
        # Analytical is faster.
        
        grad_D = np.zeros(K)
        for i_idx, s_i in enumerate(support):
            # Find j such that s_j = k_max - s_i
            # And j must be in support
            # We can use a dictionary for fast lookup
            pass # We'll build a dict outside loop for speed
            
        # Let's precompute a mapping from sum to list of pairs
        # This is static for the support
        pair_map = {}
        for i_idx in range(K):
            for j_idx in range(i_idx, K): # i<=j to avoid double counting? No, conv sums all i,j.
                # Actually conv[k] = sum_{i,j} h_i h_j delta(i+j-k)
                # So for a specific k, we sum over all pairs (i,j) in support with s_i+s_j=k.
                # The derivative of w_i w_j w.r.t w_a is:
                # 1 if a==i and a==j (i.e. i=j=a) -> 2 w_a
                # 1 if a==i and i!=j -> w_j
                # 1 if a==j and i!=j -> w_i
                # 0 otherwise
                
                s_sum = support[i_idx] + support[j_idx]
                if s_sum not in pair_map:
                    pair_map[s_sum] = []
                pair_map[s_sum].append((i_idx, j_idx))
        
        # Now compute grad_D for k_max
        if k_max in pair_map:
            for i_idx, j_idx in pair_map[k_max]:
                if i_idx == j_idx:
                    grad_D[i_idx] += 2.0 * w[i_idx]
                else:
                    grad_D[i_idx] += w[j_idx]
                    grad_D[j_idx] += w[i_idx]
        
        # Ratio R = N / D
        # dR/dw = (dN * D - N * dD) / D^2
        # We want to maximize R, so we move in direction of dR/dw.
        # grad_R = (2 * w * D - N * grad_D) / D^2
        # Let N = sum(w^2)
        N_val = np.sum(w * w)
        
        grad_R = (2.0 * w * max_val - N_val * grad_D) / (max_val * max_val)
        
        # Update w
        w_new = w + lr * grad_R
        
        # Project to simplex: non-negative and sum=1
        # Simple projection: clip to 0, then normalize
        w_new = np.clip(w_new, 0, None)
        sum_w = np.sum(w_new)
        if sum_w > 1e-10:
            w_new /= sum_w
        else:
            w_new = w # Fallback
        
        h_full[support] = w_new
        score = compute_ratio(h_full)
        
        if score > best_score:
            best_score = score
            best_w = w_new.copy()
            
        # Adaptive learning rate
        lr *= decay
        
    return best_w

def construct():
    deadline = time.time() + 140.0
    
    # Phase 1: Find a good Sidon set (or near-Sidon)
    # The current program finds a Sidon set. 
    # Sidon sets maximize the "spread" of sums, which minimizes the peak of the convolution for uniform weights.
    # For uniform weights on a Sidon set, conv peak is 1/K (since each sum is unique, max overlap is 1 pair? 
    # Wait. If Sidon, each sum i+j is unique. 
    # conv[k] = sum h_i h_j. If h_i = 1/K, then conv[k] = 1/K^2 for each unique sum.
    # There are K^2 pairs (i,j). But i+j=k. 
    # If Sidon, each k has at most 1 pair (i,j) with i<=j? No.
    # Sidon: x_i + x_j = x_k + x_l implies {i,j}={k,l}.
    # So for a specific sum value S, there is only one unordered pair {i,j}.
    # However, conv sums over ordered pairs? No, conv is sum_{i} h_i h_{k-i}.
    # If i != j, the pair (i,j) and (j,i) both contribute.
    # So if i+j=S and i!=j, conv[S] gets contributions from (i,j) and (j,i).
    # If i=j, conv[S] gets contribution from (i,i).
    # So for Sidon set with uniform weights 1/K:
    # If S = 2i (diagonal), conv[S] = (1/K)^2 = 1/K^2.
    # If S = i+j (i!=j), conv[S] = 2 * (1/K)^2 = 2/K^2.
    # So max conv is 2/K^2 (unless no off-diagonal sums exist, which is impossible for K>1).
    # Ratio = (K * (1/K)^2) / (1 * 2/K^2) = (1/K) / (2/K^2) = K/2.
    
    # So a Sidon set with uniform weights gives score K/2.
    # Max Sidon set in [0, N-1] has size approx sqrt(N).
    # For N=50, sqrt(50) ~ 7. So K=7 gives score 3.5.
    # Current score is 1.14. This suggests the current program's Sidon search might be failing or the score calculation is different?
    # Wait, the current program score is 1.1484.
    # If K=7, score should be 3.5.
    # Why is it 1.14?
    # Maybe the Sidon set found is small? Or maybe the "Sidon" definition in the code is too strict or wrong?
    # Or maybe the evaluator uses a different definition?
    # The task says "supremum norm of the autoconvolution".
    # Let's check the current program's logic.
    # It returns h[idx] = 1/K.
    # If K is small, say K=2, score = 1.
    # If K=1, score = 1.
    # If K=3, score = 1.5.
    # So if the current score is 1.14, it's likely finding a set with K=2 or effectively behaving like K=2?
    # Or maybe the Sidon search is broken.
    
    # Let's assume the Sidon search works and finds K~7.
    # Then we optimize weights.
    # Can we do better than K/2?
    # If we use non-uniform weights, we might reduce the peak further?
    # The peak is determined by the max number of pairs summing to the same index.
    # For a Sidon set, the max number of pairs is 1 (or 2 ordered).
    # We cannot reduce this below 1 pair for any sum that exists.
    # However, we can choose weights to minimize the *value* of the peak.
    # If the peak comes from a pair (i,j) with i!=j, value is 2 w_i w_j.
    # If we make w_i and w_j small, we reduce the peak.
    # But we must keep sum w_i = 1.
    # This is a continuous optimization problem.
    
    # Let's try to find a better Sidon set first.
    sidon_set = find_sidon_set(N, max_time=5.0)
    K = len(sidon_set)
    
    if K == 0:
        # Fallback
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
    
    # Optimize weights on this support
    w = optimize_weights(sidon_set, max_time=10.0)
    
    h = np.zeros(N)
    h[sidon_set] = w
    
    score = compute_ratio(h)
    
    # If we have time left, try other supports?
    # E.g. try to add one more element if it doesn't increase peak too much?
    # Or try different Sidon sets.
    
    if time.time() < deadline - 10.0:
        # Try a few random restarts of Sidon search + weight optimization
        best_score = score
        best_h = h.copy()
        
        for _ in range(10):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb the Sidon set? Or find a new one?
            # Let's just try a new random Sidon search
            random_seed = random.randint(0, 10000)
            random.seed(random_seed)
            new_sidon = find_sidon_set(N, max_time=2.0)
            if len(new_sidon) == 0:
                continue
                
            new_w = optimize_weights(new_sidon, max_time=5.0)
            if new_w is None:
                continue
                
            new_h = np.zeros(N)
            new_h[new_sidon] = new_w
            new_score = compute_ratio(new_h)
            
            if new_score > best_score:
                best_score = new_score
                best_h = new_h
        
        return list(best_h)
    
    return list(h)

# Test locally
if __name__ == "__main__":
    h = construct()
    print(f"Score: {compute_ratio(h)}")
    print(f"Support size: {np.sum(np.array(h) > 1e-9)}")
