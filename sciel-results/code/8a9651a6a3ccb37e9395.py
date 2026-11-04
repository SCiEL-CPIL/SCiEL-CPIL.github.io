import numpy as np
import random

N = 400

def construct():
    # The task is to minimize the upper bound on C3 (autoconvolution peak)
    # for a signed step function with 400 intervals on [-1/4, 1/4].
    # This is equivalent to finding a sparse vector x in R^N such that
    # ||x||_2 = 1 and max_k |(x * x)[k]| is minimized.
    #
    # Strategy:
    # 1. Select a sparse support S of size M.
    # 2. Assign values to x such that ||x||_2 = 1.
    # 3. Optimize the support S to minimize the collision count in differences.
    #    Ideally, we want a Sidon set where all differences s_i - s_j are distinct
    #    (except for s_i - s_i = 0 and s_i - s_j = -(s_j - s_i)).
    #    If S is a Sidon set, the maximum off-peak convolution value is 1 (in units of 1/M^2 if unnormalized,
    #    but we normalize to L2=1).
    #
    # Let x[s] = 1/sqrt(M) for s in S.
    # Then (x*x)[d] = sum_{s_i - s_j = d} x[s_i] x[s_j] = (1/M) * (number of pairs (i,j) with s_i - s_j = d).
    # For d=0, the sum is M * (1/M) = 1.
    # For d != 0, if S is Sidon, the number of pairs is at most 1 (actually exactly 1 if the difference exists in the set of differences).
    # Wait, for a Sidon set, all differences s_i - s_j for i != j are distinct?
    # No, a Sidon set (B_2 sequence) means all sums s_i + s_j are distinct.
    # This implies that if s_i + s_j = s_k + s_l, then {i,j}={k,l}.
    # This is equivalent to saying that the differences s_i - s_j are distinct for i != j?
    # Let's check. If s_i - s_j = s_k - s_l, then s_i + s_l = s_k + s_j.
    # So yes, distinct differences (for i!=j) is equivalent to Sidon property regarding sums.
    # However, we must be careful with signs. s_i - s_j = d and s_j - s_i = -d.
    # So for a given d != 0, there can be at most one pair (i,j) such that s_i - s_j = d.
    # Thus, the max off-peak value is 1/M.
    #
    # We want to maximize M to minimize 1/M.
    # The maximum size of a Sidon set in Z_N (or integers 0..N-1) is approx sqrt(N).
    # For N=400, sqrt(400) = 20. So M=20 is a good target.
    #
    # We will use a randomized greedy algorithm to find a large Sidon set.
    # We try multiple random permutations of indices and greedily add indices
    # if they maintain the Sidon property.

    def is_sidon_candidate(S, new_elem):
        # Check if adding new_elem to S maintains the Sidon property.
        # We need to ensure that no new difference collides with existing differences.
        # Existing differences: {s_i - s_j for i, j in S}
        # New differences: {new_elem - s for s in S} and {s - new_elem for s in S}
        # We need these new differences to not be in the existing difference set.
        # Note: We don't care about 0 (new_elem - new_elem), but new_elem is not in S yet.
        # The differences involving new_elem are new_elem - s and s - new_elem.
        
        diff_set = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    diff_set.add(S[i] - S[j])
        
        for s in S:
            d1 = new_elem - s
            d2 = s - new_elem
            if d1 in diff_set or d2 in diff_set:
                return False
        return True

    def find_sidon_set(target_size, n_trials=100):
        best_set = []
        for _ in range(n_trials):
            S = []
            # Random permutation of indices
            indices = list(range(N))
            random.shuffle(indices)
            
            for idx in indices:
                if len(S) >= target_size:
                    break
                # Check if idx can be added
                # Optimization: Instead of rebuilding diff_set every time,
                # we can maintain it incrementally, but for small M (<=20), rebuilding is fast.
                if is_sidon_candidate(S, idx):
                    S.append(idx)
            
            if len(S) > len(best_set):
                best_set = S[:]
                if len(best_set) == target_size:
                    break
        
        return best_set

    # Try to find a Sidon set of size 20
    target_M = 20
    support = find_sidon_set(target_M, n_trials=50)
    
    # If we didn't find a set of size 20, try smaller sizes or just use what we have
    M = len(support)
    
    if M == 0:
        # Fallback: random sparse set
        support = list(random.sample(range(N), 10))
        M = 10
    
    # Construct the vector x
    x = np.zeros(N)
    x[support] = 1.0 / np.sqrt(M)
    
    return x
