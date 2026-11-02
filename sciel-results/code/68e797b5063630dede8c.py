"""
Program for minimizing the normalized maximum of autoconvolution of a
nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Approach:
- The uniform (box) function gives a triangular autoconvolution with peak 1.
- We use a Greedy Combinatorial Construction to find a sparse set of indices
  such that the sums of pairs of indices (which determine the autoconvolution
  peaks) have minimal collisions.
- By keeping the function sparse (few non-zero intervals), we reduce the
  magnitude of the convolution peaks, as the energy is spread out in a way
  that avoids constructive interference at the center.
- We greedily select indices one by one, choosing the index that minimizes
  the maximum count of any sum of pairs.
- We test several sparsity levels (k) and return the best configuration.
"""

import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    def greedy_construct(k):
        indices = []
        sum_counts = np.zeros(2 * N, dtype=np.int32)
        current_max = 0
        
        for step in range(k):
            best_i = -1
            best_max_val = np.inf
            
            # Vectorize the evaluation of all possible i
            # For each i, the new sums are i + j for j in indices
            # We need max(current_max, max(sum_counts[i+j] + 1))
            
            # If indices is empty, max is 1 (self-sum i+i? No, autoconv includes i*i)
            # Wait, autoconv sum is sum_{i,j} h_i h_j delta_{m, i+j}.
            # If we add h_i, the new terms are h_i h_j and h_j h_i.
            # Since h is symmetric in the sum, we just add 1 to count[i+j] for all existing j.
            # And also 1 to count[i+i] for the self-term?
            # Yes, the term h_i * h_i contributes to sum 2i.
            # So for each existing j (including the new i itself? No, we add i to the set).
            # When adding i:
            # For each j in indices (old), add 2 to sum_counts[i+j]? 
            # No.
            # (sum + h_i)^2 = sum^2 + 2 h_i sum + h_i^2.
            # The cross terms are 2 * h_i * h_j.
            # The self term is h_i^2.
            # So for each j in indices (old), sum_counts[i+j] += 2.
            # And sum_counts[2*i] += 1.
            
            # Let's re-evaluate the cost.
            # We want to minimize the max value in sum_counts after update.
            
            if len(indices) == 0:
                # First element: sum_counts[2*i] becomes 1. Max is 1.
                # Any i is equivalent.
                best_i = 0 
                indices.append(best_i)
                sum_counts[2 * best_i] += 1
                current_max = 1
                continue
            
            # For each candidate i in 0..N-1:
            # new_counts = sum_counts.copy()
            # new_counts[i + np.array(indices)] += 2
            # new_counts[2*i] += 1
            # candidate_max = np.max(new_counts)
            
            # Vectorized approach:
            # We need to find i that minimizes max( current_max, max(sum_counts[i+indices]+2), sum_counts[2*i]+1 )
            # Note: current_max might be invalid if the max was at a position that gets updated?
            # No, we only increase values. So the new max is at least current_max.
            # The new max is max( current_max, max_{j in indices} (sum_counts[i+j] + 2), sum_counts[2*i] + 1 )
            
            # Precompute indices array
            idx_arr = np.array(indices)
            
            # For all i, we want sum_counts[i + idx_arr]
            # This is a matrix of size N x k
            # sums = idx_arr[None, :] + np.arange(N)[:, None]
            # vals = sum_counts[sums]
            # max_cross = np.max(vals + 2, axis=1)
            
            # And self term: sum_counts[2*np.arange(N)] + 1
            # max_self = sum_counts[2*np.arange(N)] + 1
            
            # total_max = np.maximum.reduce([np.full(N, current_max), max_cross, max_self])
            # best_i = np.argmin(total_max)
            
            # To save memory, we can do this in chunks or just loop if k is small.
            # k is small (<=100). N=600.
            # Matrix 600x100 is 60k elements. Very fast.
            
            i_range = np.arange(N)
            sums_matrix = i_range[:, None] + idx_arr[None, :]
            cross_vals = sum_counts[sums_matrix] + 2
            max_cross = np.max(cross_vals, axis=1)
            
            self_vals = sum_counts[2 * i_range] + 1
            
            # The new max is the max of the old max (if not touched) and the new values.
            # Since we only increase, the new max is max(old_max, new_values).
            # However, if the old_max was at a position that is NOT in sums_matrix or 2*i_range, it remains.
            # So candidate_max = max(current_max, max_cross[i], self_vals[i])
            
            candidate_max = np.maximum(current_max, np.maximum(max_cross, self_vals))
            
            best_i = np.argmin(candidate_max)
            best_max_val = candidate_max[best_i]
            
            # Update state
            indices.append(best_i)
            sum_counts[best_i + idx_arr] += 2
            sum_counts[2 * best_i] += 1
            current_max = best_max_val
            
        h = np.zeros(N)
        h[indices] = 1.0 / np.sqrt(k)
        return h

    best_h = None
    best_score = np.inf
    
    # Try different sparsity levels
    for k in [5, 10, 20, 50, 100]:
        h = greedy_construct(k)
        # Verify with actual convolution
        ac = np.convolve(h, h, mode='full')
        score = np.max(ac)
        
        if score < best_score:
            best_score = score
            best_h = h
            
    return list(best_h)
# EVOLVE-BLOCK-END
